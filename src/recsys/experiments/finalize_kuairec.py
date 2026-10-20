"""评测 KuaiRec 上的 Popularity / MF / Two-Tower，并保存可复现的结果。

这是 ``finalize_experiment`` 的 KuaiRec 版本：先只用验证集选配置，
锁定 checkpoint 之后再跑一次测试集。测试集数字只在这里产生一次。

用法：

    python -m recsys.experiments.finalize_kuairec --prepare-only
    python -m recsys.experiments.finalize_kuairec --split validation
    python -m recsys.experiments.finalize_kuairec --split test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from recsys.config import KUAIREC_WATCH_RATIO_THRESHOLD, SPLIT_STRATEGY
from recsys.data.loader import (
    prepare_dataset,
    resolve_checkpoint_dataset,
    resolve_positive_threshold,
)
from recsys.data.schema import PreparedData, seen_items_for_split
from recsys.evaluation.evaluator import (
    evaluate_embedding_model,
    evaluate_recommendations,
    get_evaluation_interactions,
    group_items_by_user,
    load_model,
)
from recsys.evaluation.sasrec_evaluator import (
    evaluate_sasrec_model,
    load_sasrec_model,
)
from recsys.experiments.compare import load_mf_model, print_comparison_table
from recsys.models.popularity import (
    generate_popularity_recommendations,
    rank_items_by_popularity,
)
from recsys.training.trainer import resolve_device


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(counts: list[int]) -> dict[str, float | int]:
    values = np.asarray(counts, dtype=np.int64)
    if values.size == 0:
        return {}
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "min": int(values.min()),
        "mean": float(values.mean()),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": int(values.max()),
    }


def describe_split(prepared_data: PreparedData) -> dict:
    """切分规模与几个容易出错的量：冷启动物品、同时间戳并列、剩余候选数。"""
    train_items_by_user = group_items_by_user(prepared_data.train_interactions)
    test_seen_items_by_user = seen_items_for_split(prepared_data, "test")
    validation_items_by_user = group_items_by_user(
        prepared_data.validation_interactions
    )
    test_items_by_user = group_items_by_user(prepared_data.test_interactions)
    # group_items_by_user 只留物品集合，时间戳要单独收一份。
    validation_timestamps_by_user: dict[int, set[float]] = defaultdict(set)
    for interaction in prepared_data.validation_interactions:
        validation_timestamps_by_user[interaction.user_idx].add(
            interaction.timestamp
        )
    train_item_indices = {
        interaction.item_idx for interaction in prepared_data.train_interactions
    }
    validation_items = {
        interaction.item_idx for interaction in prepared_data.validation_interactions
    }
    test_items = {
        interaction.item_idx for interaction in prepared_data.test_interactions
    }

    per_user_train = [
        len(train_items_by_user.get(index, set()))
        for index in range(prepared_data.num_users)
    ]
    per_item_train = [0] * prepared_data.num_items
    for interaction in prepared_data.train_interactions:
        per_item_train[interaction.item_idx] += 1

    candidate_counts = []
    for user_idx in sorted(test_items_by_user):
        seen = test_seen_items_by_user.get(user_idx, set())
        candidate_counts.append(prepared_data.num_items - len(seen))

    same_timestamp = 0
    for interaction in prepared_data.test_interactions:
        if interaction.timestamp in validation_timestamps_by_user.get(
            interaction.user_idx, set()
        ):
            same_timestamp += 1

    return {
        "name": "KuaiRec 2.0 big_matrix",
        "positive_rule": "first exposure watch_ratio >= threshold (inclusive)",
        "positive_threshold": KUAIREC_WATCH_RATIO_THRESHOLD,
        "duplicate_handling": (
            "first exposure per (user, video); later replays excluded"
        ),
        "num_users": prepared_data.num_users,
        "num_items": prepared_data.num_items,
        "positive_interactions": (
            len(prepared_data.train_interactions)
            + len(prepared_data.validation_interactions)
            + len(prepared_data.test_interactions)
        ),
        "train_interactions": len(prepared_data.train_interactions),
        "validation_interactions": len(prepared_data.validation_interactions),
        "test_interactions": len(prepared_data.test_interactions),
        "density_of_train": len(prepared_data.train_interactions)
        / (prepared_data.num_users * prepared_data.num_items),
        "evaluation_users": len(test_items_by_user),
        "train_interactions_per_user": _distribution(per_user_train),
        "train_interactions_per_item": _distribution(per_item_train),
        "items_observed_in_train": len(train_item_indices),
        "items_absent_from_train": prepared_data.num_items - len(train_item_indices),
        "validation_target_items_absent_from_train": len(
            validation_items - train_item_indices
        ),
        "test_target_items_absent_from_train": len(test_items - train_item_indices),
        "validation_test_same_timestamp_users": same_timestamp,
        "validation_test_same_timestamp_ratio": (
            same_timestamp / len(prepared_data.test_interactions)
        ),
        "candidate_items_per_user_after_filter": _distribution(candidate_counts),
    }


def evaluate_popularity(
    prepared_data: PreparedData, split: str, ks: tuple[int, ...]
) -> dict[str, float]:
    evaluation_interactions = get_evaluation_interactions(prepared_data, split)
    evaluation_users = sorted(group_items_by_user(evaluation_interactions))
    train_items_by_user = seen_items_for_split(prepared_data, split)
    popularity_ranking = rank_items_by_popularity(
        prepared_data.train_interactions, prepared_data.num_items
    )
    recommendations = generate_popularity_recommendations(
        user_indices=evaluation_users,
        popularity_ranking=popularity_ranking,
        train_items_by_user=train_items_by_user,
        k=max(ks),
    )
    return evaluate_recommendations(recommendations, evaluation_interactions, ks)


def evaluate_models(
    prepared_data: PreparedData,
    checkpoint_dir: Path,
    split: str,
    ks: tuple[int, ...] = (10, 20),
    user_batch_size: int = 512,
    device_name: str = "auto",
    print_results: bool = True,
    include_sasrec: bool = True,
) -> dict[str, dict[str, float]]:
    """所有已固定的模型走同一套「先过滤目标前已曝光物品、再全量排序」的评测。

    SASRec 是序列模型，它的标准协议会把 validation 交互也算进测试历史，
    因此和 ID-only 的模型不可直接比较。``evaluate_sasrec_controlled``
    另外给一份 train-history-only 输入对照；已看物品的过滤规则保持不变。
    """
    device = resolve_device(device_name)
    results = {"Popularity": evaluate_popularity(prepared_data, split, ks)}

    mf_model, mf_checkpoint = load_mf_model(
        checkpoint_dir / "matrix_factorization.pt", device
    )
    two_tower_model, two_tower_checkpoint = load_model(
        checkpoint_dir / "two_tower.pt", device
    )
    for name, checkpoint in (
        ("MF", mf_checkpoint),
        ("Two-Tower", two_tower_checkpoint),
    ):
        if resolve_checkpoint_dataset(checkpoint) != "kuairec":
            raise ValueError(f"{name} checkpoint was not trained on KuaiRec")
        model_config = checkpoint["model_config"]
        if model_config["num_users"] != prepared_data.num_users:
            raise ValueError(f"{name} user vocabulary does not match the data")
        if model_config["num_items"] != prepared_data.num_items:
            raise ValueError(f"{name} item vocabulary does not match the data")

    results["MF"] = evaluate_embedding_model(
        mf_model, prepared_data, split, ks, user_batch_size, device
    )
    results["Two-Tower"] = evaluate_embedding_model(
        two_tower_model, prepared_data, split, ks, user_batch_size, device
    )

    sasrec_path = checkpoint_dir / "sasrec.pt"
    if include_sasrec and sasrec_path.exists():
        sasrec_model, sasrec_checkpoint = load_sasrec_model(sasrec_path, device)
        if resolve_checkpoint_dataset(sasrec_checkpoint) != "kuairec":
            raise ValueError("SASRec checkpoint was not trained on KuaiRec")
        if sasrec_checkpoint["model_config"]["num_items"] != prepared_data.num_items:
            raise ValueError("SASRec item vocabulary does not match the data")
        results["SASRec"] = evaluate_sasrec_model(
            model=sasrec_model,
            prepared_data=prepared_data,
            split=split,
            ks=ks,
            batch_size=user_batch_size,
            device=device,
        )

    if print_results:
        print_comparison_table(results, ks, f"KuaiRec evaluation split: {split}")
    return results


def evaluate_sasrec_controlled(
    prepared_data: PreparedData,
    checkpoint_dir: Path,
    split: str,
    ks: tuple[int, ...] = (10, 20),
    batch_size: int = 512,
    device_name: str = "auto",
) -> dict[str, float]:
    """同一个 SASRec checkpoint，改用只有训练历史的输入作受控对照。"""
    sasrec_path = checkpoint_dir / "sasrec.pt"
    if not sasrec_path.exists():
        return {}
    device = resolve_device(device_name)
    sasrec_model, _ = load_sasrec_model(sasrec_path, device)
    return evaluate_sasrec_model(
        model=sasrec_model,
        prepared_data=prepared_data,
        split=split,
        ks=ks,
        batch_size=batch_size,
        device=device,
        include_validation_in_test_history=False,
    )


def _model_config_block(
    name: str, checkpoint_path: Path, metrics: dict[str, float]
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_config = checkpoint.get("train_config", {})
    return {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "dataset": resolve_checkpoint_dataset(checkpoint),
        "positive_threshold": resolve_positive_threshold(
            resolve_checkpoint_dataset(checkpoint),
            checkpoint.get("positive_threshold", train_config.get("positive_threshold")),
        ),
        "random_seed": train_config.get("seed"),
        "model_config": checkpoint["model_config"],
        "training_config": train_config,
        "epoch_losses": checkpoint.get("epoch_losses"),
        "best_epoch": checkpoint.get("best_epoch"),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="默认 artifacts/kuairec",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="先用 validation 选配置，锁定后再用 test 跑一次",
    )
    parser.add_argument("--user-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只准备数据并写数据规模，不加载模型",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or (args.artifacts_dir / "kuairec")
    device = resolve_device(args.device)
    prepared_data = prepare_dataset(
        dataset="kuairec",
        data_dir=args.data_dir,
    )
    split_description = describe_split(prepared_data)
    print(json.dumps(split_description, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    ks = (10, 20)
    metrics = evaluate_models(
        prepared_data=prepared_data,
        checkpoint_dir=checkpoint_dir,
        split=args.split,
        ks=ks,
        user_batch_size=args.user_batch_size,
        device_name=args.device,
    )
    controlled = evaluate_sasrec_controlled(
        prepared_data=prepared_data,
        checkpoint_dir=checkpoint_dir,
        split=args.split,
        ks=ks,
        batch_size=args.user_batch_size,
        device_name=args.device,
    )
    if controlled:
        print(
            "SASRec controlled (train history only): "
            + " ".join(f"{name}={value:.6f}" for name, value in controlled.items())
        )

    results_path = args.artifacts_dir / "kuairec_results.json"
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if payload.get("evaluation_protocol", {}).get("split_strategy") != SPLIT_STRATEGY:
            raise ValueError(
                "Existing results use another split protocol; archive them before finalizing"
            )
    else:
        payload = {
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": str(device),
            },
            "evaluation_protocol": {
                "split_strategy": SPLIT_STRATEGY,
                "ordering": "strict train < validation < test timestamps; ambiguous users remain in train",
                "positive_rule": "first exposure watch_ratio >= threshold; later replays cannot relabel or move the event",
                "id_only_seen_filter": "all observed item exposures strictly before target time (including validation at test time)",
                "validation_use": "configuration and checkpoint selection only",
                "test_use": "one locked run after the configuration is frozen",
                "target_in_model_input": False,
            },
            "dataset": split_description,
            "runs": {},
        }
    payload["dataset"] = split_description
    payload["runs"][args.split] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    if controlled:
        payload["runs"][args.split]["sasrec_train_history_only"] = controlled
        payload["evaluation_protocol"]["sasrec_standard_history"] = (
            "train + validation interactions"
        )
        payload["evaluation_protocol"]["sasrec_controlled_history"] = (
            "train-only model input; test-time seen filter is unchanged"
        )
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {results_path}")

    checkpoint_files = [
        ("MF", "matrix_factorization.pt"),
        ("Two-Tower", "two_tower.pt"),
    ]
    if "SASRec" in metrics:
        checkpoint_files.append(("SASRec", "sasrec.pt"))
    model_configs = {
        name: _model_config_block(name, checkpoint_dir / filename, metrics[name])
        for name, filename in checkpoint_files
    }
    for name, config_block in model_configs.items():
        for evaluated_split in ("validation", "test"):
            run = payload["runs"].get(evaluated_split, {})
            if name in run.get("metrics", {}):
                config_block[f"{evaluated_split}_metrics"] = run["metrics"][name]
    if controlled:
        model_configs["SASRec"]["train_history_only_metrics"] = controlled
    configs_path = args.artifacts_dir / "kuairec_model_configs.json"
    configs_path.write_text(
        json.dumps(model_configs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {configs_path}")


if __name__ == "__main__":
    main()
