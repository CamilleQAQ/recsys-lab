"""重新评测已固定的 checkpoint，并保存可复现的最终结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from recsys.experiments.compare import compare_models, print_comparison_table
from recsys.experiments.compare_sasrec import compare_with_sasrec
from recsys.config import SPLIT_STRATEGY
from recsys.data.movielens import prepare_movielens
from recsys.evaluation.sasrec_evaluator import evaluate_sasrec_model, load_sasrec_model
from recsys.training.trainer import resolve_device


def checkpoint_sha256(path: Path) -> str:
    """分块计算 checkpoint 的 SHA-256，用于记录「结果对应哪个文件」。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_training_config(checkpoint: dict, config_key: str) -> dict:
    """读取 checkpoint 里保存的训练配置。

    PyTorch 训练模块把它存成 ``train_config``，
    而 Two-Tower 的调参实验把同类内容存成 ``experiment_config``。
    """
    return checkpoint[config_key]


def get_two_tower_metadata(
    checkpoint: dict,
    evaluated_validation_metrics: dict[str, float],
) -> tuple[dict, int | None, dict[str, float], str]:
    """兼容调参脚本和普通训练脚本保存的两种 checkpoint。"""
    if "experiment_config" in checkpoint:
        training_config = checkpoint["experiment_config"]
        selection_note = "selected by validation Recall@20"
    elif "train_config" in checkpoint:
        training_config = checkpoint["train_config"]
        selection_note = "selected by validation Recall@20" if checkpoint.get("best_epoch") else "fixed-epoch training"
    else:
        raise ValueError("Two-Tower checkpoint has no saved training configuration")

    best_epoch = checkpoint.get("best_epoch", training_config.get("epochs"))
    validation_metrics = checkpoint.get(
        "best_validation_metrics", evaluated_validation_metrics
    )
    return training_config, best_epoch, validation_metrics, selection_note


def save_json(path: Path, payload: dict) -> None:
    """把结果写成 UTF-8 的 JSON，中文不转义、缩进两个空格。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def finalize_experiment(
    data_dir: Path,
    artifacts_dir: Path,
    device_name: str = "auto",
) -> dict:
    """全程不训练，只做评测，并把指标与 checkpoint 元信息保存下来。"""
    import faiss
    from recsys.evaluation.faiss_evaluator import run_faiss_evaluation
    seed = 42
    ks = (10, 20)
    device = resolve_device(device_name)
    # 每个数据集的 checkpoint 有自己的子目录（kuairec / movielens），
    # 这样在 KuaiRec 上重训不会覆盖已经公布的 MovieLens 交付物。
    movielens_artifacts_dir = artifacts_dir / "movielens"
    two_tower_path = movielens_artifacts_dir / "two_tower.pt"
    mf_path = movielens_artifacts_dir / "matrix_factorization.pt"
    sasrec_path = movielens_artifacts_dir / "sasrec.pt"
    prepared_data = prepare_movielens(data_dir)

    validation_results = compare_models(
        data_dir=data_dir,
        two_tower_checkpoint=two_tower_path,
        mf_checkpoint=mf_path,
        split="validation",
        ks=ks,
        device_name=device_name,
        print_results=False,
    )
    sasrec_model, sasrec_checkpoint = load_sasrec_model(sasrec_path, device)
    validation_results["SASRec"] = evaluate_sasrec_model(
        sasrec_model,
        prepared_data,
        split="validation",
        ks=ks,
        batch_size=512,
        device=device,
    )
    print_comparison_table(validation_results, ks, "Locked-checkpoint validation")

    test_results = compare_with_sasrec(
        data_dir=data_dir,
        two_tower_checkpoint=two_tower_path,
        mf_checkpoint=mf_path,
        sasrec_checkpoint=sasrec_path,
        device_name=device_name,
    )
    # 历史长度对齐的受控对比：同一个 SASRec checkpoint，改用「只有训练历史」的
    # 测试输入，这样才和只用 ID 的基线处在同一起跑线上。
    controlled_metrics = evaluate_sasrec_model(
        sasrec_model,
        prepared_data,
        split="test",
        ks=ks,
        batch_size=512,
        device=device,
        include_validation_in_test_history=False,
    )
    print(
        "SASRec controlled test (train history only): "
        + " ".join(
            f"{name}={value:.6f}" for name, value in controlled_metrics.items()
        )
    )

    faiss_results = run_faiss_evaluation(
        checkpoint_path=two_tower_path,
        data_dir=data_dir,
        split="test",
        ks=ks,
        seed=seed,
        device_name=device_name,
    )

    two_tower_checkpoint = torch.load(
        two_tower_path, map_location="cpu", weights_only=True
    )
    mf_checkpoint = torch.load(mf_path, map_location="cpu", weights_only=True)
    (
        two_tower_training_config,
        two_tower_best_epoch,
        two_tower_validation_metrics,
        two_tower_selection_note,
    ) = get_two_tower_metadata(
        two_tower_checkpoint, validation_results["Two-Tower"]
    )
    two_tower_config = {
        "checkpoint": two_tower_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256(two_tower_path),
        "split_strategy": two_tower_checkpoint["split_strategy"],
        "random_seed": two_tower_training_config["seed"],
        "model_config": two_tower_checkpoint["model_config"],
        "training_config": two_tower_training_config,
        "best_epoch": two_tower_best_epoch,
        "best_validation_metrics": two_tower_validation_metrics,
        "selection_note": two_tower_selection_note,
    }
    sasrec_config = {
        "checkpoint": sasrec_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256(sasrec_path),
        "split_strategy": sasrec_checkpoint["split_strategy"],
        "random_seed": get_training_config(sasrec_checkpoint, "train_config")["seed"],
        "model_config": sasrec_checkpoint["model_config"],
        "training_config": get_training_config(sasrec_checkpoint, "train_config"),
        "best_epoch": sasrec_checkpoint["best_epoch"],
        "best_validation_metrics": sasrec_checkpoint["best_validation_metrics"],
    }
    mf_config = {
        "checkpoint": mf_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256(mf_path),
        "split_strategy": mf_checkpoint["split_strategy"],
        "random_seed": get_training_config(mf_checkpoint, "train_config")["seed"],
        "model_config": mf_checkpoint["model_config"],
        "training_config": get_training_config(mf_checkpoint, "train_config"),
        "trained_epochs": get_training_config(mf_checkpoint, "train_config")["epochs"],
        "best_epoch": mf_checkpoint.get("best_epoch"),
        "selection_note": "selected by validation Recall@20",
    }

    # MovieLens 的时间戳只精确到秒，这里统计有多少用户的验证集与测试集交互
    # 落在同一秒内 —— 这类用户先后顺序无法从数据本身判定。
    validation_by_user = {
        interaction.user_idx: interaction
        for interaction in prepared_data.validation_interactions
    }
    same_timestamp_count = sum(
        validation_by_user[interaction.user_idx].timestamp == interaction.timestamp
        for interaction in prepared_data.test_interactions
    )
    train_item_indices = {
        interaction.item_idx for interaction in prepared_data.train_interactions
    }
    validation_item_indices = {
        interaction.item_idx
        for interaction in prepared_data.validation_interactions
    }
    test_item_indices = {
        interaction.item_idx for interaction in prepared_data.test_interactions
    }

    # 下面这几份切分规模会被多次引用，先算好避免重复计算。
    num_train_interactions = len(prepared_data.train_interactions)
    num_validation_interactions = len(prepared_data.validation_interactions)
    num_test_interactions = len(prepared_data.test_interactions)
    observed_in_train = len(train_item_indices)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "faiss": faiss.__version__,
            "device": str(device),
        },
        "random_seed": seed,
        "dataset": {
            "name": "MovieLens-1M",
            "positive_rating_threshold": 4,
            "num_users": prepared_data.num_users,
            "num_items": prepared_data.num_items,
            "positive_interactions": (
                num_train_interactions
                + num_validation_interactions
                + num_test_interactions
            ),
            "train_interactions": num_train_interactions,
            "validation_interactions": num_validation_interactions,
            "test_interactions": num_test_interactions,
            "validation_test_same_timestamp_users": same_timestamp_count,
            "items_observed_in_train": observed_in_train,
            "items_absent_from_train": prepared_data.num_items - observed_in_train,
            "validation_target_items_absent_from_train": len(
                validation_item_indices - train_item_indices
            ),
            "test_target_items_absent_from_train": len(
                test_item_indices - train_item_indices
            ),
        },
        "evaluation_protocol": {
            "split_strategy": SPLIT_STRATEGY,
            "ordering": "strict train < validation < test timestamps; ambiguous users remain in train",
            "catalog_definition": "all positive-interaction item IDs before splitting",
            "validation_use": "hyperparameter selection and checkpoint selection only",
            "test_use": "evaluation after configurations and checkpoints are locked",
            "id_only_seen_filter": "all observed item exposures strictly before target time (including validation at test time)",
            "sasrec_validation_history": "train interactions",
            "sasrec_standard_test_history": "train + validation interactions",
            "sasrec_standard_test_seen_filter": "all observed item exposures strictly before target time",
            "target_in_model_input": False,
            "determinism_note": (
                "seeded Python and PyTorch; exact GPU bitwise "
                "reproducibility is not guaranteed"
            ),
        },
        "models": {
            "Popularity": {
                "random_seed": None,
                "model_config": {
                    "ranking": "descending train interaction count",
                },
                "best_epoch": None,
                "validation_metrics": validation_results["Popularity"],
                "test_metrics": test_results["Popularity"],
            },
            "MF": {
                "random_seed": mf_config["random_seed"],
                "model_config": mf_config["model_config"],
                "training_config": mf_config["training_config"],
                "best_epoch": mf_config["best_epoch"],
                "selection_note": mf_config["selection_note"],
                "validation_metrics": validation_results["MF"],
                "test_metrics": test_results["MF"],
            },
            "Two-Tower": {
                "random_seed": two_tower_config["random_seed"],
                "model_config": two_tower_config["model_config"],
                "training_config": two_tower_config["training_config"],
                "best_epoch": two_tower_config["best_epoch"],
                "validation_metrics": validation_results["Two-Tower"],
                "test_metrics": test_results["Two-Tower"],
            },
            "SASRec": {
                "random_seed": sasrec_config["random_seed"],
                "model_config": sasrec_config["model_config"],
                "training_config": sasrec_config["training_config"],
                "best_epoch": sasrec_config["best_epoch"],
                "validation_metrics": validation_results["SASRec"],
                "test_metrics": test_results["SASRec"],
            },
        },
        "supplementary_controlled_evaluation": {
            "name": "SASRec train-history-only test",
            "purpose": (
                "train-only SASRec input with the same test-time seen-item filter"
            ),
            "metrics": controlled_metrics,
        },
        "faiss_index_flat_ip": faiss_results,
    }
    save_json(movielens_artifacts_dir / "two_tower_config.json", two_tower_config)
    save_json(movielens_artifacts_dir / "mf_config.json", mf_config)
    save_json(movielens_artifacts_dir / "sasrec_config.json", sasrec_config)
    save_json(artifacts_dir / "results.json", results)
    print(f"Saved final results to {artifacts_dir / 'results.json'}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finalize_experiment(args.data_dir, args.artifacts_dir, args.device)
