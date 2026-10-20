"""用完全相同的方式评测 Popularity、矩阵分解和 Two-Tower。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from recsys.config import SPLIT_STRATEGY
from recsys.data.loader import (
    prepare_dataset,
    resolve_checkpoint_dataset,
    resolve_positive_threshold,
)
from recsys.data.schema import seen_items_for_split
from recsys.evaluation.evaluator import (
    evaluate_embedding_model,
    evaluate_recommendations,
    get_evaluation_interactions,
    group_items_by_user,
    load_model,
)
from recsys.models.matrix_factorization import MatrixFactorization
from recsys.models.popularity import (
    generate_popularity_recommendations,
    rank_items_by_popularity,
)
from recsys.training.trainer import resolve_device


def print_comparison_table(
    results: dict[str, dict[str, float]],
    ks: tuple[int, ...],
    title: str,
) -> None:
    """打印一张列顺序固定的指标对比表。"""
    metric_names = [
        metric_name
        for k in ks
        for metric_name in (f"Recall@{k}", f"NDCG@{k}", f"HitRate@{k}")
    ]
    header = f"{'Model':<12}" + "".join(
        f"{metric_name:>13}" for metric_name in metric_names
    )
    print(title)
    print(header)
    for model_name, metrics in results.items():
        values = "".join(f"{metrics[name]:>13.6f}" for name in metric_names)
        print(f"{model_name:<12}{values}")


def load_mf_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[MatrixFactorization, dict]:
    """按 checkpoint 里保存的配置把矩阵分解模型重建出来。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run python -m recsys.training.train_mf first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("split_strategy") != SPLIT_STRATEGY:
        raise ValueError("MF checkpoint does not use the current temporal split")
    model = MatrixFactorization(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def compare_models(
    data_dir: Path,
    two_tower_checkpoint: Path,
    mf_checkpoint: Path,
    split: str = "test",
    ks: tuple[int, ...] = (10, 20),
    user_batch_size: int = 512,
    device_name: str = "auto",
    print_results: bool = True,
) -> dict[str, dict[str, float]]:
    """对三个模型跑同一套「先过滤、再全量排序」的评测。

    数据集与正反馈阈值从 Two-Tower checkpoint 里读回来，所以 MovieLens 和
    KuaiRec 走的是同一条评测路径，只有数据源不同。
    """
    device = resolve_device(device_name)
    two_tower_model, two_tower_checkpoint_data = load_model(
        two_tower_checkpoint, device
    )
    dataset = resolve_checkpoint_dataset(two_tower_checkpoint_data)
    positive_threshold = resolve_positive_threshold(
        dataset,
        two_tower_checkpoint_data.get("positive_threshold"),
    )
    prepared_data = prepare_dataset(
        dataset=dataset,
        data_dir=data_dir,
        positive_threshold=positive_threshold,
    )
    evaluation_interactions = get_evaluation_interactions(prepared_data, split)
    evaluation_users = sorted(group_items_by_user(evaluation_interactions))
    train_items_by_user = seen_items_for_split(prepared_data, split)

    popularity_ranking = rank_items_by_popularity(
        prepared_data.train_interactions, prepared_data.num_items
    )
    popularity_recommendations = generate_popularity_recommendations(
        user_indices=evaluation_users,
        popularity_ranking=popularity_ranking,
        train_items_by_user=train_items_by_user,
        k=max(ks),
    )
    results = {
        "Popularity": evaluate_recommendations(
            popularity_recommendations, evaluation_interactions, ks
        )
    }

    mf_model, mf_checkpoint_data = load_mf_model(mf_checkpoint, device)
    # 两个 embedding 模型都必须与评测所用数据的词表一致，
    # 否则索引空间对不上，指标没有意义。
    for name, checkpoint in (
        ("MF", mf_checkpoint_data),
        ("Two-Tower", two_tower_checkpoint_data),
    ):
        if resolve_checkpoint_dataset(checkpoint) != dataset:
            raise ValueError(f"{name} checkpoint was trained on another dataset")
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

    if print_results:
        print_comparison_table(
            results,
            ks,
            f"{dataset} evaluation split: {split} ({len(evaluation_users):,} users)",
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--two-tower-checkpoint",
        type=Path,
        default=Path("artifacts/movielens/two_tower.pt"),
    )
    parser.add_argument(
        "--mf-checkpoint",
        type=Path,
        default=Path("artifacts/movielens/matrix_factorization.pt"),
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--user-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_models(
        data_dir=args.data_dir,
        two_tower_checkpoint=args.two_tower_checkpoint,
        mf_checkpoint=args.mf_checkpoint,
        split=args.split,
        user_batch_size=args.user_batch_size,
        device_name=args.device,
    )
