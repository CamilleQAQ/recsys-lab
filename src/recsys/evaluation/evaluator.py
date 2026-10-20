"""对训练好的 Two-Tower 模型做「全量排序」精确评测。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn

from recsys.config import SPLIT_STRATEGY
from recsys.data.loader import (
    prepare_dataset,
    resolve_checkpoint_dataset,
    resolve_positive_threshold,
)
from recsys.data.schema import Interaction, PreparedData, seen_items_for_split
from recsys.evaluation.metrics import compute_ranking_metrics
from recsys.models.two_tower import TwoTowerModel
from recsys.training.trainer import resolve_device


def group_items_by_user(
    interactions: Sequence[Interaction],
) -> dict[int, set[int]]:
    """把交互列表整理成「用户 -> 交互过的物品集合」。"""
    items_by_user: dict[int, set[int]] = defaultdict(set)
    for interaction in interactions:
        items_by_user[interaction.user_idx].add(interaction.item_idx)
    return dict(items_by_user)


def brute_force_top_k(
    user_indices: Sequence[int],
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    train_items_by_user: Mapping[int, set[int]],
    k: int,
) -> dict[int, list[int]]:
    """屏蔽掉目标时刻前已看过的物品后，用精确矩阵乘法给全部物品排序。"""
    if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise ValueError("user_embeddings and item_embeddings must be matrices")
    if user_embeddings.shape[0] != len(user_indices):
        raise ValueError("one user embedding is required for each user index")
    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError("user and item embedding dimensions must match")

    if not 0 < k <= item_embeddings.shape[0]:
        raise ValueError("k must be between 1 and the number of items")

    # user_embeddings: [user_batch_size, embedding_dim]
    # item_embeddings: [num_items, embedding_dim]
    scores = user_embeddings @ item_embeddings.T
    # scores: [user_batch_size, num_items]

    # 目标时刻前已经观测到的物品不能再被推荐，因此把它们的分数压到 -inf，
    # 这样它们永远不会进入 Top-K。留出的验证集物品不在这个屏蔽范围内，
    # 因为切分是按时间做的，它本来就不在目标时刻之前的历史里。
    for row_index, user_idx in enumerate(user_indices):
        seen_items = train_items_by_user.get(user_idx, set())
        if len(seen_items) > item_embeddings.shape[0] - k:
            raise ValueError(f"user {user_idx} has fewer than k unseen candidate items")
        if seen_items:
            seen_item_tensor = torch.tensor(
                sorted(seen_items), dtype=torch.long, device=scores.device
            )
            scores[row_index, seen_item_tensor] = -torch.inf

    top_item_indices = torch.topk(scores, k=k, dim=1).indices
    # top_item_indices: [user_batch_size, k]
    return {
        user_idx: item_indices
        for user_idx, item_indices in zip(
            user_indices, top_item_indices.cpu().tolist(), strict=True
        )
    }


@torch.no_grad()
def generate_recommendations(
    model: nn.Module,
    user_indices: Sequence[int],
    num_items: int,
    train_items_by_user: Mapping[int, set[int]],
    k: int,
    user_batch_size: int,
    device: torch.device,
) -> dict[int, list[int]]:
    """物品只编码一次，然后分批为每个用户做精确排序。"""
    if user_batch_size <= 0:
        raise ValueError("user_batch_size must be positive")

    model.eval()
    all_item_ids = torch.arange(num_items, dtype=torch.long, device=device)
    # all_item_ids: [num_items]
    all_item_embeddings = model.encode_items(all_item_ids)
    # all_item_embeddings: [num_items, output_dim]

    recommendations: dict[int, list[int]] = {}
    for start in range(0, len(user_indices), user_batch_size):
        batch_user_indices = list(user_indices[start : start + user_batch_size])
        batch_user_ids = torch.tensor(
            batch_user_indices, dtype=torch.long, device=device
        )
        # batch_user_ids: [user_batch_size]
        batch_user_embeddings = model.encode_users(batch_user_ids)
        # batch_user_embeddings: [user_batch_size, output_dim]
        recommendations.update(
            brute_force_top_k(
                user_indices=batch_user_indices,
                user_embeddings=batch_user_embeddings,
                item_embeddings=all_item_embeddings,
                train_items_by_user=train_items_by_user,
                k=k,
            )
        )
    return recommendations


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TwoTowerModel, dict]:
    """按 checkpoint 里保存的配置把模型重建出来。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run python -m recsys.training.trainer --dataset <name> first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("split_strategy") != SPLIT_STRATEGY:
        raise ValueError(
            "Checkpoint uses the old data split. "
            "Retrain it with python -m recsys.training.trainer."
        )
    model = TwoTowerModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def get_evaluation_interactions(
    prepared_data: PreparedData, split: str
) -> list[Interaction]:
    """返回指定切分留出的评测交互。"""
    if split == "validation":
        return prepared_data.validation_interactions
    if split == "test":
        return prepared_data.test_interactions
    raise ValueError("split must be 'validation' or 'test'")


def evaluate_recommendations(
    recommendations: Mapping[int, Sequence[int]],
    evaluation_interactions: Sequence[Interaction],
    ks: Sequence[int],
) -> dict[str, float]:
    """所有推荐模型共用同一套指标实现，保证对比是公平的。"""
    ground_truth = group_items_by_user(evaluation_interactions)
    return compute_ranking_metrics(recommendations, ground_truth, ks=ks)


def evaluate_embedding_model(
    model: nn.Module,
    prepared_data: PreparedData,
    split: str,
    ks: Sequence[int],
    user_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """评测任何提供 ``encode_users`` / ``encode_items`` 的模型。"""
    evaluation_interactions = get_evaluation_interactions(prepared_data, split)
    if not evaluation_interactions:
        raise ValueError(f"The temporal split contains no {split} interactions")
    train_items_by_user = seen_items_for_split(prepared_data, split)
    evaluation_items_by_user = group_items_by_user(evaluation_interactions)
    evaluation_users = sorted(evaluation_items_by_user)
    recommendations = generate_recommendations(
        model=model,
        user_indices=evaluation_users,
        num_items=prepared_data.num_items,
        train_items_by_user=train_items_by_user,
        k=max(ks),
        user_batch_size=user_batch_size,
        device=device,
    )
    return evaluate_recommendations(recommendations, evaluation_interactions, ks)


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    ks: Sequence[int] = (10, 20),
    user_batch_size: int = 512,
    device_name: str = "auto",
    split: str = "test",
) -> dict[str, float]:
    """在验证集或测试集上评测一个 Two-Tower checkpoint。"""
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive cutoffs")

    device = resolve_device(device_name)
    model, checkpoint = load_model(checkpoint_path, device)
    train_config = checkpoint.get("train_config", {})
    # 数据集与正反馈阈值必须从 checkpoint 里读回来，不能默认用 MovieLens：
    # 换了数据集或阈值，用户/物品词表就对不上，评测结果没有意义。
    dataset = resolve_checkpoint_dataset(checkpoint)
    positive_threshold = resolve_positive_threshold(
        dataset,
        checkpoint.get("positive_threshold", train_config.get("positive_threshold")),
    )

    prepared_data = prepare_dataset(
        dataset=dataset,
        data_dir=data_dir,
        positive_threshold=positive_threshold,
    )
    model_config = checkpoint["model_config"]
    if prepared_data.num_users != model_config["num_users"]:
        raise ValueError("Data user vocabulary does not match the checkpoint")
    if prepared_data.num_items != model_config["num_items"]:
        raise ValueError("Data item vocabulary does not match the checkpoint")
    evaluation_interactions = get_evaluation_interactions(prepared_data, split)
    metrics = evaluate_embedding_model(
        model=model,
        prepared_data=prepared_data,
        split=split,
        ks=ks,
        user_batch_size=user_batch_size,
        device=device,
    )

    print(f"Device: {device}")
    print(
        f"Dataset: {dataset} (positive threshold {positive_threshold:g})"
    )
    print(f"Evaluation split: chronological {split}")
    print(
        f"Users: {len(group_items_by_user(evaluation_interactions)):,}, "
        f"relevant interactions: {len(evaluation_interactions):,}, "
        f"candidate items: {prepared_data.num_items:,}"
    )
    print("Ranking: exact brute-force user_embeddings @ item_embeddings.T")
    print("All observed item exposures before target time are filtered before Top-K")
    for k in ks:
        for metric_name in ("Recall", "NDCG", "HitRate"):
            name = f"{metric_name}@{k}"
            print(f"{name}: {metrics[name]:.6f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/movielens/two_tower.pt")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--user-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        ks=args.ks,
        user_batch_size=args.user_batch_size,
        device_name=args.device,
        split=args.split,
    )
