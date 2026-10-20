"""推荐评测常用的 Top-K 排序指标。"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


# 推荐结果：用户索引 -> 按分数从高到低排好的物品索引序列。
Recommendations = Mapping[int, Sequence[int]]
# 标准答案：用户索引 -> 该用户真正相关的物品集合。
GroundTruth = Mapping[int, Collection[int]]


def _require_relevant_items_for_every_user(ground_truth: GroundTruth) -> None:
    """每个参与评测的用户都必须至少有一个相关物品。"""
    if any(len(relevant_items) == 0 for relevant_items in ground_truth.values()):
        raise ValueError("each evaluated user must have at least one relevant item")


def _validate_inputs(ground_truth: GroundTruth, k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ground_truth:
        raise ValueError("ground_truth must contain at least one user")
    _require_relevant_items_for_every_user(ground_truth)


def recall_at_k(
    recommendations: Recommendations, ground_truth: GroundTruth, k: int
) -> float:
    """每个用户在前 K 个推荐里命中相关物品的平均比例。"""
    _validate_inputs(ground_truth, k)
    recalls = []
    for user_idx, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_idx, ())[:k]
        hits = len(set(recommended_items).intersection(relevant_items))
        recalls.append(hits / len(relevant_items))
    return sum(recalls) / len(recalls)


def hit_rate_at_k(
    recommendations: Recommendations, ground_truth: GroundTruth, k: int
) -> float:
    """前 K 个推荐里至少命中一个相关物品的用户比例。

    在 leave-one-out（每个用户只留一个测试物品）的协议下，每个用户的相关
    物品只有一个，因此 HitRate@K 与 Recall@K 的数值完全相同。
    """
    _validate_inputs(ground_truth, k)
    hits = 0
    for user_idx, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_idx, ())[:k]
        if set(recommended_items).intersection(relevant_items):
            hits += 1
    return hits / len(ground_truth)


def ndcg_at_k(
    recommendations: Recommendations, ground_truth: GroundTruth, k: int
) -> float:
    """二值相关性下，归一化折损累计增益（NDCG）的平均值。"""
    _validate_inputs(ground_truth, k)
    user_scores = []

    for user_idx, relevant_items in ground_truth.items():
        recommended_items = recommendations.get(user_idx, ())[:k]
        # DCG 只累加「前 K 名里真正命中的位置」的折损增益。
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, item_idx in enumerate(recommended_items)
            if item_idx in relevant_items
        )
        # 理想排序会把所有相关物品尽量排在前面。
        # min(len(relevant_items), k) 表示其中最多还能放进前 K 名的个数。
        ideal_hits = min(len(relevant_items), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        user_scores.append(dcg / ideal_dcg)

    return sum(user_scores) / len(user_scores)


def compute_ranking_metrics(
    recommendations: Recommendations,
    ground_truth: GroundTruth,
    ks: Sequence[int] = (10, 20),
) -> dict[str, float]:
    """对每个截断位置 K 计算全部排序指标。"""
    if not ks:
        raise ValueError("ks must contain at least one cutoff")

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recall_at_k(recommendations, ground_truth, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(recommendations, ground_truth, k)
        metrics[f"HitRate@{k}"] = hit_rate_at_k(
            recommendations, ground_truth, k
        )
    return metrics
