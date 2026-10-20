"""只用训练集交互拟合的「热门度」推荐基线。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from recsys.data.movielens import Interaction


def rank_items_by_popularity(
    train_interactions: Sequence[Interaction], num_items: int
) -> list[int]:
    """返回按训练集交互次数降序排列的全部物品索引。"""
    item_counts = [0] * num_items
    for interaction in train_interactions:
        item_counts[interaction.item_idx] += 1

    # 次数相同时用物品索引兜底，保证排序结果稳定可复现。
    item_indices_by_popularity = sorted(
        range(num_items), key=lambda item_idx: (-item_counts[item_idx], item_idx)
    )
    return item_indices_by_popularity


def generate_popularity_recommendations(
    user_indices: Sequence[int],
    popularity_ranking: Sequence[int],
    train_items_by_user: Mapping[int, set[int]],
    k: int,
) -> dict[int, list[int]]:
    """给每个用户推荐全局最热门、且他还没看过的前 k 个物品。"""
    if k <= 0:
        raise ValueError("k must be positive")

    recommendations: dict[int, list[int]] = {}
    for user_idx in user_indices:
        seen_items = train_items_by_user.get(user_idx, set())
        unseen_popular_items = [
            item_idx for item_idx in popularity_ranking if item_idx not in seen_items
        ]
        if len(unseen_popular_items) < k:
            raise ValueError(f"user {user_idx} has fewer than k unseen candidate items")
        recommendations[user_idx] = unseen_popular_items[:k]
    return recommendations
