"""用 FAISS 的 IndexFlatIP 做精确向量检索。

IndexFlatIP 执行的是穷举式内积检索，属于「精确索引」，
而不是近似最近邻（ANN）索引。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import faiss
import numpy as np
import torch
from torch import nn


def as_float32_numpy(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
    """转成 FAISS 能接受的连续 float32 矩阵。"""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must have shape [num_vectors, embedding_dim]")
    return np.ascontiguousarray(array)


def check_l2_normalized(embeddings: np.ndarray, name: str) -> None:
    """只要存在长度不为 1 的行就立刻报错。

    向量必须是 L2 归一化的，内积检索才等价于余弦相似度；
    这里提前拦住，避免排序结果悄悄出错却没人发现。
    """
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f"{name} must be L2 normalized before inner-product search")


@dataclass
class FaissItemIndex:
    """FAISS 索引，外加两套显式的位置到物品 ID 的映射。

    FAISS 只认「插入顺序的下标」，所以需要自己记录每个下标对应哪个
    内部 item_idx，以及进一步对应哪个原始 movie_id。
    """

    index: faiss.IndexFlatIP
    position_to_item_idx: np.ndarray
    index_to_original_movie_id: list[int]

    @property
    def num_items(self) -> int:
        return self.index.ntotal

    @property
    def embedding_dim(self) -> int:
        return self.index.d

    def search(
        self, query_embeddings: torch.Tensor | np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回内积分数与内部的 item_idx 值。"""
        queries = as_float32_numpy(query_embeddings)
        check_l2_normalized(queries, "query embeddings")
        if queries.shape[1] != self.embedding_dim:
            raise ValueError("query and index embedding dimensions must match")
        if not 0 < k <= self.num_items:
            raise ValueError("k must be between 1 and the number of indexed items")

        scores, positions = self.index.search(queries, k)
        # positions: [num_queries, k]，是 FAISS 内部按插入顺序排的下标
        internal_item_indices = self.position_to_item_idx[positions]
        # internal_item_indices: [num_queries, k]
        return scores, internal_item_indices

    def to_original_movie_ids(self, item_indices: Sequence[int]) -> list[int]:
        """把内部 item_idx 还原成 MovieLens 的原始 movie_id。"""
        return [self.index_to_original_movie_id[item_idx] for item_idx in item_indices]


def build_faiss_index(
    item_embeddings: torch.Tensor | np.ndarray,
    position_to_item_idx: Sequence[int],
    index_to_original_movie_id: Sequence[int],
) -> FaissItemIndex:
    """在归一化后的物品向量上构建精确的 IndexFlatIP 索引。"""
    vectors = as_float32_numpy(item_embeddings)
    check_l2_normalized(vectors, "item embeddings")
    item_indices = np.asarray(position_to_item_idx, dtype=np.int64)
    if len(item_indices) != len(vectors):
        raise ValueError("one internal item_idx is required per item embedding")
    if set(item_indices.tolist()) != set(range(len(vectors))):
        raise ValueError(
            "position_to_item_idx must contain every internal item index once"
        )
    if len(index_to_original_movie_id) != len(vectors):
        raise ValueError("original movie ID mapping must cover every indexed item")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return FaissItemIndex(
        index=index,
        position_to_item_idx=item_indices,
        index_to_original_movie_id=list(index_to_original_movie_id),
    )


def faiss_top_k_from_embeddings(
    user_indices: Sequence[int],
    user_embeddings: torch.Tensor | np.ndarray,
    item_index: FaissItemIndex,
    train_items_by_user: Mapping[int, set[int]],
    k: int,
    initial_candidates: int = 100,
) -> dict[int, list[int]]:
    """检索、过滤已看物品，并保留 K 个未见过的内部 item_idx。"""
    queries = as_float32_numpy(user_embeddings)
    if len(user_indices) != len(queries):
        raise ValueError("one user embedding is required for each user index")
    if not 0 < k <= item_index.num_items:
        raise ValueError("k must be between 1 and the number of indexed items")

    recommendations: dict[int, list[int]] = {}
    # 这里的「行」是 user_indices / queries 中的位置，而不是用户 ID。
    unresolved_rows = list(range(len(user_indices)))
    search_k = min(max(k, initial_candidates), item_index.num_items)

    # 通常第一次多取一些候选就够了。只有当某个用户的已看物品大量排在前列时，
    # 才把这些「还没凑够 k 个」的用户挑出来，用更大的 K 重新检索一次。
    while unresolved_rows:
        _, candidate_item_indices = item_index.search(
            queries[unresolved_rows], search_k
        )
        still_unresolved_rows = []
        for result_row, original_row in enumerate(unresolved_rows):
            user_idx = user_indices[original_row]
            seen_items = train_items_by_user.get(user_idx, set())
            unseen_items = [
                int(item_idx)
                for item_idx in candidate_item_indices[result_row]
                if int(item_idx) not in seen_items
            ]
            if len(unseen_items) >= k:
                recommendations[user_idx] = unseen_items[:k]
            elif search_k == item_index.num_items:
                raise ValueError(f"user {user_idx} has fewer than k unseen items")
            else:
                still_unresolved_rows.append(original_row)

        unresolved_rows = still_unresolved_rows
        search_k = min(search_k * 2, item_index.num_items)

    return recommendations


@torch.no_grad()
def encode_users_and_items(
    model: nn.Module,
    user_indices: Sequence[int],
    num_items: int,
    device: torch.device,
    user_batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次性编码待检索用户和全部物品，返回 CPU 上的 float32 张量。"""
    model.eval()
    item_ids = torch.arange(num_items, dtype=torch.long, device=device)
    item_embeddings = model.encode_items(item_ids).float().cpu()
    # item_embeddings: [num_items, embedding_dim]

    user_embedding_batches = []
    for start in range(0, len(user_indices), user_batch_size):
        batch_user_ids = torch.tensor(
            user_indices[start : start + user_batch_size],
            dtype=torch.long,
            device=device,
        )
        batch_user_embeddings = model.encode_users(batch_user_ids)
        # batch_user_embeddings: [user_batch_size, embedding_dim]
        user_embedding_batches.append(batch_user_embeddings.float().cpu())
    user_embeddings = torch.cat(user_embedding_batches, dim=0)
    # user_embeddings: [num_users, embedding_dim]
    return user_embeddings, item_embeddings


def top_k_agreement(
    first: Mapping[int, Sequence[int]],
    second: Mapping[int, Sequence[int]],
    user_indices: Sequence[int],
    k: int,
) -> tuple[float, float]:
    """返回平均 Top-K 重合率，以及「顺序完全一致」的用户比例。"""
    if not user_indices:
        raise ValueError("user_indices must not be empty")
    overlap_sum = 0.0
    exact_matches = 0
    for user_idx in user_indices:
        first_items = list(first[user_idx][:k])
        second_items = list(second[user_idx][:k])
        overlap_sum += len(set(first_items).intersection(second_items)) / k
        if first_items == second_items:
            exact_matches += 1
    return overlap_sum / len(user_indices), exact_matches / len(user_indices)
