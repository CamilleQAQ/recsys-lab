"""只用 ID 作为特征的 Two-Tower 召回模型，训练时使用 in-batch negatives。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Tower(nn.Module):
    """ID embedding -> MLP -> L2 归一化向量。

    ``tower_type="embedding"`` 时跳过 MLP，直接学习输出向量（即最朴素的
    隐向量查表）；``tower_type="mlp"`` 时让 embedding 再经过两层 MLP。
    两种情况下返回的向量都会投影到单位球面上，这样内积就等于余弦相似度。
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        tower_type: str = "mlp",
    ) -> None:
        super().__init__()
        if tower_type not in {"embedding", "mlp"}:
            raise ValueError("tower_type must be 'embedding' or 'mlp'")
        self.tower_type = tower_type
        if tower_type == "embedding":
            self.embedding = nn.Embedding(num_entities, output_dim)
            self.mlp = nn.Identity()
        else:
            self.embedding = nn.Embedding(num_entities, embedding_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        # entity_ids: [batch_size]
        entity_embeddings = self.embedding(entity_ids)
        # entity_embeddings: [batch_size, embedding_dim]；当
        # tower_type == "embedding" 时这里其实是 output_dim，
        # 因为 MLP 被替换成了 nn.Identity。
        tower_vectors = self.mlp(entity_embeddings)
        # tower_vectors: [batch_size, output_dim]
        normalized_vectors = F.normalize(tower_vectors, p=2, dim=1)
        # normalized_vectors: [batch_size, output_dim]
        return normalized_vectors


class TwoTowerModel(nn.Module):
    """用户塔与物品塔各自独立，但归一化后的输出共享同一个向量空间。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        temperature: float = 0.2,
        tower_type: str = "mlp",
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.user_tower = Tower(
            num_users, embedding_dim, hidden_dim, output_dim, tower_type
        )
        self.item_tower = Tower(
            num_items, embedding_dim, hidden_dim, output_dim, tower_type
        )
        self.temperature = temperature
        self.tower_type = tower_type

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_ids)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.encode_users(user_ids)
        item_embeddings = self.encode_items(item_ids)
        # user_embeddings: [batch_size, output_dim]
        # item_embeddings: [batch_size, output_dim]

        similarity_matrix = user_embeddings @ item_embeddings.T
        # similarity_matrix: [batch_size, batch_size]
        # 第 i 行是「用户 i 与本批次内每一个物品」的相似度。因此对角线上的
        # (i, i) 就是真实发生过的正样本对，其余位置天然成为 in-batch 负样本。
        logits = similarity_matrix / self.temperature
        # logits: [batch_size, batch_size]
        return logits
