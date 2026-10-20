"""隐式反馈场景下最简版的矩阵分解模型。"""

from __future__ import annotations

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """学习用户与物品的隐向量，二者点积即为偏好分数。

    这是最经典的 MF 基线：不带非线性，也没有「用户-物品」交叉特征，
    每个用户和每个物品各自只有一个隐向量。
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # 用较小的随机值初始化，既能打破各隐向量之间的对称性，
        # 又不会让初始点积离 0 太远。
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """按用户索引取出对应的隐向量。"""
        # user_ids: [batch_size]
        user_embeddings = self.user_embedding(user_ids)
        # user_embeddings: [batch_size, embedding_dim]
        return user_embeddings

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """按物品索引取出对应的隐向量。"""
        # item_ids: [batch_size]
        item_embeddings = self.item_embedding(item_ids)
        # item_embeddings: [batch_size, embedding_dim]
        return item_embeddings

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_embeddings = self.encode_users(user_ids)
        item_embeddings = self.encode_items(item_ids)
        # user_embeddings: [batch_size, embedding_dim]
        # item_embeddings: [batch_size, embedding_dim]
        # 逐元素相乘后在第 1 维求和，等价于每一对 (用户, 物品) 的点积。
        logits = (user_embeddings * item_embeddings).sum(dim=1)
        # logits: [batch_size]
        return logits
