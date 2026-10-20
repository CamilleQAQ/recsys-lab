"""一个小而好读的 SASRec 模型，用 PyTorch 手写实现。"""

from __future__ import annotations

import math

import torch
from torch import nn


def build_causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """值为 True 的位置是「未来位置」，注意力不允许看到它们。"""
    return torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )


class SASRecBlock(nn.Module):
    """先做 LayerNorm 的因果自注意力，再接一个前馈网络。"""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """按「注意力 + 前馈」两个子层依次计算，并各自带残差连接。

        ``causal_mask`` 负责挡住未来位置；``padding_mask`` 标出补齐位置，
        每过一个子层都会把补齐位置重新清零，避免它们混进后面的表示。
        """
        normalized = self.attention_norm(hidden_states)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.attention_dropout(attention_output)
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        normalized = self.feed_forward_norm(hidden_states)
        feed_forward_output = self.feed_forward(normalized)
        hidden_states = hidden_states + self.feed_forward_dropout(feed_forward_output)
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    """根据因果编码后的物品序列，预测下一个物品。"""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        # token 0 留给补齐（padding）。内部 item_idx 为 i 的物品用 token i + 1 表示。
        self.item_embedding = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                SASRecBlock(embedding_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """用较小方差初始化 embedding，并把补齐位置的向量固定为零。"""
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, item_sequence: torch.Tensor) -> torch.Tensor:
        # item_sequence: [batch_size, seq_len]，0 表示补齐位置
        batch_size, seq_len = item_sequence.shape
        if seq_len > self.max_seq_len:
            raise ValueError("sequence is longer than max_seq_len")
        padding_mask = item_sequence.eq(0)
        # padding_mask: [batch_size, seq_len]

        position_ids = torch.arange(seq_len, device=item_sequence.device)
        position_embeddings = self.position_embedding(position_ids)
        # position_embeddings: [seq_len, embedding_dim]
        # 乘 sqrt(d) 是 transformer 里常见的输入缩放，让 embedding 的量级
        # 与紧接着要相加的位置编码保持在同一个尺度上。
        hidden_states = self.item_embedding(item_sequence) * math.sqrt(
            self.embedding_dim
        )
        # hidden_states: [batch_size, seq_len, embedding_dim]
        hidden_states = hidden_states + position_embeddings.unsqueeze(0)
        hidden_states = self.input_dropout(hidden_states)
        # 进入注意力之前，先把补齐位置整体置零。
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        causal_mask = build_causal_attention_mask(seq_len, item_sequence.device)
        # causal_mask: [seq_len, seq_len]
        for block in self.blocks:
            hidden_states = block(hidden_states, causal_mask, padding_mask)
        hidden_states = self.output_norm(hidden_states)
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # hidden_states: [batch_size, seq_len, embedding_dim]
        return hidden_states

    def sequence_representation(self, item_sequence: torch.Tensor) -> torch.Tensor:
        """取出每条序列「最后一个非补齐位置」的隐状态，作为整条序列的表示。"""
        hidden_states = self.forward(item_sequence)
        valid_lengths = item_sequence.ne(0).sum(dim=1)
        if torch.any(valid_lengths == 0):
            raise ValueError("each sequence must contain at least one item")
        last_positions = valid_lengths - 1
        batch_indices = torch.arange(
            item_sequence.shape[0], device=item_sequence.device
        )
        representation = hidden_states[batch_indices, last_positions]
        # representation: [batch_size, embedding_dim]
        return representation

    def score_all_items(self, item_sequence: torch.Tensor) -> torch.Tensor:
        """给全部候选物品打分，分数是它们与序列末位表示的相似度。

        这里直接用原始点积当分数：``generate_sasrec_recommendations`` 只关心
        排序结果，因此不需要再做 softmax。
        """
        representation = self.sequence_representation(item_sequence)
        candidate_embeddings = self.item_embedding.weight[1:]
        # candidate_embeddings: [num_items, embedding_dim]
        scores = representation @ candidate_embeddings.T
        # scores: [batch_size, num_items]
        return scores
