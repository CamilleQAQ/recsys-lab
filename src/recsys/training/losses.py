"""训练用的损失函数。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def in_batch_cross_entropy_loss(
    logits: torch.Tensor, false_negative_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """把对角线上的用户-物品对当作正样本，其余位置全部当作负样本。

    ``false_negative_mask[i, j]`` 可以标记「物品 j 其实是用户 i 在训练集中
    已经交互过的正样本」。这类位置会被压到 ``-inf``，softmax 后概率为 0，
    从而避免把一次正确的关联当成错误来惩罚。
    """
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square [batch_size, batch_size] matrix")

    batch_size = logits.shape[0]
    if false_negative_mask is not None:
        if false_negative_mask.shape != logits.shape:
            raise ValueError("false_negative_mask must have the same shape as logits")
        # 对角线必须保留，它就是第 i 行的目标正样本。
        is_positive_pair = torch.eye(
            batch_size, dtype=torch.bool, device=logits.device
        )
        is_false_negative = false_negative_mask.to(logits.device) & ~is_positive_pair
        logits = logits.masked_fill(is_false_negative, -torch.inf)
    # 批内第 i 个样本是 (user_i, positive_item_i)，它的正样本得分就是
    # logits[i, i]，所以每一行的分类目标正好是自己的行号：
    # [0, 1, ..., batch_size - 1]，其余每一列都是负样本。
    targets = torch.arange(batch_size, device=logits.device)
    # targets: [batch_size]
    return F.cross_entropy(logits, targets)
