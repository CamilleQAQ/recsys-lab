"""可复现性相关的小工具。"""

from __future__ import annotations

import random

import torch


def set_random_seed(seed: int) -> None:
    """设置 Python 与 PyTorch 的随机种子，让实验可以复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
