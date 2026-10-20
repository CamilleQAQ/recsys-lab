"""训练用的负采样工具，两个训练流程共用同一套实现。"""

from __future__ import annotations

import random

import torch


def sample_negative_items(
    user_ids: torch.Tensor,
    train_items_by_user: dict[int, set[int]],
    num_items: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """为每个用户采样一个「不在其训练历史里」的物品作为负样本。

    候选物品按均匀分布随机抽取；如果恰好抽到该用户已经交互过的物品就重新抽，
    因此返回的每一行对相应用户来说都是真正的负样本。
    """
    user_list = user_ids.cpu().tolist()
    negative_items = torch.randint(
        low=0,
        high=num_items,
        size=(len(user_list),),
        generator=generator,
        dtype=torch.long,
    )

    # 记录「抽到的物品不是合法负样本」的位置，后面需要重新抽。
    invalid_positions = [
        position
        for position, (user_idx, item_idx) in enumerate(
            zip(user_list, negative_items.tolist(), strict=True)
        )
        if item_idx in train_items_by_user[user_idx]
    ]
    while invalid_positions:
        resampled_items = torch.randint(
            low=0,
            high=num_items,
            size=(len(invalid_positions),),
            generator=generator,
            dtype=torch.long,
        )
        for position, item_idx in zip(
            invalid_positions, resampled_items.tolist(), strict=True
        ):
            negative_items[position] = item_idx
        # 只保留「重抽之后仍然撞上已看物品」的位置，继续下一轮重抽。
        invalid_positions = [
            position
            for position in invalid_positions
            if negative_items[position].item()
            in train_items_by_user[user_list[position]]
        ]
    return negative_items


def sample_negative_item(
    num_items: int,
    seen_items: set[int],
    random_source: random.Random,
) -> int:
    """随机取一个该用户从未交互过的物品，并返回加 1 之后的 token。

    这里用的是拒绝采样：只要用户没有看遍全部物品，循环就一定会结束。
    返回 ``item_idx + 1``，因为 token 0 要留给补齐位置。
    """
    if len(seen_items) >= num_items:
        raise ValueError("cannot sample a negative when every item is seen")
    negative_item_idx = random_source.randrange(num_items)
    while negative_item_idx in seen_items:
        negative_item_idx = random_source.randrange(num_items)
    return negative_item_idx + 1
