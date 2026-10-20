"""训练 Two-Tower 召回基线（MovieLens / KuaiRec 二选一）。"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from recsys.config import DEFAULT_DATASET, SPLIT_STRATEGY
from recsys.data.loader import (
    SUPPORTED_DATASETS,
    prepare_dataset,
    resolve_positive_threshold,
)
from recsys.data.movielens import Interaction, MovieLensDataset
from recsys.models.two_tower import TwoTowerModel
from recsys.training.losses import in_batch_cross_entropy_loss
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    epochs: int = 9
    batch_size: int = 256
    learning_rate: float = 1e-2
    embedding_dim: int = 128
    hidden_dim: int = 128
    output_dim: int = 128
    temperature: float = 0.1
    tower_type: str = "embedding"
    mask_false_negatives: bool = True
    positive_threshold: float | None = None
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None


def resolve_device(device: str) -> torch.device:
    """``"auto"`` 表示有 CUDA 就用 GPU，否则回退到 CPU；其他值按原样解析。"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_dataloader(
    dataset: MovieLensDataset, batch_size: int, seed: int, shuffle: bool = True
) -> DataLoader:
    """创建 DataLoader，并用固定种子的生成器保证打乱顺序可复现。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_one_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    train_positive_matrix: torch.Tensor | None = None,
) -> float:
    """训练一个 epoch，返回「实际用到的批次」上的平均损失。"""
    model.train()
    loss_sum = 0.0
    batches_seen = 0

    for batch_number, batch in enumerate(dataloader):
        if max_batches is not None and batch_number >= max_batches:
            break

        user_ids = batch["user_idx"].to(device)
        item_ids = batch["item_idx"].to(device)
        # user_ids: [batch_size]
        # item_ids: [batch_size]

        logits = model(user_ids, item_ids)
        # logits: [batch_size, batch_size]
        false_negative_mask = None
        if train_positive_matrix is not None:
            # 先取批内用户，再取批内物品，两步索引得到子矩阵。
            # false_negative_mask[i, j] 为 True 表示「物品 j 是用户 i 在训练集
            # 里已经交互过的正样本」，因此不能把它当负样本。
            false_negative_mask = train_positive_matrix[user_ids][:, item_ids]
            # false_negative_mask: [batch_size, batch_size]
        loss = in_batch_cross_entropy_loss(logits, false_negative_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        batches_seen += 1

    if batches_seen == 0:
        raise ValueError("Training DataLoader produced no batches")
    return loss_sum / batches_seen


def build_train_positive_matrix(
    interactions: Sequence[Interaction],
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """构建整张 MovieLens 规模的真值表，用于屏蔽误判的负样本。

    当 ``[user_idx, item_idx]`` 这对组合确实出现在训练集里时，该位置为 True。
    有了它，训练时就能忽略「批内偶然出现的假负样本」。
    """
    positive_matrix = torch.zeros(
        (num_users, num_items), dtype=torch.bool, device=device
    )
    # 用高级索引把观测到的正样本一次性散射到真值表上。
    user_indices = torch.tensor(
        [interaction.user_idx for interaction in interactions],
        dtype=torch.long,
        device=device,
    )
    item_indices = torch.tensor(
        [interaction.item_idx for interaction in interactions],
        dtype=torch.long,
        device=device,
    )
    positive_matrix[user_indices, item_indices] = True
    return positive_matrix


def train(config: TrainConfig) -> tuple[TwoTowerModel, list[float]]:
    """完整跑一遍训练流程，保存 checkpoint 并返回模型与每轮损失。"""
    set_random_seed(config.seed)
    device = resolve_device(config.device)

    positive_threshold = resolve_positive_threshold(
        config.dataset, config.positive_threshold
    )
    prepared_data = prepare_dataset(
        dataset=config.dataset,
        data_dir=config.data_dir,
        positive_threshold=positive_threshold,
    )
    train_dataset = MovieLensDataset(prepared_data.train_interactions)
    train_loader = create_dataloader(
        train_dataset, config.batch_size, config.seed, shuffle=True
    )

    model = TwoTowerModel(
        num_users=prepared_data.num_users,
        num_items=prepared_data.num_items,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        temperature=config.temperature,
        tower_type=config.tower_type,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_positive_matrix = None
    if config.mask_false_negatives:
        # 仅在开启屏蔽时才构建这张真值表，因为它的大小是「用户数 x 物品数」。
        train_positive_matrix = build_train_positive_matrix(
            prepared_data.train_interactions,
            prepared_data.num_users,
            prepared_data.num_items,
            device,
        )

    num_train = len(prepared_data.train_interactions)
    num_validation = len(prepared_data.validation_interactions)
    num_test = len(prepared_data.test_interactions)
    print(f"Device: {device}")
    print(
        f"Dataset: {config.dataset} "
        f"(positive threshold {positive_threshold:g})"
    )
    print(
        f"Positive interactions: {num_train + num_validation + num_test:,} "
        f"(train={num_train:,}, validation={num_validation:,}, test={num_test:,})"
    )
    print(
        f"Vocabulary: {prepared_data.num_users:,} users, "
        f"{prepared_data.num_items:,} items"
    )

    epoch_losses: list[float] = []
    best_epoch = 0
    best_validation_metrics: dict[str, float] = {}
    best_state_dict = None
    from recsys.evaluation.evaluator import evaluate_embedding_model
    for epoch in range(config.epochs):
        average_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=config.max_batches_per_epoch,
            train_positive_matrix=train_positive_matrix,
        )
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch + 1:02d}/{config.epochs:02d} - loss: {average_loss:.6f}")
        if prepared_data.validation_interactions:
            metrics = evaluate_embedding_model(
                model, prepared_data, "validation", (10, 20), 512, device
            )
            print(f"  validation Recall@20: {metrics['Recall@20']:.6f}")
            if not best_validation_metrics or metrics["Recall@20"] > best_validation_metrics["Recall@20"]:
                best_epoch = epoch + 1
                best_validation_metrics = metrics
                best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.artifacts_dir / config.dataset / "two_tower.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint 里同时存模型权重和重建模型所需的全部配置，
    # 这样评估脚本只看这一个文件就能把模型还原出来。
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "split_strategy": SPLIT_STRATEGY,
            "dataset": config.dataset,
            "positive_threshold": positive_threshold,
            "model_config": {
                "num_users": prepared_data.num_users,
                "num_items": prepared_data.num_items,
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "output_dim": config.output_dim,
                "temperature": config.temperature,
                "tower_type": config.tower_type,
            },
            "train_config": json_ready_config(config),
            "epoch_losses": epoch_losses,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation_metrics,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    from recsys.data.loader import resolve_dataset_data_dir
    mapping_path = resolve_dataset_data_dir(config.dataset, config.data_dir) / "processed" / "id_mappings.json"
    print(f"ID mappings saved to {mapping_path}")
    return model, epoch_losses


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="每个数据集写进它自己的子目录，避免互相覆盖",
    )
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--tower-type", choices=["embedding", "mlp"], default="embedding"
    )
    parser.add_argument(
        "--no-mask-false-negatives",
        action="store_false",
        dest="mask_false_negatives",
        help="Disable masking of known train positives in off-diagonal logits.",
    )
    parser.set_defaults(mask_false_negatives=True)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=None,
        help="rating threshold for MovieLens or watch_ratio threshold for KuaiRec; "
        "defaults to the audited value in recsys.config",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Optional smoke-run limit; omit it for full training.",
    )
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
