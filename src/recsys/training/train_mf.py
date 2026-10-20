"""用「隐式正样本 + 随机采样负样本」训练矩阵分解模型。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from recsys.config import DEFAULT_DATASET, SPLIT_STRATEGY
from recsys.data.loader import (
    SUPPORTED_DATASETS,
    prepare_dataset,
    resolve_positive_threshold,
)
from recsys.data.movielens import MovieLensDataset
from recsys.evaluation.evaluator import group_items_by_user
from recsys.models.matrix_factorization import MatrixFactorization
from recsys.training.samplers import sample_negative_items
from recsys.training.trainer import create_dataloader, resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class MFTrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 1e-2
    embedding_dim: int = 64
    positive_threshold: float | None = None
    seed: int = 42
    device: str = "auto"
    max_batches_per_epoch: int | None = None


def train_one_mf_epoch(
    model: MatrixFactorization,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_items_by_user: dict[int, set[int]],
    num_items: int,
    generator: torch.Generator,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """训练一个 epoch，返回实际用到的批次上的平均损失。"""
    model.train()
    loss_sum = 0.0
    batches_seen = 0

    for batch_number, batch in enumerate(dataloader):
        if max_batches is not None and batch_number >= max_batches:
            break

        user_ids = batch["user_idx"]
        positive_item_ids = batch["item_idx"]
        negative_item_ids = sample_negative_items(
            user_ids, train_items_by_user, num_items, generator
        )

        user_ids = user_ids.to(device)
        positive_item_ids = positive_item_ids.to(device)
        negative_item_ids = negative_item_ids.to(device)
        # user_ids: [batch_size]
        # positive_item_ids: [batch_size]
        # negative_item_ids: [batch_size]

        positive_logits = model(user_ids, positive_item_ids)
        negative_logits = model(user_ids, negative_item_ids)
        # positive_logits: [batch_size]
        # negative_logits: [batch_size]

        # 对两半一起做二分类交叉熵：观测到的组合标签为 1，
        # 随机采样的未交互物品标签为 0。
        logits = torch.cat([positive_logits, negative_logits])
        labels = torch.cat(
            [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]
        )
        # logits: [2 * batch_size]
        # labels: [2 * batch_size]
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        batches_seen += 1

    if batches_seen == 0:
        raise ValueError("Training DataLoader produced no batches")
    return loss_sum / batches_seen


def train_mf(config: MFTrainConfig) -> tuple[MatrixFactorization, list[float]]:
    """完整跑一遍 MF 训练流程，保存 checkpoint 并返回模型与每轮损失。"""
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
    train_items_by_user = group_items_by_user(prepared_data.train_interactions)
    train_dataset = MovieLensDataset(prepared_data.train_interactions)
    train_loader = create_dataloader(
        train_dataset, config.batch_size, config.seed, shuffle=True
    )

    model = MatrixFactorization(
        num_users=prepared_data.num_users,
        num_items=prepared_data.num_items,
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 固定种子的生成器，保证每次运行的负采样序列一致。
    negative_generator = torch.Generator().manual_seed(config.seed)

    print(f"Device: {device}")
    print(
        f"Dataset: {config.dataset} (positive threshold {positive_threshold:g})"
    )
    print(
        f"Train interactions: {len(prepared_data.train_interactions):,}, "
        f"users: {prepared_data.num_users:,}, items: {prepared_data.num_items:,}"
    )
    epoch_losses: list[float] = []
    best_epoch = 0
    best_validation_metrics: dict[str, float] = {}
    best_state_dict = None
    for epoch in range(config.epochs):
        average_loss = train_one_mf_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            train_items_by_user=train_items_by_user,
            num_items=prepared_data.num_items,
            generator=negative_generator,
            device=device,
            max_batches=config.max_batches_per_epoch,
        )
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch + 1:02d}/{config.epochs:02d} - loss: {average_loss:.6f}")
        if prepared_data.validation_interactions:
            from recsys.evaluation.evaluator import evaluate_embedding_model
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

    checkpoint_path = config.artifacts_dir / config.dataset / "matrix_factorization.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
            },
            "train_config": json_ready_config(config),
            "epoch_losses": epoch_losses,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation_metrics,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model, epoch_losses


def parse_args() -> MFTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=None,
        help="defaults to the audited value in recsys.config for the chosen dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    return MFTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_mf(parse_args())
