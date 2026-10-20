"""训练 SASRec，并用验证集 Recall@20 选择要保存的 checkpoint。"""

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
from recsys.evaluation.sasrec_evaluator import evaluate_sasrec_model
from recsys.models.sasrec import SASRec
from recsys.data.dataset import SASRecTrainDataset
from recsys.training.trainer import resolve_device
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class SASRecTrainConfig:
    dataset: str = DEFAULT_DATASET
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    embedding_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    max_seq_len: int = 50
    dropout: float = 0.2
    positive_threshold: float | None = None
    seed: int = 42
    device: str = "auto"


def train_one_sasrec_epoch(
    model: SASRec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回该轮各批次的平均损失。

    SASRec 的训练目标是逐位置的二分类：真实的「下一个物品」标签为 1，
    随机采样的一个未交互物品标签为 0。
    """
    model.train()
    total_loss = 0.0
    batches_seen = 0
    for batch in dataloader:
        input_sequence = batch["input_sequence"].to(device)
        positive_sequence = batch["positive_sequence"].to(device)
        negative_sequence = batch["negative_sequence"].to(device)
        # 每个张量都是: [batch_size, max_seq_len]

        hidden_states = model(input_sequence)
        # hidden_states: [batch_size, max_seq_len, embedding_dim]
        positive_embeddings = model.item_embedding(positive_sequence)
        negative_embeddings = model.item_embedding(negative_sequence)
        # item embeddings: [batch_size, max_seq_len, embedding_dim]
        # 每个位置的隐状态与该位置的正/负物品向量做点积，得到该位置的分数。
        positive_logits = (hidden_states * positive_embeddings).sum(dim=-1)
        negative_logits = (hidden_states * negative_embeddings).sum(dim=-1)
        # logits: [batch_size, max_seq_len]

        # 补齐位置被排除在外：只有真正带有「下一个物品」目标的位置才计入损失。
        valid_positions = positive_sequence.ne(0)
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits[valid_positions],
            torch.ones_like(positive_logits[valid_positions]),
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_logits[valid_positions],
            torch.zeros_like(negative_logits[valid_positions]),
        )
        loss = positive_loss + negative_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches_seen += 1
    return total_loss / batches_seen


def train_sasrec(config: SASRecTrainConfig) -> tuple[SASRec, dict]:
    """只用训练集序列训练，保存哪个 epoch 完全由验证集决定。"""
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
    dataset = SASRecTrainDataset(
        prepared_data.train_interactions,
        num_items=prepared_data.num_items,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
    )
    generator = torch.Generator().manual_seed(config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = SASRec(
        num_items=prepared_data.num_items,
        max_seq_len=config.max_seq_len,
        embedding_dim=config.embedding_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_state_dict = None
    epochs_without_improvement = 0
    history = []
    print(
        f"Device: {device}, users with train sequences: {len(dataset):,}, "
        f"max_seq_len: {config.max_seq_len}"
    )
    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_sasrec_epoch(model, dataloader, optimizer, device)
        validation_metrics = evaluate_sasrec_model(
            model,
            prepared_data,
            split="validation",
            ks=(10, 20),
            batch_size=config.batch_size,
            device=device,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, **validation_metrics}
        )
        print(
            f"epoch={epoch:02d} loss={train_loss:.6f} "
            f"val_R@10={validation_metrics['Recall@10']:.6f} "
            f"val_R@20={validation_metrics['Recall@20']:.6f} "
            f"val_NDCG@10={validation_metrics['NDCG@10']:.6f} "
            f"val_NDCG@20={validation_metrics['NDCG@20']:.6f}"
        )
        if not best_metrics or (
            validation_metrics["Recall@20"] > best_metrics["Recall@20"]
        ):
            best_epoch = epoch
            best_metrics = validation_metrics
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a checkpoint")
    # 把模型恢复到验证指标最好的那一轮，而不是最后一轮。
    model.load_state_dict(best_state_dict)
    checkpoint_path = config.artifacts_dir / config.dataset / "sasrec.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint 同时保存权重、能重建模型的结构配置，以及选轮依据和完整训练历史。
    checkpoint = {
        "model_state_dict": best_state_dict,
        "split_strategy": SPLIT_STRATEGY,
        "dataset": config.dataset,
        "positive_threshold": positive_threshold,
        "model_config": {
            "num_items": prepared_data.num_items,
            "max_seq_len": config.max_seq_len,
            "embedding_dim": config.embedding_dim,
            "num_blocks": config.num_blocks,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
        },
        "train_config": json_ready_config(config),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    print(
        f"Best epoch: {best_epoch}, validation Recall@20: "
        f"{best_metrics['Recall@20']:.6f}"
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return model, checkpoint


def parse_args() -> SASRecTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=None,
        help="defaults to the audited value in recsys.config for the chosen dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return SASRecTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_sasrec(parse_args())
