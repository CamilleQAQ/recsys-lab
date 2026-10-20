"""跑一组受控的 Two-Tower 实验，并用验证集选择 checkpoint。"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from recsys.config import SPLIT_STRATEGY
from recsys.data.movielens import MovieLensDataset, prepare_movielens
from recsys.evaluation.evaluator import evaluate_embedding_model
from recsys.models.two_tower import TwoTowerModel
from recsys.training.trainer import (
    build_train_positive_matrix,
    create_dataloader,
    resolve_device,
    train_one_epoch,
)
from recsys.utils.checkpoint import json_ready_config
from recsys.utils.seed import set_random_seed


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts/experiments")
    max_epochs: int = 5
    patience: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    embedding_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 64
    temperature: float = 0.2
    tower_type: str = "mlp"
    mask_false_negatives: bool = False
    seed: int = 42
    device: str = "auto"


def run_experiment(config: ExperimentConfig) -> dict:
    """只用训练集训练，并依据验证集 Recall@20 选择要保存的 checkpoint。"""
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    prepared_data = prepare_movielens(config.data_dir)
    train_loader = create_dataloader(
        MovieLensDataset(prepared_data.train_interactions),
        config.batch_size,
        config.seed,
        shuffle=True,
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
        # 这组实验默认关闭屏蔽，打开时才需要额外构建真值表。
        train_positive_matrix = build_train_positive_matrix(
            prepared_data.train_interactions,
            prepared_data.num_users,
            prepared_data.num_items,
            device,
        )

    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    print(f"Experiment: {config.name}")
    print(
        f"tower={config.tower_type}, dim={config.output_dim}, "
        f"temperature={config.temperature}, mask={config.mask_false_negatives}"
    )
    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            train_positive_matrix=train_positive_matrix,
        )
        validation_metrics = evaluate_embedding_model(
            model=model,
            prepared_data=prepared_data,
            split="validation",
            ks=(10, 20),
            user_batch_size=512,
            device=device,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation_metrics,
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch:02d} loss={train_loss:.6f} "
            f"val_R@10={validation_metrics['Recall@10']:.6f} "
            f"val_R@20={validation_metrics['Recall@20']:.6f} "
            f"val_NDCG@10={validation_metrics['NDCG@10']:.6f}"
        )

        score = validation_metrics["Recall@20"]
        # 以验证集 Recall@20 为准记录当前最好的一轮；连续无提升则早停。
        if not best_metrics or score > best_metrics["Recall@20"]:
            best_epoch = epoch
            best_metrics = validation_metrics
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state_dict is None:
        raise RuntimeError("Experiment did not produce a checkpoint")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / f"{config.name}.pt"
    # 这组实验沿用 experiment_config 这个键名保存训练配置。
    checkpoint = {
        "model_state_dict": best_state_dict,
        "split_strategy": SPLIT_STRATEGY,
        "model_config": {
            "num_users": prepared_data.num_users,
            "num_items": prepared_data.num_items,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "temperature": config.temperature,
            "tower_type": config.tower_type,
        },
        "experiment_config": json_ready_config(config),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    result = {
        "experiment": config.name,
        "best_epoch": best_epoch,
        **best_metrics,
        "checkpoint": str(checkpoint_path),
    }
    with (config.output_dir / f"{config.name}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)
    print(
        f"Best epoch: {best_epoch}, validation Recall@20: "
        f"{best_metrics['Recall@20']:.6f}"
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    return result


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments")
    )
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--tower-type", choices=["embedding", "mlp"], default="mlp")
    parser.add_argument("--mask-false-negatives", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return ExperimentConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_experiment(parse_args())
