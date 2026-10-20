"""SASRec 在验证集 / 测试集上的全量排序评测。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from recsys.config import SPLIT_STRATEGY
from recsys.data.loader import (
    prepare_dataset,
    resolve_checkpoint_dataset,
    resolve_positive_threshold,
)
from recsys.data.schema import PreparedData
from recsys.evaluation.evaluator import evaluate_recommendations
from recsys.models.sasrec import SASRec
from recsys.data.dataset import build_evaluation_examples
from recsys.training.trainer import resolve_device


@torch.no_grad()
def generate_sasrec_recommendations(
    model: SASRec,
    prepared_data: PreparedData,
    split: str,
    k: int,
    batch_size: int,
    device: torch.device,
    include_validation_in_test_history: bool = True,
) -> dict[int, list[int]]:
    """按时间顺序构造历史、给全部物品打分，并过滤掉已看过的物品。"""
    examples = build_evaluation_examples(
        prepared_data,
        split=split,
        max_seq_len=model.max_seq_len,
        include_validation_in_test_history=include_validation_in_test_history,
    )
    model.eval()
    recommendations: dict[int, list[int]] = {}
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        sequences = torch.tensor(
            [example.input_sequence for example in batch_examples],
            dtype=torch.long,
            device=device,
        )
        # sequences: [batch_size, max_seq_len]
        scores = model.score_all_items(sequences)
        # scores: [batch_size, num_items]
        for row, example in enumerate(batch_examples):
            # 该用户已经交互过的物品不能再被推荐，把它们的分数压到 -inf。
            seen_items = torch.tensor(
                sorted(example.seen_item_indices),
                dtype=torch.long,
                device=device,
            )
            scores[row, seen_items] = -torch.inf
        top_items = torch.topk(scores, k=k, dim=1).indices.cpu().tolist()
        for example, item_indices in zip(batch_examples, top_items, strict=True):
            recommendations[example.user_idx] = item_indices
    return recommendations


def evaluate_sasrec_model(
    model: SASRec,
    prepared_data: PreparedData,
    split: str,
    ks: Sequence[int],
    batch_size: int,
    device: torch.device,
    include_validation_in_test_history: bool = True,
) -> dict[str, float]:
    """为指定切分的每个评测用户生成推荐并计算指标。"""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    recommendations = generate_sasrec_recommendations(
        model,
        prepared_data,
        split,
        max(ks),
        batch_size,
        device,
        include_validation_in_test_history,
    )
    evaluation_interactions = (
        prepared_data.validation_interactions
        if split == "validation"
        else prepared_data.test_interactions
    )
    return evaluate_recommendations(recommendations, evaluation_interactions, ks)


def load_sasrec_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[SASRec, dict]:
    """按 checkpoint 里保存的配置把 SASRec 重建出来。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run python -m recsys.training.train_sasrec first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("split_strategy") != SPLIT_STRATEGY:
        raise ValueError("SASRec checkpoint uses an incompatible data split")
    model = SASRec(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_sasrec_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    split: str = "test",
    ks: Sequence[int] = (10, 20),
    batch_size: int = 512,
    device_name: str = "auto",
    include_validation_in_test_history: bool = True,
) -> dict[str, float]:
    """评测一个已经固定的 SASRec checkpoint，不训练也不调参。"""
    device = resolve_device(device_name)
    model, checkpoint = load_sasrec_model(checkpoint_path, device)
    # 数据集与阈值从 checkpoint 读回来，避免拿 MovieLens 的词表去评 KuaiRec。
    dataset = resolve_checkpoint_dataset(checkpoint)
    positive_threshold = resolve_positive_threshold(
        dataset, checkpoint.get("positive_threshold")
    )
    prepared_data = prepare_dataset(
        dataset=dataset,
        data_dir=data_dir,
        positive_threshold=positive_threshold,
    )
    if checkpoint["model_config"]["num_items"] != prepared_data.num_items:
        raise ValueError("SASRec item vocabulary does not match the data")
    metrics = evaluate_sasrec_model(
        model,
        prepared_data,
        split,
        ks,
        batch_size,
        device,
        include_validation_in_test_history,
    )
    history = (
        "train + validation"
        if split == "test" and include_validation_in_test_history
        else "train"
    )
    print(f"Dataset: {dataset} (positive threshold {positive_threshold:g})")
    print(f"Evaluation split: chronological {split}")
    print(f"Model input history: {history}")
    print("Seen-item filter: all observed exposures before target time")
    print("Target interaction is excluded from the model input")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/movielens/sasrec.pt")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--train-history-only",
        action="store_true",
        help=(
            "Supplementary test protocol; omit to use the standard "
            "train+validation history."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_sasrec_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        ks=args.ks,
        batch_size=args.batch_size,
        device_name=args.device,
        include_validation_in_test_history=not args.train_history_only,
    )
