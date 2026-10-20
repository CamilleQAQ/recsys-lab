"""在按时间切分的测试集上评测全部已固定的模型 checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path

from recsys.experiments.compare import compare_models, print_comparison_table
from recsys.data.loader import (
    prepare_dataset,
    resolve_checkpoint_dataset,
    resolve_positive_threshold,
)
from recsys.evaluation.sasrec_evaluator import evaluate_sasrec_model, load_sasrec_model
from recsys.training.trainer import resolve_device


def compare_with_sasrec(
    data_dir: Path,
    two_tower_checkpoint: Path,
    mf_checkpoint: Path,
    sasrec_checkpoint: Path,
    batch_size: int = 512,
    device_name: str = "auto",
) -> dict[str, dict[str, float]]:
    """评测四个已经固定的 checkpoint，不重新训练也不调参。"""
    ks = (10, 20)
    device = resolve_device(device_name)
    results = compare_models(
        data_dir=data_dir,
        two_tower_checkpoint=two_tower_checkpoint,
        mf_checkpoint=mf_checkpoint,
        split="test",
        ks=ks,
        user_batch_size=batch_size,
        device_name=device_name,
        print_results=False,
    )
    sasrec_model, checkpoint = load_sasrec_model(sasrec_checkpoint, device)
    dataset = resolve_checkpoint_dataset(checkpoint)
    prepared_data = prepare_dataset(
        dataset=dataset,
        data_dir=data_dir,
        positive_threshold=resolve_positive_threshold(
            dataset, checkpoint.get("positive_threshold")
        ),
    )
    if checkpoint["model_config"]["num_items"] != prepared_data.num_items:
        raise ValueError("SASRec item vocabulary does not match the data")
    results["SASRec"] = evaluate_sasrec_model(
        sasrec_model,
        prepared_data,
        split="test",
        ks=ks,
        batch_size=batch_size,
        device=device,
    )

    print()
    print_comparison_table(results, ks, "Final test comparison")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--two-tower-checkpoint",
        type=Path,
        default=Path("artifacts/movielens/two_tower.pt"),
    )
    parser.add_argument(
        "--mf-checkpoint",
        type=Path,
        default=Path("artifacts/movielens/matrix_factorization.pt"),
    )
    parser.add_argument(
        "--sasrec-checkpoint",
        type=Path,
        default=Path("artifacts/movielens/sasrec.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_with_sasrec(
        data_dir=args.data_dir,
        two_tower_checkpoint=args.two_tower_checkpoint,
        mf_checkpoint=args.mf_checkpoint,
        sasrec_checkpoint=args.sasrec_checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
    )
