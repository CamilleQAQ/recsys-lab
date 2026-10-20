"""把精确的暴力检索与精确的 FAISS IndexFlatIP 检索做对比。"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from recsys.data.movielens import prepare_movielens
from recsys.data.schema import seen_items_for_split
from recsys.evaluation.evaluator import (
    brute_force_top_k,
    evaluate_recommendations,
    get_evaluation_interactions,
    group_items_by_user,
    load_model,
)
from recsys.retrieval.faiss_index import (
    build_faiss_index,
    encode_users_and_items,
    faiss_top_k_from_embeddings,
    top_k_agreement,
)
from recsys.training.trainer import resolve_device


def brute_force_from_embeddings(
    user_indices: list[int],
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    train_items_by_user: dict[int, set[int]],
    k: int,
    user_batch_size: int,
) -> dict[int, list[int]]:
    """分批复用已有的精确矩阵乘法检索。"""
    recommendations: dict[int, list[int]] = {}
    for start in range(0, len(user_indices), user_batch_size):
        batch_users = user_indices[start : start + user_batch_size]
        recommendations.update(
            brute_force_top_k(
                user_indices=batch_users,
                user_embeddings=user_embeddings[start : start + user_batch_size],
                item_embeddings=item_embeddings,
                train_items_by_user=train_items_by_user,
                k=k,
            )
        )
    return recommendations


def run_faiss_evaluation(
    checkpoint_path: Path,
    data_dir: Path,
    split: str = "test",
    ks: tuple[int, ...] = (10, 20),
    user_batch_size: int = 512,
    initial_candidates: int = 100,
    benchmark_repeats: int = 3,
    agreement_users: int = 100,
    seed: int = 42,
    device_name: str = "auto",
) -> dict:
    """校验检索正确性、排序指标、ID 映射，并测量检索耗时。"""
    if benchmark_repeats <= 0:
        raise ValueError("benchmark_repeats must be positive")
    device = resolve_device(device_name)
    prepared_data = prepare_movielens(data_dir)
    model, checkpoint = load_model(checkpoint_path, device)
    model_config = checkpoint["model_config"]
    if model_config["num_users"] != prepared_data.num_users:
        raise ValueError("Checkpoint user vocabulary does not match the data")
    if model_config["num_items"] != prepared_data.num_items:
        raise ValueError("Checkpoint item vocabulary does not match the data")

    evaluation_interactions = get_evaluation_interactions(prepared_data, split)
    evaluation_users = sorted(group_items_by_user(evaluation_interactions))
    train_items_by_user = seen_items_for_split(prepared_data, split)

    # 用户和物品只编码一次，这样计时循环测到的就纯粹是检索耗时。
    user_embeddings, item_embeddings = encode_users_and_items(
        model,
        evaluation_users,
        prepared_data.num_items,
        device,
        user_batch_size,
    )

    build_start = time.perf_counter()
    item_index = build_faiss_index(
        item_embeddings=item_embeddings,
        position_to_item_idx=list(range(prepared_data.num_items)),
        index_to_original_movie_id=prepared_data.mappings.index_to_item,
    )
    index_build_seconds = time.perf_counter() - build_start

    # 先给两种实现各跑一小批，把 CPU 内核预热掉，再开始正式计时。
    warmup_size = min(16, len(evaluation_users))
    brute_force_from_embeddings(
        evaluation_users[:warmup_size],
        user_embeddings[:warmup_size],
        item_embeddings,
        train_items_by_user,
        max(ks),
        warmup_size,
    )
    faiss_top_k_from_embeddings(
        evaluation_users[:warmup_size],
        user_embeddings[:warmup_size],
        item_index,
        train_items_by_user,
        max(ks),
        initial_candidates,
    )

    # 多次重复取中位数，并且只对「检索」计时，不包含编码时间。
    brute_force_times = []
    faiss_times = []
    brute_force_recommendations = {}
    faiss_recommendations = {}
    for _ in range(benchmark_repeats):
        start = time.perf_counter()
        brute_force_recommendations = brute_force_from_embeddings(
            evaluation_users,
            user_embeddings,
            item_embeddings,
            train_items_by_user,
            max(ks),
            user_batch_size,
        )
        brute_force_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        faiss_recommendations = faiss_top_k_from_embeddings(
            evaluation_users,
            user_embeddings,
            item_index,
            train_items_by_user,
            max(ks),
            initial_candidates,
        )
        faiss_times.append(time.perf_counter() - start)

    brute_force_metrics = evaluate_recommendations(
        brute_force_recommendations, evaluation_interactions, ks
    )
    faiss_metrics = evaluate_recommendations(
        faiss_recommendations, evaluation_interactions, ks
    )

    # 用固定种子抽样一部分用户，检查两种检索结果是否一致。
    rng = random.Random(seed)
    sampled_users = rng.sample(
        evaluation_users, min(agreement_users, len(evaluation_users))
    )
    overlap, exact_match = top_k_agreement(
        brute_force_recommendations,
        faiss_recommendations,
        sampled_users,
        max(ks),
    )

    brute_seconds = statistics.median(brute_force_times)
    faiss_seconds = statistics.median(faiss_times)
    num_users = len(evaluation_users)
    benchmark = {
        "num_users": num_users,
        "num_items": prepared_data.num_items,
        "embedding_dim": item_index.embedding_dim,
        "index_build_seconds": index_build_seconds,
        "brute_force_total_seconds": brute_seconds,
        "brute_force_ms_per_user": brute_seconds * 1000 / num_users,
        "faiss_total_seconds": faiss_seconds,
        "faiss_ms_per_user": faiss_seconds * 1000 / num_users,
    }

    sample_user = sampled_users[0]
    sample_internal_items = faiss_recommendations[sample_user][:5]
    sample_original_movie_ids = item_index.to_original_movie_ids(
        sample_internal_items
    )
    print("FAISS index: IndexFlatIP (exact inner-product search, not ANN)")
    print(
        f"Index vectors: {item_index.num_items:,}, "
        f"embedding dimension: {item_index.embedding_dim}, dtype: float32"
    )
    print(
        f"Mapping example: internal item_idx {sample_internal_items} -> "
        f"original movie_id {sample_original_movie_ids}"
    )
    print(
        f"Top-{max(ks)} agreement on {len(sampled_users)} sampled users: "
        f"overlap={overlap:.6f}, exact_order={exact_match:.6f}"
    )
    print("Metrics")
    for method, metrics in (
        ("Brute-force", brute_force_metrics),
        ("FAISS IndexFlatIP", faiss_metrics),
    ):
        values = " ".join(f"{name}={value:.6f}" for name, value in metrics.items())
        print(f"{method}: {values}")
    print("Benchmark (median retrieval-only time)")
    print(
        f"users={num_users:,} items={prepared_data.num_items:,} "
        f"dimension={item_index.embedding_dim} repeats={benchmark_repeats}"
    )
    print(
        f"Brute-force: {brute_seconds:.6f}s total, "
        f"{benchmark['brute_force_ms_per_user']:.6f}ms/user"
    )
    print(
        f"FAISS IndexFlatIP: {faiss_seconds:.6f}s total, "
        f"{benchmark['faiss_ms_per_user']:.6f}ms/user"
    )
    print(f"FAISS one-time index build: {index_build_seconds:.6f}s")

    return {
        "brute_force_metrics": brute_force_metrics,
        "faiss_metrics": faiss_metrics,
        "top_k_overlap": overlap,
        "exact_order_match_rate": exact_match,
        "benchmark": benchmark,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/movielens/two_tower.pt")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--user-batch-size", type=int, default=512)
    parser.add_argument("--initial-candidates", type=int, default=100)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--agreement-users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_faiss_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        user_batch_size=args.user_batch_size,
        initial_candidates=args.initial_candidates,
        benchmark_repeats=args.benchmark_repeats,
        agreement_users=args.agreement_users,
        seed=args.seed,
        device_name=args.device,
    )
