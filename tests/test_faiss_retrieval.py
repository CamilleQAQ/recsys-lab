import numpy as np
import pytest
pytest.importorskip("faiss")
import torch
from torch.nn import functional as F

from recsys.data.movielens import Interaction
from recsys.evaluation.evaluator import brute_force_top_k, evaluate_recommendations
from recsys.retrieval.faiss_index import (
    build_faiss_index,
    faiss_top_k_from_embeddings,
    top_k_agreement,
)


def test_index_shape_search_shape_and_item_mapping():
    # FAISS position 0 deliberately represents internal item_idx 2.
    item_embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    item_index = build_faiss_index(
        item_embeddings,
        position_to_item_idx=[2, 0, 1],
        index_to_original_movie_id=[100, 200, 300],
    )

    scores, internal_item_indices = item_index.search(
        np.array([[1.0, 0.0]], dtype=np.float32), k=2
    )

    assert item_index.num_items == 3
    assert item_index.embedding_dim == 2
    assert scores.shape == (1, 2)
    assert internal_item_indices.shape == (1, 2)
    assert internal_item_indices[0, 0] == 2
    assert item_index.to_original_movie_ids([2, 0]) == [300, 100]


def test_faiss_seen_item_filtering_overfetches_candidates():
    item_embeddings = F.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]]),
        dim=1,
    )
    item_index = build_faiss_index(
        item_embeddings,
        position_to_item_idx=range(4),
        index_to_original_movie_id=[10, 20, 30, 40],
    )

    recommendations = faiss_top_k_from_embeddings(
        user_indices=[7],
        user_embeddings=torch.tensor([[1.0, 0.0]]),
        item_index=item_index,
        train_items_by_user={7: {0, 1}},
        k=2,
        initial_candidates=2,
    )

    assert recommendations[7] == [2, 3]
    assert not set(recommendations[7]).intersection({0, 1})


def test_faiss_and_brute_force_top_k_are_identical():
    generator = torch.Generator().manual_seed(11)
    user_embeddings = F.normalize(torch.randn(5, 8, generator=generator), dim=1)
    item_embeddings = F.normalize(torch.randn(20, 8, generator=generator), dim=1)
    user_indices = list(range(5))
    seen_items = {user_idx: {user_idx, user_idx + 5} for user_idx in user_indices}
    item_index = build_faiss_index(
        item_embeddings,
        position_to_item_idx=range(20),
        index_to_original_movie_id=list(range(100, 120)),
    )

    brute_force = brute_force_top_k(
        user_indices, user_embeddings, item_embeddings, seen_items, k=5
    )
    faiss_recommendations = faiss_top_k_from_embeddings(
        user_indices,
        user_embeddings,
        item_index,
        seen_items,
        k=5,
        initial_candidates=5,
    )
    overlap, exact_order = top_k_agreement(
        brute_force, faiss_recommendations, user_indices, k=5
    )

    assert overlap == pytest.approx(1.0)
    assert exact_order == pytest.approx(1.0)


def test_faiss_recommendations_use_existing_evaluation_metrics():
    recommendations = {0: [2, 3], 1: [0, 3]}
    test_interactions = [Interaction(0, 2, 10), Interaction(1, 3, 10)]

    metrics = evaluate_recommendations(recommendations, test_interactions, ks=(1, 2))

    assert metrics["Recall@1"] == pytest.approx(0.5)
    assert metrics["Recall@2"] == pytest.approx(1.0)
    assert metrics["HitRate@2"] == pytest.approx(1.0)
