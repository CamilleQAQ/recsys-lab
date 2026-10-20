import torch

from recsys.data.movielens import Interaction
from recsys.evaluation.evaluator import (
    brute_force_top_k,
    evaluate_recommendations,
    generate_recommendations,
    group_items_by_user,
)
from recsys.models.matrix_factorization import MatrixFactorization
from recsys.models.two_tower import TwoTowerModel
from recsys.models.popularity import generate_popularity_recommendations, rank_items_by_popularity


def test_brute_force_ranking_filters_training_items():
    user_embeddings = torch.tensor([[1.0, 0.0]])
    item_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ]
    )

    recommendations = brute_force_top_k(
        user_indices=[7],
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        train_items_by_user={7: {0}},
        k=2,
    )

    assert recommendations[7] == [1, 2]
    assert 0 not in recommendations[7]


def test_three_models_use_the_same_metric_pipeline():
    train_interactions = [
        Interaction(0, 0, 1),
        Interaction(1, 1, 1),
    ]
    test_interactions = [
        Interaction(0, 2, 2),
        Interaction(1, 3, 2),
    ]
    seen_items = group_items_by_user(train_interactions)
    users = [0, 1]

    popularity_recommendations = generate_popularity_recommendations(
        users,
        rank_items_by_popularity(train_interactions, num_items=4),
        seen_items,
        k=2,
    )
    mf_recommendations = generate_recommendations(
        MatrixFactorization(2, 4, embedding_dim=4),
        users,
        num_items=4,
        train_items_by_user=seen_items,
        k=2,
        user_batch_size=2,
        device=torch.device("cpu"),
    )
    two_tower_recommendations = generate_recommendations(
        TwoTowerModel(2, 4, embedding_dim=4, hidden_dim=4, output_dim=4),
        users,
        num_items=4,
        train_items_by_user=seen_items,
        k=2,
        user_batch_size=2,
        device=torch.device("cpu"),
    )

    for recommendations in (
        popularity_recommendations,
        mf_recommendations,
        two_tower_recommendations,
    ):
        metrics = evaluate_recommendations(
            recommendations, test_interactions, ks=(1, 2)
        )
        assert set(metrics) == {
            "Recall@1",
            "NDCG@1",
            "HitRate@1",
            "Recall@2",
            "NDCG@2",
            "HitRate@2",
        }
