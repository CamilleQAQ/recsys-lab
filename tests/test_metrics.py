import pytest

from recsys.evaluation.metrics import compute_ranking_metrics, hit_rate_at_k, ndcg_at_k, recall_at_k


@pytest.fixture
def ranking_example():
    recommendations = {
        0: [1, 2, 3],
        1: [4, 5, 6],
    }
    ground_truth = {
        0: {1, 3},
        1: {5},
    }
    return recommendations, ground_truth


def test_recall_at_k(ranking_example):
    recommendations, ground_truth = ranking_example

    assert recall_at_k(recommendations, ground_truth, k=1) == pytest.approx(0.25)
    assert recall_at_k(recommendations, ground_truth, k=2) == pytest.approx(0.75)


def test_hit_rate_at_k(ranking_example):
    recommendations, ground_truth = ranking_example

    assert hit_rate_at_k(recommendations, ground_truth, k=1) == pytest.approx(0.5)
    assert hit_rate_at_k(recommendations, ground_truth, k=2) == pytest.approx(1.0)


def test_ndcg_at_k(ranking_example):
    recommendations, ground_truth = ranking_example

    assert ndcg_at_k(recommendations, ground_truth, k=1) == pytest.approx(0.5)
    assert ndcg_at_k(recommendations, ground_truth, k=2) == pytest.approx(
        0.622038473168458
    )


def test_compute_ranking_metrics_returns_all_requested_cutoffs(ranking_example):
    recommendations, ground_truth = ranking_example

    metrics = compute_ranking_metrics(recommendations, ground_truth, ks=(1, 2))

    assert set(metrics) == {
        "Recall@1",
        "NDCG@1",
        "HitRate@1",
        "Recall@2",
        "NDCG@2",
        "HitRate@2",
    }
