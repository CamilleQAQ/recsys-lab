from recsys.data.movielens import Interaction
from recsys.models.popularity import (
    generate_popularity_recommendations,
    rank_items_by_popularity,
)


def test_popularity_filters_items_seen_in_train():
    train_interactions = [
        Interaction(0, 0, 1),
        Interaction(1, 0, 1),
        Interaction(2, 0, 1),
        Interaction(0, 1, 2),
        Interaction(1, 1, 2),
        Interaction(2, 2, 2),
    ]
    ranking = rank_items_by_popularity(train_interactions, num_items=4)

    recommendations = generate_popularity_recommendations(
        user_indices=[0],
        popularity_ranking=ranking,
        train_items_by_user={0: {0, 1}},
        k=2,
    )

    assert ranking == [0, 1, 2, 3]
    assert recommendations[0] == [2, 3]
