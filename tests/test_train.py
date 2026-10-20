import torch

from recsys.data.movielens import Interaction, MovieLensDataset
from recsys.models.two_tower import TwoTowerModel
from recsys.training.losses import in_batch_cross_entropy_loss
from recsys.training.trainer import create_dataloader, train_one_epoch
from recsys.utils.seed import set_random_seed


def test_short_training_flow_runs_and_reduces_loss():
    set_random_seed(7)
    interactions = [Interaction(user_idx=i, item_idx=i, timestamp=i) for i in range(8)]
    dataset = MovieLensDataset(interactions)
    dataloader = create_dataloader(dataset, batch_size=8, seed=7, shuffle=False)
    model = TwoTowerModel(
        num_users=8,
        num_items=8,
        embedding_dim=8,
        hidden_dim=16,
        output_dim=8,
        temperature=0.2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    batch = next(iter(dataloader))
    with torch.no_grad():
        initial_loss = in_batch_cross_entropy_loss(
            model(batch["user_idx"], batch["item_idx"])
        ).item()

    for _ in range(20):
        train_one_epoch(model, dataloader, optimizer, torch.device("cpu"))

    with torch.no_grad():
        final_loss = in_batch_cross_entropy_loss(
            model(batch["user_idx"], batch["item_idx"])
        ).item()

    assert final_loss < initial_loss
