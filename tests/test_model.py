import torch

from recsys.models.two_tower import TwoTowerModel
from recsys.training.losses import in_batch_cross_entropy_loss


def test_tower_and_similarity_shapes_and_normalization():
    model = TwoTowerModel(
        num_users=5,
        num_items=7,
        embedding_dim=8,
        hidden_dim=12,
        output_dim=6,
    )
    user_ids = torch.tensor([0, 1, 2, 3])
    item_ids = torch.tensor([1, 2, 3, 4])

    user_vectors = model.encode_users(user_ids)
    item_vectors = model.encode_items(item_ids)
    logits = model(user_ids, item_ids)

    assert user_vectors.shape == (4, 6)
    assert item_vectors.shape == (4, 6)
    assert logits.shape == (4, 4)
    assert torch.allclose(user_vectors.norm(dim=1), torch.ones(4), atol=1e-6)
    assert torch.allclose(item_vectors.norm(dim=1), torch.ones(4), atol=1e-6)


def test_loss_can_backward():
    model = TwoTowerModel(num_users=4, num_items=4, output_dim=8)
    ids = torch.arange(4)

    loss = in_batch_cross_entropy_loss(model(ids, ids))
    loss.backward()

    assert loss.item() > 0
    assert model.user_tower.embedding.weight.grad is not None
    assert model.item_tower.embedding.weight.grad is not None


def test_false_negative_mask_keeps_diagonal_and_masks_other_positives():
    logits = torch.tensor(
        [[3.0, 3.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        requires_grad=True,
    )
    false_negative_mask = torch.tensor(
        [[True, True, False], [False, True, False], [False, False, True]]
    )

    masked_loss = in_batch_cross_entropy_loss(logits, false_negative_mask)
    unmasked_loss = in_batch_cross_entropy_loss(logits)
    masked_loss.backward()

    assert masked_loss < unmasked_loss
    assert logits.grad is not None


def test_embedding_only_tower_output_shape():
    model = TwoTowerModel(
        num_users=4,
        num_items=5,
        output_dim=16,
        tower_type="embedding",
    )

    assert model.encode_users(torch.arange(4)).shape == (4, 16)
    assert model.encode_items(torch.arange(5)).shape == (5, 16)
