import torch
from torch.nn import functional as F

from recsys.models.matrix_factorization import MatrixFactorization


def test_mf_forward_shape_and_loss_backward():
    model = MatrixFactorization(num_users=4, num_items=6, embedding_dim=8)
    user_ids = torch.tensor([0, 1, 2, 3])
    item_ids = torch.tensor([1, 2, 3, 4])

    logits = model(user_ids, item_ids)
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()

    assert logits.shape == (4,)
    assert model.user_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad is not None
