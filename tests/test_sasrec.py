import torch
from torch.nn import functional as F

from recsys.data.movielens import IdMappings, Interaction, PreparedData
from recsys.evaluation.sasrec_evaluator import (
    evaluate_sasrec_model,
    generate_sasrec_recommendations,
)
from recsys.models.sasrec import SASRec, build_causal_attention_mask


def make_prepared_data() -> PreparedData:
    mappings = IdMappings(
        user_to_index={1: 0, 2: 1},
        index_to_user=[1, 2],
        item_to_index={10: 0, 11: 1, 12: 2, 13: 3, 14: 4},
        index_to_item=[10, 11, 12, 13, 14],
    )
    return PreparedData(
        train_interactions=[
            Interaction(0, 0, 1),
            Interaction(0, 1, 2),
            Interaction(1, 1, 1),
            Interaction(1, 2, 2),
        ],
        validation_interactions=[Interaction(0, 2, 3), Interaction(1, 3, 3)],
        test_interactions=[Interaction(0, 3, 4), Interaction(1, 4, 4)],
        mappings=mappings,
    )


def make_model() -> SASRec:
    return SASRec(
        num_items=5,
        max_seq_len=4,
        embedding_dim=8,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
    )


def test_causal_mask_blocks_future_positions():
    mask = build_causal_attention_mask(4, torch.device("cpu"))

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        ),
    )


def test_forward_shape_padding_and_loss_backward():
    model = make_model()
    sequence = torch.tensor([[1, 2, 3, 0], [2, 4, 0, 0]])

    hidden_states = model(sequence)
    scores = model.score_all_items(sequence)
    loss = F.binary_cross_entropy_with_logits(
        scores[:, 0], torch.ones(2)
    )
    loss.backward()

    assert hidden_states.shape == (2, 4, 8)
    assert scores.shape == (2, 5)
    assert torch.equal(hidden_states[0, 3], torch.zeros(8))
    assert torch.equal(hidden_states[1, 2:], torch.zeros(2, 8))
    assert model.item_embedding.weight.grad is not None
    assert torch.isfinite(loss)


def test_future_items_do_not_change_earlier_hidden_states():
    model = make_model().eval()
    first_sequence = torch.tensor([[1, 2, 3, 4]])
    second_sequence = torch.tensor([[1, 2, 5, 3]])

    with torch.no_grad():
        first_hidden = model(first_sequence)
        second_hidden = model(second_sequence)

    assert torch.allclose(first_hidden[:, :2], second_hidden[:, :2], atol=1e-6)


def test_sasrec_uses_existing_evaluation_pipeline_and_filters_history():
    prepared_data = make_prepared_data()
    model = make_model()

    recommendations = generate_sasrec_recommendations(
        model,
        prepared_data,
        split="validation",
        k=2,
        batch_size=2,
        device=torch.device("cpu"),
    )
    metrics = evaluate_sasrec_model(
        model,
        prepared_data,
        split="validation",
        ks=(1, 2),
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert set(recommendations[0]).isdisjoint({0, 1})
    assert set(recommendations[1]).isdisjoint({1, 2})
    assert set(metrics) == {
        "Recall@1",
        "NDCG@1",
        "HitRate@1",
        "Recall@2",
        "NDCG@2",
        "HitRate@2",
    }
