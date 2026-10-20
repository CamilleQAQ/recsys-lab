import torch
import pytest

from recsys.data.movielens import IdMappings, Interaction, PreparedData
from recsys.data.dataset import (
    SASRecTrainDataset,
    build_evaluation_examples,
    pad_item_sequence,
)


def make_prepared_data() -> PreparedData:
    mappings = IdMappings(
        user_to_index={1: 0},
        index_to_user=[1],
        item_to_index={10: 0, 11: 1, 12: 2, 13: 3, 14: 4},
        index_to_item=[10, 11, 12, 13, 14],
    )
    return PreparedData(
        train_interactions=[
            Interaction(0, 0, 1),
            Interaction(0, 1, 2),
        ],
        validation_interactions=[Interaction(0, 2, 3)],
        test_interactions=[Interaction(0, 3, 4)],
        mappings=mappings,
    )


def test_padding_and_truncation_keep_newest_items():
    assert pad_item_sequence([0, 2], max_seq_len=4) == [1, 3, 0, 0]
    assert pad_item_sequence([0, 1, 2, 3], max_seq_len=3) == [2, 3, 4]


def test_training_sequence_has_aligned_targets_and_unseen_negatives():
    interactions = [
        Interaction(0, 0, 1),
        Interaction(0, 1, 2),
        Interaction(0, 2, 3),
        Interaction(0, 3, 4),
    ]
    dataset = SASRecTrainDataset(
        interactions, num_items=6, max_seq_len=3, seed=7
    )

    sample = dataset[0]

    assert sample["input_sequence"].tolist() == [1, 2, 3]
    assert sample["positive_sequence"].tolist() == [2, 3, 4]
    # Tokens 1..4 are the user's train positives, so negatives cannot use them.
    assert set(sample["negative_sequence"].tolist()).isdisjoint({1, 2, 3, 4})


def test_validation_and_test_histories_do_not_use_future_items():
    prepared_data = make_prepared_data()

    validation = build_evaluation_examples(
        prepared_data, split="validation", max_seq_len=4
    )[0]
    test = build_evaluation_examples(
        prepared_data, split="test", max_seq_len=4
    )[0]
    controlled_test = build_evaluation_examples(
        prepared_data,
        split="test",
        max_seq_len=4,
        include_validation_in_test_history=False,
    )[0]

    # Validation sees train only and predicts item_idx 2.
    assert validation.input_sequence == [1, 2, 0, 0]
    assert validation.seen_item_indices == {0, 1}
    assert validation.target_item_idx == 2
    assert validation.target_item_idx not in validation.seen_item_indices
    # Test may use the already-observed validation item, but never its own target.
    assert test.input_sequence == [1, 2, 3, 0]
    assert test.seen_item_indices == {0, 1, 2}
    assert test.target_item_idx == 3
    assert test.target_item_idx not in test.seen_item_indices
    # The supplementary controlled protocol deliberately keeps train history only.
    assert controlled_test.input_sequence == [1, 2, 0, 0]
    assert controlled_test.seen_item_indices == {0, 1, 2}
    assert controlled_test.target_item_idx == 3


def test_short_sequence_is_right_padded():
    dataset = SASRecTrainDataset(
        [Interaction(0, 1, 1), Interaction(0, 3, 2)],
        num_items=5,
        max_seq_len=4,
    )
    sample = dataset[0]

    assert torch.equal(sample["input_sequence"], torch.tensor([2, 0, 0, 0]))
    assert torch.equal(sample["positive_sequence"], torch.tensor([4, 0, 0, 0]))
    assert torch.count_nonzero(sample["negative_sequence"]).item() == 1


def test_same_timestamp_does_not_create_next_item_training_target():
    dataset = SASRecTrainDataset(
        [Interaction(0, 0, 1), Interaction(0, 1, 1), Interaction(0, 2, 2)],
        num_items=5,
        max_seq_len=3,
    )
    sample = dataset[0]
    assert sample["positive_sequence"].tolist() == [0, 3, 0]
    assert sample["negative_sequence"][0].item() == 0


def test_evaluation_rejects_same_second_history():
    prepared = make_prepared_data()
    prepared.validation_interactions[0] = Interaction(0, 2, 2)
    with pytest.raises(ValueError, match="strictly precede"):
        build_evaluation_examples(prepared, "validation", max_seq_len=4)
