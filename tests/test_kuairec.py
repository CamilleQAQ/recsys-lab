import csv
import json

import pytest

from recsys.data.kuairec import (
    aggregate_pairs,
    encode_positive_interactions,
    prepare_kuairec_from_file,
    read_matrix,
)
from recsys.data.movielens import prepare_movielens_from_file
from recsys.data.schema import InteractionDataset, PreparedData, seen_items_for_split
from recsys.experiments.analyze_kuairec import analyze_big_matrix


HEADER = (
    "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
)


def write_matrix(path, rows):
    path.write_text(HEADER + "".join(rows), encoding="utf-8")


def test_kuairec_parsing_threshold_and_contiguous_mapping(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(
        matrix_path,
        [
            "20,300,5000,10000,x,x,3.250,0.5\n",
            "10,100,4000,10000,x,x,1.125,0.4\n",
            "10,200,8000,10000,x,x,2.500,0.8\n",
        ],
    )
    rows = read_matrix(matrix_path)
    interactions, mappings = encode_positive_interactions(rows, 0.5)

    # Thresholding is inclusive, and fractional timestamps are not truncated.
    assert len(interactions) == 2
    assert interactions[0].timestamp == 2.5
    assert mappings.user_to_index == {10: 0, 20: 1}
    assert mappings.index_to_item == [200, 300]
    assert {row.user_idx for row in interactions} == {0, 1}
    assert {row.item_idx for row in interactions} == {0, 1}


def test_kuairec_merges_duplicate_pairs_per_user(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(
        matrix_path,
        [
            "1,10,1,1,x,x,4.0,1.0\n",
            "1,20,1,1,x,x,1.0,1.0\n",
            "1,20,1,1,x,x,2.0,1.0\n",
            "1,30,1,1,x,x,3.0,1.0\n",
            "2,40,1,1,x,x,1.0,1.0\n",
            "2,50,1,1,x,x,2.0,0.4\n",
        ],
    )
    prepared = prepare_kuairec_from_file(matrix_path, tmp_path / "processed", 0.8)

    # First exposure is the event: 1.0 → train, 3.0 → validation, 4.0 → test.
    assert [row.timestamp for row in prepared.train_interactions if row.user_idx == 0] == [
        1.0
    ]
    assert [row.timestamp for row in prepared.validation_interactions] == [3.0]
    assert [row.timestamp for row in prepared.test_interactions] == [4.0]
    assert len(prepared.train_interactions) == 2  # 用户 1 的 train + 用户 2（不足 3 条）
    for user_idx in (0, 1):
        seen = {
            row.item_idx
            for row in prepared.train_interactions
            if row.user_idx == user_idx
        }
        held_out = {
            row.item_idx
            for row in prepared.validation_interactions + prepared.test_interactions
            if row.user_idx == user_idx
        }
        # 留出目标不能同时出现在训练历史里，否则会被「已看过就不推荐」过滤掉。
        assert seen.isdisjoint(held_out)


def test_kuairec_positive_pair_with_a_negative_replay(tmp_path):
    """A negative replay must not move or relabel the first positive event."""
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(
        matrix_path,
        [
            "1,10,1,1,x,x,1.0,1.0\n",
            "1,10,1,1,x,x,5.0,0.1\n",
            "1,20,1,1,x,x,2.0,1.0\n",
            "1,30,1,1,x,x,3.0,1.0\n",
        ],
    )
    interactions, _ = encode_positive_interactions(
        read_matrix(matrix_path), 1.0
    )
    pair = [row for row in interactions if row.item_idx == 0][0]
    assert pair.timestamp == 1.0


def test_kuairec_late_positive_replay_is_not_a_new_item(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(matrix_path, [
        "1,10,1,1,x,x,3.0,1.0\n",
        "1,10,1,1,x,x,1.0,0.2\n",
        "1,20,1,1,x,x,2.0,1.0\n",
    ])
    rows = read_matrix(matrix_path)
    for threshold in (0.5, 0.8, 1.0):
        interactions, mapping = encode_positive_interactions(rows, threshold)
        assert mapping.index_to_item == [20]
        assert interactions[0].timestamp == 2.0
    interactions, mapping = encode_positive_interactions(rows, 0.2)
    assert mapping.index_to_item == [10, 20]
    assert [row.timestamp for row in interactions] == [1.0, 2.0]


def test_kuairec_same_time_conflicting_first_label_is_excluded(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(matrix_path, [
        "1,10,1,1,x,x,1.0,0.1\n",
        "1,10,1,1,x,x,1.0,1.0\n",
        "1,20,1,1,x,x,2.0,1.0\n",
    ])
    _, mapping = encode_positive_interactions(read_matrix(matrix_path), 1.0)
    assert mapping.index_to_item == [20]
    _, reversed_mapping = encode_positive_interactions(
        list(reversed(read_matrix(matrix_path))), 1.0
    )
    assert reversed_mapping.index_to_item == [20]


def test_duplicate_audit_counts_replays_before_first_positive(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(matrix_path, [
        "1,10,1,1,x,x,3.0,1.0\n",
        "1,10,1,1,x,x,1.0,0.1\n",
        "1,10,1,1,x,x,2.0,0.2\n",
        "1,20,1,1,x,x,1.0,1.0\n",
        "1,20,1,1,x,x,2.0,0.1\n",
    ])
    _, audit = aggregate_pairs(read_matrix(matrix_path), 1.0)
    assert audit.unique_positive_pairs == 1
    assert audit.duplicate_events_on_positive_pairs == 1
    assert audit.mixed_signal_pairs == 2


def test_kuairec_seen_filter_includes_earlier_negative_playback(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(matrix_path, [
        "1,10,1,1,x,x,1.0,1.0\n",
        "1,20,1,1,x,x,2.0,0.2\n",
        "1,30,1,1,x,x,3.0,1.0\n",
        "1,40,1,1,x,x,4.0,1.0\n",
        "2,20,1,1,x,x,1.0,1.0\n",
        "2,50,1,1,x,x,2.0,1.0\n",
        "2,60,1,1,x,x,3.0,1.0\n",
    ])
    data = prepare_kuairec_from_file(matrix_path, tmp_path / "processed", 1.0)
    protocol = json.loads((tmp_path / "processed" / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["positive_rule"] == "first exposure watch_ratio >= threshold"
    user = data.mappings.user_to_index[1]
    negative_item = data.mappings.item_to_index[20]
    assert negative_item in seen_items_for_split(data, "validation")[user]
    assert negative_item in seen_items_for_split(data, "test")[user]
    for split, targets in (("validation", data.validation_interactions), ("test", data.test_interactions)):
        seen = seen_items_for_split(data, split)
        assert all(row.item_idx not in seen[row.user_idx] for row in targets)


def test_common_model_input_format_is_dataset_independent(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(
        matrix_path,
        [
            "1,10,1,1,x,x,1.1,1.0\n",
            "1,20,1,1,x,x,2.2,1.0\n",
            "1,30,1,1,x,x,3.3,1.0\n",
        ],
    )
    prepared = prepare_kuairec_from_file(matrix_path, tmp_path / "processed", 1.0)
    assert isinstance(prepared, PreparedData)
    ratings_path = tmp_path / "ratings.dat"
    ratings_path.write_text(
        "1::10::5::1\n1::20::5::2\n1::30::5::3\n", encoding="utf-8"
    )
    movielens = prepare_movielens_from_file(
        ratings_path, tmp_path / "movielens_processed"
    )

    for dataset in (prepared, movielens):
        sample = InteractionDataset(dataset.train_interactions)[0]
        assert set(sample) == {"user_idx", "item_idx", "timestamp"}
        assert sample["user_idx"].dtype.is_floating_point is False
        assert sample["item_idx"].dtype.is_floating_point is False
        assert sample["timestamp"].dtype.is_floating_point is True


def test_kuairec_audit_reports_duplicates_ties_and_split(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(
        matrix_path,
        [
            "1,10,1,1,x,x,1.0,2.0\n",
            "1,10,1,1,x,x,2.0,2.0\n",
            "1,20,1,1,x,x,2.0,1.0\n",
            "1,30,1,1,x,x,3.0,0.8\n",
            "2,40,1,1,x,x,1.0,0.5\n",
        ],
    )
    result = analyze_big_matrix(matrix_path)

    assert result["interactions"] == 5
    assert result["duplicates"]["duplicate_rows_beyond_first"] == 1
    assert result["duplicates"]["repeated_pair_groups_with_different_timestamps"] == 1
    assert result["timestamp_ties"]["rows_beyond_first_in_user_timestamp_ties"] == 1
    assert result["thresholds"]["0.8"]["positive_interactions"] == 4
    assert result["thresholds"]["0.8"]["split"] == {
        "train_interactions": 2,
        "validation_interactions": 1,
        "test_interactions": 1,
        "validation_test_same_timestamp_users": 0,
        "validation_test_same_timestamp_ratio": 0.0,
        "cold_validation_interactions": 1,
        "cold_validation_items": 1,
        "cold_test_interactions": 1,
        "cold_test_items": 1,
    }


def test_kuairec_missing_required_column_fails_clearly(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "video_id", "timestamp"])
        writer.writerow([1, 2, 3])
    with pytest.raises(ValueError, match="missing required columns"):
        read_matrix(matrix_path)


def test_kuairec_nonfinite_timestamp_is_rejected(tmp_path):
    matrix_path = tmp_path / "big_matrix.csv"
    write_matrix(matrix_path, ["1,10,1,1,x,x,nan,1.0\n"])
    with pytest.raises(ValueError, match="finite"):
        read_matrix(matrix_path)
    with pytest.raises(ValueError, match="finite"):
        encode_positive_interactions([], float("nan"))
