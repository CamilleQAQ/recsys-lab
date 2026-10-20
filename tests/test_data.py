import csv
import json
import pytest

from torch.utils.data import DataLoader

from recsys.data.movielens import MovieLensDataset, prepare_movielens_from_file
from recsys.data.loader import prepare_dataset
from recsys.data.schema import seen_items_for_split


def test_dataset_reads_movielens_and_keeps_valid_indices(tmp_path):
    ratings_path = tmp_path / "ratings.dat"
    ratings_path.write_text(
        "10::100::5::1000\n"
        "10::200::4::2000\n"
        "10::300::5::3000\n"
        "20::100::5::1100\n"
        "20::300::3::1200\n"
        "20::400::5::2100\n"
        "20::500::4::3100\n",
        encoding="utf-8",
    )

    prepared = prepare_movielens_from_file(
        ratings_path, tmp_path / "processed"
    )
    dataset = MovieLensDataset(prepared.train_interactions)
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert len(prepared.train_interactions) == 2
    assert len(prepared.validation_interactions) == 2
    assert len(prepared.test_interactions) == 2
    assert batch["user_idx"].min().item() >= 0
    assert batch["user_idx"].max().item() < prepared.num_users
    assert batch["item_idx"].min().item() >= 0
    assert batch["item_idx"].max().item() < prepared.num_items
    assert batch["timestamp"].shape == (2,)

    mappings = json.loads(
        (tmp_path / "processed" / "id_mappings.json").read_text(encoding="utf-8")
    )
    assert mappings["user"]["original_to_index"]["10"] == 0
    assert mappings["user"]["index_to_original"][0] == 10
    assert mappings["item"]["original_to_index"]["500"] == 4
    assert mappings["item"]["index_to_original"][4] == 500
    protocol = json.loads((tmp_path / "processed" / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["split_strategy"] == "per_user_last_two_strict_timestamp_v2"
    assert protocol["positive_threshold"] == 4

    with (tmp_path / "processed" / "interactions.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        saved_rows = list(csv.DictReader(file))
    assert set(saved_rows[0]) == {"user_idx", "item_idx", "timestamp", "split"}
    assert {row["split"] for row in saved_rows} == {"train", "validation", "test"}


def test_temporal_train_validation_test_split_has_no_overlap(tmp_path):
    ratings_path = tmp_path / "ratings.dat"
    ratings_path.write_text(
        "1::1::5::400\n1::2::5::100\n1::3::5::300\n1::4::5::200\n",
        encoding="utf-8",
    )
    prepared = prepare_movielens_from_file(ratings_path, tmp_path / "processed")

    assert [row.timestamp for row in prepared.train_interactions] == [100, 200]
    assert [row.timestamp for row in prepared.validation_interactions] == [300]
    assert [row.timestamp for row in prepared.test_interactions] == [400]

    train_rows = set(prepared.train_interactions)
    validation_rows = set(prepared.validation_interactions)
    test_rows = set(prepared.test_interactions)
    assert train_rows.isdisjoint(validation_rows)
    assert train_rows.isdisjoint(test_rows)
    assert validation_rows.isdisjoint(test_rows)


def test_users_with_fewer_than_three_positives_stay_in_train(tmp_path):
    ratings_path = tmp_path / "ratings.dat"
    ratings_path.write_text(
        "1::1::5::100\n1::2::4::200\n",
        encoding="utf-8",
    )

    prepared = prepare_movielens_from_file(ratings_path, tmp_path / "processed")

    assert len(prepared.train_interactions) == 2
    assert prepared.validation_interactions == []
    assert prepared.test_interactions == []


def test_same_second_boundary_is_never_ordered_by_item_id(tmp_path):
    ratings_path = tmp_path / "ratings.dat"
    ratings_path.write_text(
        "1::1::5::100\n1::2::5::200\n1::3::5::300\n1::4::5::300\n"
        "2::1::5::100\n2::2::5::200\n2::3::5::300\n",
        encoding="utf-8",
    )
    prepared = prepare_movielens_from_file(ratings_path, tmp_path / "processed")
    assert [row.user_idx for row in prepared.validation_interactions] == [1]
    assert [row.user_idx for row in prepared.test_interactions] == [1]
    assert sum(row.user_idx == 0 for row in prepared.train_interactions) == 4
    assert prepared.validation_interactions[0].timestamp < prepared.test_interactions[0].timestamp


def test_fractional_movielens_threshold_is_not_truncated(tmp_path):
    ratings_path = tmp_path / "raw" / "ml-1m" / "ratings.dat"
    ratings_path.parent.mkdir(parents=True)
    ratings_path.write_text("1::10::4::1\n1::20::5::2\n", encoding="utf-8")
    prepared = prepare_dataset("movielens", tmp_path, positive_threshold=4.5)
    assert prepared.mappings.index_to_item == [20]
    with pytest.raises(ValueError, match="finite"):
        prepare_dataset("movielens", tmp_path, positive_threshold=float("nan"))


def test_seen_filter_uses_only_ratings_before_each_target(tmp_path):
    ratings_path = tmp_path / "ratings.dat"
    ratings_path.write_text(
        "1::10::5::1\n1::20::5::3\n1::30::5::5\n"
        "1::40::2::2\n1::50::2::4\n"
        "2::40::5::1\n2::50::5::2\n2::60::5::3\n",
        encoding="utf-8",
    )
    prepared = prepare_movielens_from_file(ratings_path, tmp_path / "processed")
    user = prepared.mappings.user_to_index[1]
    item = prepared.mappings.item_to_index
    assert seen_items_for_split(prepared, "validation")[user] == {item[10], item[40]}
    assert seen_items_for_split(prepared, "test")[user] == {item[10], item[20], item[40], item[50]}
    for split, targets in (("validation", prepared.validation_interactions), ("test", prepared.test_interactions)):
        seen = seen_items_for_split(prepared, split)
        assert all(row.item_idx not in seen[row.user_idx] for row in targets)
