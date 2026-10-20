"""One-pass KuaiRec threshold audit plus metadata-file inspection.

This command only reads data and writes ``artifacts/kuairec_data_stats.json``.
It does not choose a production threshold or train any recommendation model.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from array import array
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR
from recsys.data.kuairec import MATRIX_REQUIRED_COLUMNS, resolve_kuairec_file


THRESHOLDS = (0.5, 0.8, 1.0, 2.0)
SIDE_FILES = (
    "item_categories.csv",
    "user_features.csv",
    "item_daily_features.csv",
    "kuairec_caption_category.csv",
)


def _grow_2d(values: np.ndarray, required_size: int) -> np.ndarray:
    if required_size <= values.shape[1]:
        return values
    new_size = max(required_size, values.shape[1] * 2)
    return np.pad(values, ((0, 0), (0, new_size - values.shape[1])))


def _grow_latest(
    timestamps: np.ndarray, items: np.ndarray, required_size: int
) -> tuple[np.ndarray, np.ndarray]:
    if required_size <= timestamps.shape[1]:
        return timestamps, items
    new_size = max(required_size, timestamps.shape[1] * 2)
    pad_width = ((0, 0), (0, new_size - timestamps.shape[1]), (0, 0))
    return (
        np.pad(timestamps, pad_width, constant_values=-np.inf),
        np.pad(items, pad_width, constant_values=-1),
    )


def _update_latest_two(
    timestamps: np.ndarray,
    items: np.ndarray,
    threshold_index: int,
    user_id: int,
    timestamp: float,
    item_id: int,
) -> None:
    """Keep the two largest deterministic ``(timestamp, item_id)`` keys."""
    first = (timestamps[threshold_index, user_id, 0], items[threshold_index, user_id, 0])
    second = (
        timestamps[threshold_index, user_id, 1],
        items[threshold_index, user_id, 1],
    )
    candidate = (timestamp, item_id)
    if candidate >= second:
        timestamps[threshold_index, user_id, 0] = second[0]
        items[threshold_index, user_id, 0] = second[1]
        timestamps[threshold_index, user_id, 1] = timestamp
        items[threshold_index, user_id, 1] = item_id
    elif candidate > first:
        timestamps[threshold_index, user_id, 0] = timestamp
        items[threshold_index, user_id, 0] = item_id


def _distribution(nonzero_counts: np.ndarray) -> dict[str, float | int]:
    values = nonzero_counts[nonzero_counts > 0]
    if not values.size:
        return {}
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "min": int(values.min()),
        "mean": float(values.mean()),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": int(values.max()),
    }


def _duplicate_statistics(
    pair_keys_buffer: array, timestamp_ms_buffer: array
) -> tuple[dict[str, int | float], dict[str, int | float]]:
    pair_keys = np.frombuffer(pair_keys_buffer, dtype=np.uint64)
    timestamp_ms = np.frombuffer(timestamp_ms_buffer, dtype=np.int64)
    total = pair_keys.size

    pair_order = np.argsort(pair_keys, kind="stable")
    sorted_pairs = pair_keys[pair_order]
    starts = np.r_[0, np.flatnonzero(sorted_pairs[1:] != sorted_pairs[:-1]) + 1]
    group_sizes = np.diff(np.r_[starts, total])
    repeated_groups_mask = group_sizes > 1
    repeated_pair_groups = int(repeated_groups_mask.sum())
    duplicate_rows = int((group_sizes[repeated_groups_mask] - 1).sum())

    sorted_times = timestamp_ms[pair_order]
    min_times = np.minimum.reduceat(sorted_times, starts)
    max_times = np.maximum.reduceat(sorted_times, starts)
    repeated_different_timestamp = int(
        (repeated_groups_mask & (min_times != max_times)).sum()
    )
    del pair_order, sorted_pairs, sorted_times, min_times, max_times

    users = (pair_keys >> np.uint64(32)).astype(np.uint32)
    time_order = np.lexsort((timestamp_ms, users))
    sorted_users = users[time_order]
    sorted_timestamp_ms = timestamp_ms[time_order]
    same_as_previous = (sorted_users[1:] == sorted_users[:-1]) & (
        sorted_timestamp_ms[1:] == sorted_timestamp_ms[:-1]
    )
    tie_rows_beyond_first = int(same_as_previous.sum())
    tie_group_starts = np.r_[
        0,
        np.flatnonzero(
            (sorted_users[1:] != sorted_users[:-1])
            | (sorted_timestamp_ms[1:] != sorted_timestamp_ms[:-1])
        )
        + 1,
    ]
    timestamp_group_sizes = np.diff(np.r_[tie_group_starts, total])
    tied_groups = int((timestamp_group_sizes > 1).sum())

    duplicate_stats = {
        "unique_user_item_pairs": int(starts.size),
        "repeated_user_item_pair_groups": repeated_pair_groups,
        "duplicate_rows_beyond_first": duplicate_rows,
        "duplicate_row_ratio": duplicate_rows / total,
        "repeated_pair_groups_with_different_timestamps": repeated_different_timestamp,
        "repeated_pair_groups_only_at_same_timestamp": (
            repeated_pair_groups - repeated_different_timestamp
        ),
    }
    timestamp_stats = {
        "repeated_user_timestamp_groups": tied_groups,
        "rows_beyond_first_in_user_timestamp_ties": tie_rows_beyond_first,
        "row_ratio_beyond_first": tie_rows_beyond_first / total,
    }
    return duplicate_stats, timestamp_stats


def analyze_big_matrix(
    matrix_path: Path, thresholds: tuple[float, ...] = THRESHOLDS
) -> dict[str, Any]:
    """Calculate exact threshold/split diagnostics in one CSV pass."""
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    user_counts = np.zeros((len(thresholds), 8192), dtype=np.int64)
    item_counts = np.zeros((len(thresholds), 16384), dtype=np.int64)
    latest_timestamps = np.full((len(thresholds), 8192, 2), -np.inf)
    latest_items = np.full((len(thresholds), 8192, 2), -1, dtype=np.int64)
    positive_counts = np.zeros(len(thresholds), dtype=np.int64)
    pair_keys = array("Q")
    timestamp_milliseconds = array("q")
    raw_users: set[int] = set()
    raw_items: set[int] = set()
    total_rows = 0
    started = time.perf_counter()

    with matrix_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        columns = set(header)
        missing = MATRIX_REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing columns in {matrix_path}: {sorted(missing)}")
        positions = {name: header.index(name) for name in MATRIX_REQUIRED_COLUMNS}

        for total_rows, row in enumerate(reader, start=1):
            user_id = int(row[positions["user_id"]])
            item_id = int(row[positions["video_id"]])
            timestamp = float(row[positions["timestamp"]])
            watch_ratio = float(row[positions["watch_ratio"]])
            if user_id >= user_counts.shape[1]:
                user_counts = _grow_2d(user_counts, user_id + 1)
                latest_timestamps, latest_items = _grow_latest(
                    latest_timestamps, latest_items, user_id + 1
                )
            if item_id >= item_counts.shape[1]:
                item_counts = _grow_2d(item_counts, item_id + 1)

            raw_users.add(user_id)
            raw_items.add(item_id)
            pair_keys.append((user_id << 32) | item_id)
            timestamp_milliseconds.append(round(timestamp * 1000))

            matched_indices = np.flatnonzero(watch_ratio >= threshold_array)
            for threshold_index in matched_indices:
                positive_counts[threshold_index] += 1
                user_counts[threshold_index, user_id] += 1
                item_counts[threshold_index, item_id] += 1
                _update_latest_two(
                    latest_timestamps,
                    latest_items,
                    int(threshold_index),
                    user_id,
                    timestamp,
                    item_id,
                )
            if total_rows % 1_000_000 == 0:
                print(f"big_matrix: {total_rows:,} rows", flush=True)

    duplicate_stats, timestamp_stats = _duplicate_statistics(
        pair_keys, timestamp_milliseconds
    )
    threshold_results: dict[str, Any] = {}
    for threshold_index, threshold in enumerate(thresholds):
        per_user = user_counts[threshold_index]
        per_item = item_counts[threshold_index]
        eligible_users = np.flatnonzero(per_user >= 3)
        validation_items = latest_items[threshold_index, eligible_users, 0]
        test_items = latest_items[threshold_index, eligible_users, 1]
        train_item_counts = per_item.copy()
        np.add.at(train_item_counts, validation_items, -1)
        np.add.at(train_item_counts, test_items, -1)
        cold_validation = train_item_counts[validation_items] == 0
        cold_test = train_item_counts[test_items] == 0
        heldout_ties = (
            latest_timestamps[threshold_index, eligible_users, 0]
            == latest_timestamps[threshold_index, eligible_users, 1]
        )
        positive_count = int(positive_counts[threshold_index])
        threshold_results[str(threshold)] = {
            "positive_interactions": positive_count,
            "positive_ratio": positive_count / total_rows,
            "active_users": int((per_user > 0).sum()),
            "users_with_at_least_3_positives": int(eligible_users.size),
            "active_items": int((per_item > 0).sum()),
            "item_coverage": int((per_item > 0).sum()) / len(raw_items),
            "positive_interactions_per_active_user": _distribution(per_user),
            "positive_interactions_per_active_item": _distribution(per_item),
            "split": {
                "train_interactions": positive_count - 2 * int(eligible_users.size),
                "validation_interactions": int(eligible_users.size),
                "test_interactions": int(eligible_users.size),
                "validation_test_same_timestamp_users": int(heldout_ties.sum()),
                "validation_test_same_timestamp_ratio": (
                    float(heldout_ties.mean()) if eligible_users.size else 0.0
                ),
                "cold_validation_interactions": int(cold_validation.sum()),
                "cold_validation_items": int(
                    np.unique(validation_items[cold_validation]).size
                ),
                "cold_test_interactions": int(cold_test.sum()),
                "cold_test_items": int(np.unique(test_items[cold_test]).size),
            },
        }

    return {
        "path": str(matrix_path),
        "columns": header,
        "interactions": total_rows,
        "users": len(raw_users),
        "items": len(raw_items),
        "density": total_rows / (len(raw_users) * len(raw_items)),
        "duplicates": duplicate_stats,
        "timestamp_ties": timestamp_stats,
        "thresholds": threshold_results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def inspect_csv(path: Path) -> dict[str, Any]:
    """Collect basic schema/cardinality information without interpreting features."""
    row_count = 0
    users: set[str] = set()
    items: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        user_position = header.index("user_id") if "user_id" in header else None
        item_column = "video_id" if "video_id" in header else "item_id"
        item_position = header.index(item_column) if item_column in header else None
        for row_count, row in enumerate(reader, start=1):
            if user_position is not None:
                users.add(row[user_position])
            if item_position is not None:
                items.add(row[item_position])
    result: dict[str, Any] = {"path": str(path), "rows": row_count, "columns": header}
    if user_position is not None:
        result["users"] = len(users)
    if item_position is not None:
        result["items"] = len(items)
    if path.name == "small_matrix.csv" and users and items:
        result["density"] = row_count / (len(users) * len(items))
    return result


def run_analysis(data_dir: Path, output_path: Path) -> dict[str, Any]:
    big_path = resolve_kuairec_file(data_dir, "big_matrix.csv")
    result = {
        "dataset": "KuaiRec 2.0",
        "scope": "historical raw-row threshold diagnostic; split and counts are not the v2 first-exposure protocol",
        "positive_rule": "watch_ratio >= threshold (inclusive)",
        "big_matrix": analyze_big_matrix(big_path),
        "small_matrix": inspect_csv(resolve_kuairec_file(data_dir, "small_matrix.csv")),
        "side_information": {
            filename: inspect_csv(resolve_kuairec_file(data_dir, filename))
            for filename in SIDE_FILES
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR / "kuairec")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "kuairec_data_stats.json",
    )
    args = parser.parse_args()
    result = run_analysis(args.data_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
