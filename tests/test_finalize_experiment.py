import pytest

from recsys.experiments.finalize_experiment import get_two_tower_metadata


def test_two_tower_metadata_supports_validation_selected_checkpoint():
    expected_metrics = {"Recall@20": 0.25}
    checkpoint = {
        "experiment_config": {"seed": 42, "max_epochs": 15},
        "best_epoch": 9,
        "best_validation_metrics": expected_metrics,
    }

    config, epoch, metrics, note = get_two_tower_metadata(checkpoint, {})

    assert config["seed"] == 42
    assert epoch == 9
    assert metrics == expected_metrics
    assert "validation" in note


def test_two_tower_metadata_supports_fixed_epoch_checkpoint():
    evaluated_metrics = {"Recall@20": 0.20}
    checkpoint = {"train_config": {"seed": 7, "epochs": 4}}

    config, epoch, metrics, note = get_two_tower_metadata(
        checkpoint, evaluated_metrics
    )

    assert config["seed"] == 7
    assert epoch == 4
    assert metrics == evaluated_metrics
    assert "fixed-epoch" in note


def test_two_tower_metadata_rejects_unknown_checkpoint_schema():
    with pytest.raises(ValueError, match="training configuration"):
        get_two_tower_metadata({}, {})
