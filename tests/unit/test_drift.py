"""Drift injection, drift detection (Evidently) and the retraining trigger."""

from datetime import UTC, datetime, timedelta

import pytest

from mlops_core.config import feature_columns
from mlops_core.drift import inject_drift, should_retrain


def test_inject_drift_shifts_only_configured_columns(splits, cfg):
    _, test_df = splits
    drifted = inject_drift(test_df, cfg, seed=1)
    drift_cols = set(cfg["drift"]["inject"])
    for col in feature_columns(cfg):
        if col in drift_cols:
            assert abs(drifted[col].mean() - test_df[col].mean()) > 0.05 * abs(test_df[col].mean())
        else:
            assert drifted[col].equals(test_df[col])
    assert drifted is not test_df  # original untouched


def test_evidently_flags_injected_drift_but_not_clean_data(splits, cfg):
    pytest.importorskip("evidently")
    from mlops_core.drift import compute_drift

    train_df, test_df = splits
    cols = feature_columns(cfg)
    threshold = cfg["drift"]["share_threshold"]

    clean = compute_drift(train_df, test_df, cols, threshold)
    drifted = compute_drift(train_df, inject_drift(test_df, cfg, seed=1), cols, threshold)

    assert clean["dataset_drift"] is False
    assert drifted["dataset_drift"] is True
    assert drifted["n_drifted_columns"] >= len(cfg["drift"]["inject"]) - 1


def test_drift_triggers_retraining():
    now = datetime.now(UTC)
    assert should_retrain(True, None, now, cooldown_minutes=15) is True


def test_no_drift_never_triggers_retraining():
    now = datetime.now(UTC)
    assert should_retrain(False, None, now, cooldown_minutes=15) is False


def test_retraining_respects_cooldown():
    now = datetime.now(UTC)
    assert should_retrain(True, now - timedelta(minutes=5), now, cooldown_minutes=15) is False
    assert should_retrain(True, now - timedelta(minutes=20), now, cooldown_minutes=15) is True
