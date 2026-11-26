"""Drift injection (used by the traffic simulator), drift detection with
Evidently (used by the monitor) and the retraining-trigger rule."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from mlops_core.config import load_config


def inject_drift(df: pd.DataFrame, cfg: dict[str, Any] | None = None, seed: int = 0) -> pd.DataFrame:
    """Return a copy of df with the configured features shifted/scaled plus a
    little noise, simulating a change in the incoming population."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    drifted = df.copy()
    for col, spec in cfg["drift"]["inject"].items():
        if col not in drifted.columns:
            continue
        values = drifted[col].astype(float)
        noise = rng.normal(0, values.std() * 0.05 if values.std() > 0 else 0.0, len(values))
        drifted[col] = values * spec.get("scale", 1.0) + spec.get("shift", 0.0) + noise
    return drifted


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    share_threshold: float,
) -> dict[str, Any]:
    """Run Evidently's dataset drift check on the given columns."""
    from evidently.metrics import DatasetDriftMetric
    from evidently.report import Report

    report = Report(metrics=[DatasetDriftMetric(columns=columns, drift_share=share_threshold)])
    report.run(reference_data=reference[columns], current_data=current[columns])
    result = report.as_dict()["metrics"][0]["result"]
    return {
        "dataset_drift": bool(result["dataset_drift"]),
        "drift_share": float(result["share_of_drifted_columns"]),
        "n_drifted_columns": int(result["number_of_drifted_columns"]),
        "n_columns": int(result["number_of_columns"]),
    }


def compute_classification_quality(current: pd.DataFrame, target: str, prediction: str) -> dict[str, float]:
    """Run Evidently's classification quality metric on rows that have ground truth."""
    from evidently import ColumnMapping
    from evidently.metrics import ClassificationQualityMetric
    from evidently.report import Report

    labelled = current.dropna(subset=[target])
    if labelled.empty or labelled[target].nunique() < 2:
        return {}
    labelled = labelled.astype({target: int, prediction: int})
    report = Report(metrics=[ClassificationQualityMetric()])
    report.run(
        reference_data=None,
        current_data=labelled[[target, prediction]],
        column_mapping=ColumnMapping(target=target, prediction=prediction),
    )
    cur = report.as_dict()["metrics"][0]["result"]["current"]
    return {k: float(cur[k]) for k in ("accuracy", "precision", "recall", "f1") if cur.get(k) is not None}


def should_retrain(
    dataset_drift: bool,
    last_triggered: datetime | None,
    now: datetime,
    cooldown_minutes: int,
) -> bool:
    """Trigger retraining when drift is detected, at most once per cooldown."""
    if not dataset_drift:
        return False
    if last_triggered is None:
        return True
    return now - last_triggered >= timedelta(minutes=cooldown_minutes)
