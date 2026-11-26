"""Monitoring job.

Every MONITOR_INTERVAL_SECONDS:
  1. read the most recent predictions from Postgres
  2. Evidently: data drift vs. the training reference + classification quality
     on predictions that have ground truth
  3. write the metrics to the drift_metrics table (Grafana reads it)
  4. if drift is detected (and the cooldown has passed) trigger the Prefect
     deployment "training-pipeline/retrain"
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text

from mlops_core.config import feature_columns, load_config, target_column
from mlops_core.db import database_url, ensure_schema
from mlops_core.drift import compute_classification_quality, compute_drift, should_retrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("monitor")

SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
INTERVAL = int(os.environ.get("MONITOR_INTERVAL_SECONDS", "60"))
RETRAIN_ENABLED = os.environ.get("RETRAIN_ON_DRIFT", "true").lower() == "true"
DEPLOYMENT = os.environ.get("RETRAIN_DEPLOYMENT", "training-pipeline/retrain")


def load_reference() -> pd.DataFrame | None:
    path = SHARED_DIR / "reference.csv"
    return pd.read_csv(path) if path.exists() else None


def load_recent(engine, window: int) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT features, prediction, actual FROM predictions ORDER BY created_at DESC LIMIT :n"),
            {"n": window},
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r[0] for r in rows])
    df["prediction"] = [r[1] for r in rows]
    df["actual"] = [r[2] for r in rows]
    return df


def last_retrain_time(engine) -> datetime | None:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT max(created_at) FROM drift_metrics WHERE retrain_triggered")
        ).scalar()


def trigger_retrain() -> bool:
    """Start a run of the Prefect deployment via Prefect's REST API."""
    api = os.environ.get("PREFECT_API_URL", "http://prefect:4200/api")
    try:
        dep = requests.get(f"{api}/deployments/name/{DEPLOYMENT}", timeout=10)
        dep.raise_for_status()
        run = requests.post(
            f"{api}/deployments/{dep.json()['id']}/create_flow_run",
            json={"tags": ["drift-triggered"]},
            timeout=10,
        )
        run.raise_for_status()
        log.warning("Drift detected — triggered retraining flow run %s", run.json()["id"])
        return True
    except Exception as exc:
        log.error("Could not trigger retraining: %s", exc)
        return False


def run_once(engine, cfg) -> None:
    reference = load_reference()
    if reference is None:
        log.info("Reference data not ready yet (training hasn't run).")
        return

    dcfg = cfg["drift"]
    current = load_recent(engine, dcfg["window_size"])
    if len(current) < dcfg["min_window"]:
        log.info("Only %d predictions logged — waiting for at least %d.", len(current), dcfg["min_window"])
        return

    cols = feature_columns(cfg)
    drift = compute_drift(reference, current, cols, dcfg["share_threshold"])
    try:
        quality = compute_classification_quality(
            current.rename(columns={"actual": target_column(cfg)}), target_column(cfg), "prediction"
        )
    except Exception as exc:  # quality is optional; drift must still be recorded
        log.warning("Classification quality failed: %s", exc)
        quality = {}

    retrain = False
    if RETRAIN_ENABLED and should_retrain(
        drift["dataset_drift"], last_retrain_time(engine), datetime.now(UTC), dcfg["retrain_cooldown_minutes"]
    ):
        retrain = trigger_retrain()

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO drift_metrics (window_size, dataset_drift, drift_share, n_drifted_columns, "
                "accuracy, f1, precision_score, recall_score, retrain_triggered) "
                "VALUES (:w, :dd, :ds, :nd, :acc, :f1, :p, :r, :rt)"
            ),
            {
                "w": len(current),
                "dd": drift["dataset_drift"],
                "ds": drift["drift_share"],
                "nd": drift["n_drifted_columns"],
                "acc": quality.get("accuracy"),
                "f1": quality.get("f1"),
                "p": quality.get("precision"),
                "r": quality.get("recall"),
                "rt": retrain,
            },
        )
    log.info(
        "window=%d drift=%s share=%.2f accuracy=%s retrain=%s",
        len(current),
        drift["dataset_drift"],
        drift["drift_share"],
        quality.get("accuracy"),
        retrain,
    )


def main() -> None:
    cfg = load_config()
    engine = create_engine(database_url(), pool_pre_ping=True)
    while True:
        try:
            ensure_schema(engine)
            break
        except Exception as exc:
            log.info("Waiting for Postgres: %s", exc)
            time.sleep(5)

    while True:
        try:
            run_once(engine, cfg)
        except Exception:
            log.exception("Monitoring iteration failed")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
