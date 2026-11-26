"""Postgres schema for logged predictions and Evidently metrics."""

from __future__ import annotations

import os

PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    features        JSONB       NOT NULL,
    prediction      INTEGER     NOT NULL,
    probability     DOUBLE PRECISION,
    model_version   TEXT,
    actual          INTEGER,            -- ground truth, filled in via /feedback
    source          TEXT                -- 'frontend', 'simulator', 'simulator-drift', ...
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
"""

METRICS_DDL = """
CREATE TABLE IF NOT EXISTS drift_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_size         INTEGER,
    dataset_drift       BOOLEAN,
    drift_share         DOUBLE PRECISION,
    n_drifted_columns   INTEGER,
    accuracy            DOUBLE PRECISION,
    f1                  DOUBLE PRECISION,
    precision_score     DOUBLE PRECISION,
    recall_score        DOUBLE PRECISION,
    retrain_triggered   BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_created_at ON drift_metrics (created_at);
"""


def database_url() -> str:
    """Monitoring database URL built from environment variables (.env)."""
    user = os.environ.get("POSTGRES_USER", "mlops")
    password = os.environ.get("POSTGRES_PASSWORD", "mlops")
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("MONITORING_DB", "monitoring")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


def ensure_schema(engine) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for ddl in (PREDICTIONS_DDL, METRICS_DDL):
            for stmt in filter(None, (s.strip() for s in ddl.split(";"))):
                conn.execute(text(stmt))
