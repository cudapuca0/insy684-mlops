-- PostgreSQL setup for the monitoring stack.
-- Runs once, the first time the postgres container starts with an empty volume
-- (mounted at /docker-entrypoint-initdb.d/init.sql in docker-compose).
--
--   mlflow      MLflow backend store (experiments, runs, model registry)
--   monitoring  logged predictions + Evidently metrics, read by Grafana

CREATE DATABASE mlflow;
CREATE DATABASE monitoring;

\connect monitoring

-- One row per prediction served by the model API.
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    features        JSONB            NOT NULL,   -- the booking sent to /predict
    probability     DOUBLE PRECISION NOT NULL,   -- predicted chance of cancelling
    prediction      SMALLINT         NOT NULL,   -- 1 = likely to cancel (probability > threshold)
    threshold       DOUBLE PRECISION,
    model_version   TEXT,                        -- MLflow registry version that served it
    actual          SMALLINT,                    -- true outcome, if/when it is known
    source          TEXT                         -- 'frontend', 'simulator', 'simulator-drift', ...
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions (source);

-- One row per monitoring check (Evidently run over the latest window of predictions).
CREATE TABLE IF NOT EXISTS drift_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_size         INTEGER,             -- predictions compared to the reference data
    dataset_drift       BOOLEAN,             -- Evidently's overall verdict
    drift_share         DOUBLE PRECISION,    -- share of feature columns that drifted
    n_drifted_columns   INTEGER,
    mean_probability    DOUBLE PRECISION,    -- average predicted cancel probability in the window
    predicted_cancel_rate DOUBLE PRECISION,  -- share of the window flagged as likely to cancel
    accuracy            DOUBLE PRECISION,    -- only when true outcomes are available
    precision_score     DOUBLE PRECISION,
    recall_score        DOUBLE PRECISION,
    f1                  DOUBLE PRECISION,
    roc_auc             DOUBLE PRECISION,
    retrain_triggered   BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_created_at ON drift_metrics (created_at);

-- Per-feature drift detail for the latest checks (one row per column per check).
CREATE TABLE IF NOT EXISTS feature_drift (
    id              BIGSERIAL PRIMARY KEY,
    check_id        BIGINT REFERENCES drift_metrics (id) ON DELETE CASCADE,
    feature         TEXT NOT NULL,
    drift_score     DOUBLE PRECISION,        -- test statistic / distance from Evidently
    drift_detected  BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_feature_drift_check ON feature_drift (check_id);
