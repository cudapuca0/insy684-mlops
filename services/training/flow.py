"""Prefect training pipeline.

load -> validate -> split -> (optional) add labelled production data ->
train every candidate as a tracked MLflow run -> pick the best ->
register it -> promote it to the `production` alias if it beats the current one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from sklearn.model_selection import train_test_split

import mlflow
from mlops_core.config import feature_columns, load_config, target_column
from mlops_core.data import load_raw_data, split_data, validate_data
from mlops_core.modeling import baseline_metrics, build_pipeline, evaluate, train

SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "training-pipeline")


@task
def load_data() -> pd.DataFrame:
    return load_raw_data()


@task
def validate(df: pd.DataFrame) -> pd.DataFrame:
    return validate_data(df)


@task
def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_data(df)


@task
def load_feedback_data() -> pd.DataFrame:
    """Labelled production traffic (predictions that received /feedback).
    Lets a drift-triggered retrain learn from the new population."""
    logger = get_run_logger()
    try:
        from sqlalchemy import create_engine, text

        from mlops_core.db import database_url

        engine = create_engine(database_url())
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT features, actual FROM predictions WHERE actual IS NOT NULL")
            ).fetchall()
    except Exception as exc:  # DB not available (e.g. first run, local dev)
        logger.warning("Skipping feedback data: %s", exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r[0] for r in rows])
    df[target_column()] = [int(r[1]) for r in rows]
    logger.info("Loaded %d labelled production rows", len(df))
    return df


@task
def save_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Reference data (for drift) and unseen test data (for the simulator)."""
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SHARED_DIR / "reference.csv", index=False)
    test_df.to_csv(SHARED_DIR / "test.csv", index=False)


@task
def train_candidate(
    candidate: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline: dict[str, float],
    trigger: str,
) -> dict[str, Any]:
    cfg = load_config()
    run_name = f"{candidate['name']}-" + "-".join(f"{k}={v}" for k, v in candidate["params"].items())
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tags({"model_type": candidate["type"], "trigger": trigger})
        mlflow.log_params({"model_type": candidate["type"], **candidate["params"]})
        mlflow.log_param("n_train_rows", len(train_df))

        pipeline = train(build_pipeline(candidate["type"], candidate["params"]), train_df)
        val_metrics = evaluate(pipeline, val_df)
        test_metrics = evaluate(pipeline, test_df)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in baseline.items()})

        X_example = train_df[feature_columns(cfg)].head(5)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=infer_signature(X_example, pipeline.predict_proba(X_example)),
            input_example=X_example,
        )
        return {
            "run_id": run.info.run_id,
            "name": run_name,
            "val": val_metrics,
            "test": test_metrics,
        }


@task
def register_and_promote(best: dict[str, Any], test_df: pd.DataFrame, force: bool) -> str:
    logger = get_run_logger()
    cfg = load_config()
    name, alias = cfg["model"]["registered_name"], cfg["model"]["alias"]
    metric = cfg["model"]["selection_metric"]
    client = MlflowClient()

    version = mlflow.register_model(f"runs:/{best['run_id']}/model", name).version
    client.set_model_version_tag(name, version, "selected_by", f"val_{metric}")

    current_score = None
    try:
        current = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
        current_score = evaluate(current, test_df)[metric]
    except Exception:
        logger.info("No current '%s' model yet — promoting the first one.", alias)

    new_score = best["test"][metric]
    if force or current_score is None or new_score >= current_score:
        client.set_registered_model_alias(name, alias, version)
        logger.info(
            "Promoted %s v%s to @%s (test %s=%.4f, previous=%s)",
            name,
            version,
            alias,
            metric,
            new_score,
            current_score,
        )
    else:
        logger.info(
            "Kept existing @%s (test %s=%.4f) — new v%s scored %.4f",
            alias,
            metric,
            current_score,
            version,
            new_score,
        )
    return str(version)


@flow(name="training-pipeline", log_prints=True)
def training_pipeline(
    use_feedback: bool = False, force_promote: bool = False, trigger: str = "manual"
) -> str:
    cfg = load_config()
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = validate(load_data())
    train_df, test_df = split(df)
    save_datasets(train_df, test_df)

    if use_feedback:
        feedback = load_feedback_data()
        if not feedback.empty:
            fb_train, fb_test = train_test_split(feedback, test_size=0.3, random_state=0)
            train_df = pd.concat([train_df, validate_data(fb_train)], ignore_index=True)
            test_df = pd.concat([test_df, validate_data(fb_test)], ignore_index=True)

    fit_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=cfg["data"]["random_state"],
        stratify=train_df[target_column(cfg)],
    )
    baseline = baseline_metrics(fit_df, test_df)

    with mlflow.start_run(run_name=f"pipeline-{trigger}"):
        mlflow.set_tag("trigger", trigger)
        results = [
            train_candidate(c, fit_df, val_df, test_df, baseline, trigger) for c in cfg["model"]["candidates"]
        ]
        metric = cfg["model"]["selection_metric"]
        best = max(results, key=lambda r: r["val"][metric])
        mlflow.log_param("best_candidate", best["name"])
        mlflow.log_metric(f"best_test_{metric}", best["test"][metric])
        print(f"Best candidate: {best['name']} (val {metric}={best['val'][metric]:.4f})")

    return register_and_promote(best, test_df, force_promote)


if __name__ == "__main__":
    training_pipeline()
