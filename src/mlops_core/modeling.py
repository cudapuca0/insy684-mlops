"""Model construction, training and evaluation (no MLflow here, so it is easy
to unit test; the Prefect flow adds tracking around these functions)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlops_core.config import feature_columns, load_config, target_column
from mlops_core.features import build_preprocessor

_ESTIMATORS = {
    "logistic_regression": lambda p, seed: LogisticRegression(max_iter=2000, random_state=seed, **p),
    "random_forest": lambda p, seed: RandomForestClassifier(random_state=seed, **p),
    "gradient_boosting": lambda p, seed: GradientBoostingClassifier(random_state=seed, **p),
}


def build_pipeline(model_type: str, params: dict[str, Any], cfg: dict[str, Any] | None = None) -> Pipeline:
    cfg = cfg or load_config()
    if model_type not in _ESTIMATORS:
        raise ValueError(f"Unknown model type '{model_type}'. Options: {sorted(_ESTIMATORS)}")
    seed = cfg["data"]["random_state"]
    return Pipeline(
        [
            ("preprocess", build_preprocessor(cfg)),
            ("model", _ESTIMATORS[model_type](params or {}, seed)),
        ]
    )


def train(pipeline: Pipeline, train_df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> Pipeline:
    cfg = cfg or load_config()
    pipeline.fit(train_df[feature_columns(cfg)], train_df[target_column(cfg)])
    return pipeline


def evaluate(model: Any, df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = cfg or load_config()
    X, y = df[feature_columns(cfg)], df[target_column(cfg)]
    pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        if y.nunique() > 1:
            metrics["roc_auc"] = roc_auc_score(y, proba)
    return {k: float(v) for k, v in metrics.items()}


def baseline_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> dict[str, float]:
    """Naive baseline: always predict the majority class from training data."""
    cfg = cfg or load_config()
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[feature_columns(cfg)], train_df[target_column(cfg)])
    return evaluate(dummy, test_df, cfg)
