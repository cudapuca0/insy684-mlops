"""Data loading, validation and splitting."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_core.config import feature_columns, load_config, project_root, target_column


class DataValidationError(ValueError):
    """Raised when the dataset does not meet the expected schema."""


def _load_sklearn_breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # sklearn encodes 0 = malignant, 1 = benign; flip so 1 = malignant (positive class)
    df["malignant"] = 1 - df.pop("target")
    return df


def load_raw_data(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    source = cfg["data"]["source"]
    if source == "sklearn_breast_cancer":
        return _load_sklearn_breast_cancer()
    if source == "csv":
        return pd.read_csv(project_root() / cfg["data"]["path"])
    raise ValueError(f"Unknown data source: {source}")


def validate_data(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Check required columns exist, the target has no missing values and the
    target is binary. Returns only the columns the model needs."""
    cfg = cfg or load_config()
    required = feature_columns(cfg) + [target_column(cfg)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    target = target_column(cfg)
    if df[target].isna().any():
        raise DataValidationError(f"Target column '{target}' has missing values")

    labels = set(df[target].unique())
    if not labels <= {0, 1}:
        raise DataValidationError(f"Target must be binary 0/1, found {sorted(labels)}")

    if len(df) < 50:
        raise DataValidationError(f"Too few rows to train on: {len(df)}")

    return df[required].copy()


def split_data(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    train, test = train_test_split(
        df,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=df[target_column(cfg)],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
