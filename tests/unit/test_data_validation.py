"""Data validation: required columns present, no missing target, binary target."""

import numpy as np
import pytest

from mlops_core.config import feature_columns, target_column
from mlops_core.data import DataValidationError, load_raw_data, validate_data


def test_raw_data_passes_validation(cfg):
    df = validate_data(load_raw_data(cfg), cfg)
    assert list(df.columns) == feature_columns(cfg) + [target_column(cfg)]
    assert len(df) > 100


def test_missing_required_column_is_rejected(cfg):
    df = load_raw_data(cfg).drop(columns=[feature_columns(cfg)[0]])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_data(df, cfg)


def test_missing_target_values_are_rejected(cfg):
    df = load_raw_data(cfg)
    df.loc[df.index[:3], target_column(cfg)] = np.nan
    with pytest.raises(DataValidationError, match="missing values"):
        validate_data(df, cfg)


def test_non_binary_target_is_rejected(cfg):
    df = load_raw_data(cfg)
    df.loc[df.index[0], target_column(cfg)] = 2
    with pytest.raises(DataValidationError, match="binary"):
        validate_data(df, cfg)
