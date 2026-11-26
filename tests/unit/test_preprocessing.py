"""Preprocessing: a known input produces the expected features."""

import numpy as np
import pandas as pd
import pytest

from mlops_core.features import build_preprocessor


@pytest.fixture
def small_cfg(cfg):
    small = {**cfg, "features": {**cfg["features"], "numeric": ["a", "b"], "categorical": ["c"]}}
    return small


def test_numeric_features_are_standardised(small_cfg):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "x"]})
    out = build_preprocessor(small_cfg).fit_transform(df)
    # middle value of [1, 2, 3] standardises to exactly 0
    assert out[1, 0] == pytest.approx(0.0)
    assert out[:, 0].mean() == pytest.approx(0.0)
    assert out[:, 0].std() == pytest.approx(1.0)


def test_categorical_features_are_one_hot_encoded(small_cfg):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "x"]})
    out = build_preprocessor(small_cfg).fit_transform(df)
    # 2 numeric + 2 one-hot columns; "x" -> [1, 0], "y" -> [0, 1]
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[:, 2:], [[1, 0], [0, 1], [1, 0]])


def test_missing_values_are_imputed_and_unknown_categories_ignored(small_cfg):
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "x"]})
    pre = build_preprocessor(small_cfg).fit(train)
    new = pd.DataFrame({"a": [np.nan], "b": [20.0], "c": ["never-seen"]})
    out = pre.transform(new)
    assert not np.isnan(out).any()
    assert out[0, 0] == pytest.approx(0.0)  # NaN -> median (2.0) -> standardised 0
    np.testing.assert_array_equal(out[0, 2:], [0, 0])


def test_project_preprocessor_output_width(cfg, dataset):
    out = build_preprocessor(cfg).fit_transform(dataset)
    assert out.shape == (len(dataset), len(cfg["features"]["numeric"]))
