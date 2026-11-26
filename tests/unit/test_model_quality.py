"""Model vs. baseline, and sensible predictions."""

import pandas as pd

from mlops_core.config import feature_columns
from mlops_core.modeling import baseline_metrics, evaluate


def test_model_beats_naive_baseline(trained_model, splits, cfg):
    train_df, test_df = splits
    model = evaluate(trained_model, test_df, cfg)
    baseline = baseline_metrics(train_df, test_df, cfg)
    assert model["accuracy"] > baseline["accuracy"] + 0.1
    assert model["f1"] > baseline["f1"]
    assert model["f1"] > 0.8


def test_predictions_are_in_valid_range(trained_model, splits, cfg):
    _, test_df = splits
    X = test_df[feature_columns(cfg)]
    proba = trained_model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(trained_model.predict(X)) <= {0, 1}


def test_increasing_risk_feature_increases_probability(trained_model, cfg):
    feature = cfg["features"]["monotonic_increasing"]
    base = pd.DataFrame([cfg["features"]["defaults"]])[feature_columns(cfg)]
    low, high = base.copy(), base.copy()
    lo, hi = cfg["features"]["bounds"][feature]
    low[feature] = lo + (hi - lo) * 0.02
    high[feature] = lo + (hi - lo) * 0.3
    p_low = trained_model.predict_proba(low)[0, 1]
    p_high = trained_model.predict_proba(high)[0, 1]
    assert p_high > p_low
