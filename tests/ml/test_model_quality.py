"""Model vs. baseline, and sensible predictions."""

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def test_model_beats_naive_baseline(trained_model, splits):
    """Better ranking than chance and better probabilities than always predicting the base rate."""
    (_, y_train), (_, y_valid), (X_test, y_test) = splits
    p = trained_model.predict_proba(X_test)[:, 1]
    base_rate = np.concatenate([y_train, y_valid]).mean()
    assert roc_auc_score(y_test, p) > 0.70
    assert brier_score_loss(y_test, p) < brier_score_loss(y_test, np.full(len(p), base_rate))


def test_probabilities_are_valid(trained_model, splits):
    _, _, (X_test, _) = splits
    proba = trained_model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(np.unique(trained_model.predict(X_test))) <= {0, 1}


def test_longer_lead_time_raises_cancellation_risk(trained_model, splits):
    """Same bookings, only the lead time changed: booking far ahead should look riskier on average."""
    _, _, (X_test, _) = splits
    short = trained_model.predict_proba(X_test.assign(lead_time=3))[:, 1].mean()
    long_ = trained_model.predict_proba(X_test.assign(lead_time=300))[:, 1].mean()
    assert long_ > short


def test_past_cancellations_raise_cancellation_risk(trained_model, splits):
    _, _, (X_test, _) = splits
    clean = trained_model.predict_proba(X_test.assign(previous_cancellations=0))[:, 1].mean()
    history = trained_model.predict_proba(X_test.assign(previous_cancellations=2))[:, 1].mean()
    assert history > clean
