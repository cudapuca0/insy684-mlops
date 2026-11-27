"""MLflow tracking + registry for the training pipeline (training/train_mlflow.py).

The end-to-end test uses a throwaway local MLflow store (SQLite in a temp folder) and the
committed 3,900-row sample, so it needs no server and no download.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "data" / "hotel_bookings_sample.csv"


def load_module():
    spec = importlib.util.spec_from_file_location("train_mlflow", ROOT / "training" / "train_mlflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_reports_the_expected_metrics():
    tm = load_module()
    y = np.array([0, 0, 1, 1])
    scores = tm.evaluate(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert set(scores) == {"brier", "precision", "recall", "f1", "predicted_cancel_rate", "roc_auc"}
    assert scores["roc_auc"] == pytest.approx(0.75)
    assert scores["predicted_cancel_rate"] == pytest.approx(0.5)  # 0.6 and 0.9 are > 0.45


def test_baseline_is_constant_training_rate():
    tm = load_module()
    proba = tm.baseline_proba(np.array([0, 1, 1, 0]), 3)
    assert proba.tolist() == [0.5, 0.5, 0.5]


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield mlflow
    mlflow.set_tracking_uri(None)


def test_training_tracks_candidates_and_registers_production_model(local_mlflow):
    from mlflow.tracking import MlflowClient

    tm = load_module()
    parent_id = tm.train_and_register(SAMPLE)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(tm.EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    names = {r.data.tags.get("mlflow.runName") for r in runs}
    assert {"training-pipeline", "baseline", "extra_trees_v2", "mlp_v0", "ensemble",
            "final-ensemble"} <= names
    for run in runs:
        if run.data.tags.get("mlflow.runName") in {"extra_trees_v2", "mlp_v0", "ensemble"}:
            assert "val_roc_auc" in run.data.metrics
            assert run.data.tags.get("mlflow.parentRunId") == parent_id

    version = client.get_model_version_by_alias(tm.MODEL_NAME, tm.MODEL_ALIAS)
    model = local_mlflow.sklearn.load_model(f"models:/{tm.MODEL_NAME}@{tm.MODEL_ALIAS}")
    _, _, (X_test, _) = tm.load_and_split(SAMPLE)
    proba = model.predict_proba(X_test.head(20))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert version.version == "1"


def test_register_joblib_only_when_registry_is_empty(local_mlflow):
    from mlflow.tracking import MlflowClient

    tm = load_module()
    joblib_path = ROOT / "model" / "cancellation_model.joblib"
    assert tm.register_joblib(joblib_path, if_empty=True) == "1"
    assert tm.register_joblib(joblib_path, if_empty=True) is None  # already has @production
    current = MlflowClient().get_model_version_by_alias(tm.MODEL_NAME, tm.MODEL_ALIAS)
    assert current.version == "1"
