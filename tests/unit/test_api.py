"""Model API behaviour, using FastAPI's in-process test client and a model
trained in the test session (no MLflow server or database needed)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mlflow")


@pytest.fixture
def client(trained_model, monkeypatch):
    monkeypatch.setenv("PREDICTION_LOGGING", "false")
    monkeypatch.setenv("MODEL_PATH", "/nonexistent")  # don't poll a registry
    from fastapi.testclient import TestClient

    from api import main

    with TestClient(main.app) as c:
        main.holder.model, main.holder.version = trained_model, "test"
        yield c


def test_valid_request_returns_readable_prediction(client, cfg):
    resp = client.post("/predict", json=cfg["features"]["defaults"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in (cfg["classes"]["positive_label"], cfg["classes"]["negative_label"])
    assert 0 <= body["probability"] <= 1


def test_out_of_range_input_is_rejected(client, cfg):
    payload = dict(cfg["features"]["defaults"])
    payload[next(iter(payload))] = -999
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_field_is_rejected(client, cfg):
    payload = dict(cfg["features"]["defaults"])
    payload.pop(next(iter(payload)))
    assert client.post("/predict", json=payload).status_code == 422


def test_metrics_endpoint_exposes_request_counters(client, cfg):
    client.post("/predict", json=cfg["features"]["defaults"])
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_request_latency_seconds" in text
