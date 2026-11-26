"""Integration test: talks to a real, running API container over HTTP.

CI starts the container (see .github/workflows/ci.yml). Locally:
    docker compose up -d api   (or the full stack)
    API_URL=http://localhost:8000 pytest tests/integration
"""

import os

import pytest
import requests

API_URL = os.environ.get("API_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not API_URL, reason="set API_URL to run integration tests"),
]


def test_health_reports_model_loaded():
    body = requests.get(f"{API_URL}/health", timeout=10).json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_round_trip(cfg):
    resp = requests.post(f"{API_URL}/predict", json=cfg["features"]["defaults"], timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_bad_input_returns_422(cfg):
    payload = dict(cfg["features"]["defaults"])
    payload[next(iter(payload))] = "not-a-number"
    assert requests.post(f"{API_URL}/predict", json=payload, timeout=10).status_code == 422


def test_metrics_are_exposed_for_prometheus():
    text = requests.get(f"{API_URL}/metrics", timeout=10).text
    assert "api_requests_total" in text
