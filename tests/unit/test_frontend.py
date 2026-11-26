"""Frontend: shows a readable prediction, and friendly errors."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from frontend import app as frontend


@pytest.fixture
def client():
    frontend.app.config["TESTING"] = True
    return frontend.app.test_client()


def _form(cfg):
    return {k: str(v) for k, v in cfg["features"]["defaults"].items()}


def test_form_renders(client, cfg):
    html = client.get("/").get_data(as_text=True)
    for col in cfg["features"]["numeric"]:
        assert f'name="{col}"' in html


def test_prediction_is_shown(client, cfg):
    fake = MagicMock(ok=True)
    fake.json.return_value = {
        "prediction_id": 1,
        "prediction": 1,
        "label": "Malignant",
        "probability": 0.87,
        "model_version": "3",
    }
    with patch.object(frontend.requests, "post", return_value=fake) as post:
        html = client.post("/", data=_form(cfg)).get_data(as_text=True)
    assert post.call_args.args[0].endswith("/predict")
    assert "Malignant" in html and "87.0%" in html


def test_non_numeric_input_shows_error_without_calling_api(client, cfg):
    form = _form(cfg)
    form[cfg["features"]["numeric"][0]] = "abc"
    with patch.object(frontend.requests, "post") as post:
        html = client.post("/", data=form).get_data(as_text=True)
    post.assert_not_called()
    assert "must be a number" in html


def test_api_down_shows_friendly_message(client, cfg):
    with patch.object(frontend.requests, "post", side_effect=requests.ConnectionError()):
        html = client.post("/", data=_form(cfg)).get_data(as_text=True)
    assert "Could not reach the prediction service" in html
