import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from api import main

GOOD = {
    "hotel": "City Hotel", "lead_time": 120, "arrival_date_year": 2017,
    "arrival_date_month": "August", "arrival_date_day_of_month": 14,
    "stays_in_weekend_nights": 1, "stays_in_week_nights": 3, "adults": 2,
    "children": 0, "babies": 0, "meal": "BB", "country": "PRT",
    "market_segment": "Online TA", "is_repeated_guest": 0,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A", "deposit_type": "No Deposit", "agent": 9,
    "company": None, "customer_type": "Transient", "adr": 110.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def test_valid_booking_returns_probability(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["cancel_probability"] <= 1
    assert body["label"] in ("Likely to cancel", "Likely to keep booking")


def test_bad_input_is_rejected(client):
    assert client.post("/predict", json={**GOOD, "lead_time": -5}).status_code == 422
    assert client.post("/predict", json={**GOOD, "hotel": "Motel"}).status_code == 422


def test_missing_field_is_rejected(client):
    bad = dict(GOOD)
    bad.pop("adr")
    assert client.post("/predict", json=bad).status_code == 422


def test_longer_lead_time_raises_risk(client):
    low = client.post("/predict", json={**GOOD, "lead_time": 5}).json()["cancel_probability"]
    high = client.post("/predict", json={**GOOD, "lead_time": 300}).json()["cancel_probability"]
    assert high > low


def test_metrics_endpoint(client):
    client.get("/health")
    assert "api_requests_total" in client.get("/metrics").text


def test_prediction_works_without_a_database(client):
    """No DATABASE_URL in the tests, so logging is off and the answer still comes back."""
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["prediction_id"] is None


def test_prediction_is_passed_to_the_logger_with_its_source(client, monkeypatch):
    seen = {}

    def fake_log(features, probability, prediction, source):
        seen.update(features=features, probability=probability, prediction=prediction, source=source)
        return 123

    monkeypatch.setattr(main, "log_prediction", fake_log)
    r = client.post("/predict", json=GOOD, headers={"X-Source": "simulator"})
    assert r.json()["prediction_id"] == 123
    assert seen["source"] == "simulator"
    assert seen["features"]["lead_time"] == 120
    assert seen["prediction"] in (0, 1)


def test_database_failure_does_not_break_predictions(client, monkeypatch):
    """Point the API at a database that is not there: the user must still get an answer."""
    broken = create_engine("postgresql+psycopg://nobody:nopass@localhost:1/none")
    monkeypatch.setattr(main, "engine", broken)
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["prediction_id"] is None


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="needs a PostgreSQL database")
def test_prediction_row_is_written_to_postgres(client, monkeypatch):
    url = os.environ["TEST_DATABASE_URL"]
    db = create_engine(url)
    monkeypatch.setattr(main, "engine", db)
    r = client.post("/predict", json=GOOD, headers={"X-Source": "pytest"})
    row_id = r.json()["prediction_id"]
    assert row_id is not None
    with db.connect() as conn:
        row = conn.execute(
            text("SELECT source, prediction, features->>'lead_time' FROM predictions WHERE id = :i"),
            {"i": row_id},
        ).one()
    assert row[0] == "pytest"
    assert row[2] == "120"
