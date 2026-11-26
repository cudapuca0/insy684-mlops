import pytest
from fastapi.testclient import TestClient

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
