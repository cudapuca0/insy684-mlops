"""Preprocessing: a known booking produces the expected engineered features, and nothing leaky survives."""

import numpy as np
import pandas as pd
import pytest
from hotel_model import LEAKY_OR_POST_BOOKING, BookingFeatureEngineer

KNOWN = {
    "hotel": "City Hotel", "is_canceled": 0, "lead_time": 45, "arrival_date_year": 2017,
    "arrival_date_month": "March", "arrival_date_week_number": 11, "arrival_date_day_of_month": 18,
    "stays_in_weekend_nights": 2, "stays_in_week_nights": 3, "adults": 2, "children": 1.0, "babies": 0,
    "meal": "Undefined", "country": "PRT", "market_segment": "Online TA", "distribution_channel": "TA/TO",
    "is_repeated_guest": 0, "previous_cancellations": 1, "previous_bookings_not_canceled": 3,
    "reserved_room_type": "A", "assigned_room_type": "A", "booking_changes": 0, "deposit_type": "No Deposit",
    "agent": 9.0, "company": np.nan, "days_in_waiting_list": 0, "customer_type": "Transient", "adr": 120.0,
    "required_car_parking_spaces": 0, "total_of_special_requests": 1,
    "reservation_status": "Check-Out", "reservation_status_date": "2017-03-23",
}


@pytest.fixture
def features():
    return BookingFeatureEngineer().transform(pd.DataFrame([KNOWN])).iloc[0]


def test_stay_and_guest_features(features):
    assert features["total_nights"] == 5
    assert features["weekend_ratio"] == pytest.approx(2 / 5)
    assert features["total_guests"] == 3
    assert features["is_family"] == 1
    assert features["adr_per_guest"] == pytest.approx(40.0)


def test_history_and_agent_features(features):
    assert features["history_count"] == 4
    assert features["historical_cancel_rate"] == pytest.approx(0.25)
    assert features["has_agent"] == 1 and features["agent"] == "9"
    assert features["has_company"] == 0
    assert features["is_domestic"] == 1


def test_date_features(features):
    assert features["lead_time_bucket"] == "31-90"
    assert features["arrival_dayofweek"] == "5"  # 18 March 2017 was a Saturday
    assert features["is_weekend_arrival"] == 1
    assert features["month_sin"] == pytest.approx(np.sin(2 * np.pi * 3 / 12))


def test_cleans_undefined_meal(features):
    assert features["meal"] == "SC"


def test_leaky_and_target_columns_are_dropped(features):
    for col in LEAKY_OR_POST_BOOKING + ["is_canceled"]:
        assert col not in features.index, f"{col} must not reach the model"
