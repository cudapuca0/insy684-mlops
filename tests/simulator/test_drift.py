"""Drift injection: drifted traffic is flagged, clean traffic is not, and payloads match the API."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulator"))

from drift import categorical_drift, inject_drift, numeric_drift  # noqa: E402
from simulate import FIELDS, load_unseen, to_payload  # noqa: E402

SAMPLE = ROOT / "tests" / "data" / "hotel_bookings_sample.csv"


@pytest.fixture(scope="module")
def unseen():
    return load_unseen(SAMPLE, since="2017-05-05")


@pytest.fixture(scope="module")
def halves(unseen):
    shuffled = unseen.sample(frac=1, random_state=1)
    mid = len(shuffled) // 2
    return shuffled.iloc[:mid], shuffled.iloc[mid:]


def test_injected_drift_is_flagged(halves):
    reference, current = halves
    drifted = inject_drift(current, seed=1)
    assert numeric_drift(reference["lead_time"], drifted["lead_time"])
    assert numeric_drift(reference["adr"], drifted["adr"])
    assert categorical_drift(reference["market_segment"], drifted["market_segment"])
    assert categorical_drift(reference["country"], drifted["country"])


def test_clean_traffic_is_not_flagged(halves):
    reference, current = halves
    assert not numeric_drift(reference["lead_time"], current["lead_time"])
    assert not numeric_drift(reference["adr"], current["adr"])
    assert not categorical_drift(reference["market_segment"], current["market_segment"])


def test_zero_strength_changes_nothing(unseen):
    assert inject_drift(unseen, seed=1, strength=0).equals(unseen)


def test_original_data_is_untouched(unseen):
    before = unseen.copy()
    inject_drift(unseen, seed=1)
    pd.testing.assert_frame_equal(unseen, before)


def test_drifted_values_stay_valid_for_the_api(unseen):
    drifted = inject_drift(unseen, seed=1)
    assert drifted["lead_time"].between(0, 800).all()
    assert drifted["adr"].between(0, 1000).all()


def test_unseen_data_is_only_the_test_period(unseen):
    assert (unseen["arrival_date_year"] == 2017).all()
    assert set(unseen["arrival_date_month"]) <= {"May", "June", "July", "August"}


def test_payload_matches_api_schema(unseen):
    payload = to_payload(inject_drift(unseen, seed=1).iloc[0])
    assert set(payload) == set(FIELDS)
    assert isinstance(payload["lead_time"], int)
    assert payload["agent"] is None or isinstance(payload["agent"], float)
    assert payload["hotel"] in ("City Hotel", "Resort Hotel")
