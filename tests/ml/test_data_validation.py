"""Data validation: the raw file has what the model needs, and cleaning removes bad rows."""

REQUIRED = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country", "market_segment",
    "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "deposit_type", "agent", "company",
    "customer_type", "adr",
]


def test_required_columns_present(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    assert not missing, f"missing columns: {missing}"


def test_target_is_binary_and_never_missing(raw):
    assert raw["is_canceled"].notna().all()
    assert set(raw["is_canceled"].unique()) <= {0, 1}


def test_categories_are_known(raw):
    assert set(raw["hotel"].unique()) <= {"City Hotel", "Resort Hotel"}
    assert set(raw["deposit_type"].unique()) <= {"No Deposit", "Non Refund", "Refundable"}


def test_cleaning_removes_bad_rows(splits):
    (X_train, _), (X_valid, _), (X_test, _) = splits
    for X in (X_train, X_valid, X_test):
        guests = X["adults"] + X["children"].fillna(0) + X["babies"]
        assert (guests > 0).all(), "bookings with no guests should be dropped"
        assert (X["adr"] >= 0).all(), "negative prices should be dropped"
        assert not X.duplicated().any(), "exact duplicates should be dropped"


def test_split_is_chronological(splits):
    """Train must end before validation starts, and validation before test."""
    import pandas as pd
    from hotel_model import MONTH_NUM

    def dates(X):
        return pd.to_datetime({"year": X["arrival_date_year"], "month": X["arrival_date_month"].map(MONTH_NUM),
                               "day": X["arrival_date_day_of_month"]})

    (X_train, _), (X_valid, _), (X_test, _) = splits
    assert dates(X_train).max() < dates(X_valid).min()
    assert dates(X_valid).max() < dates(X_test).min()
