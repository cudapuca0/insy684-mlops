"""
Custom components for the hotel cancellation model (Task 2: is_canceled).

Pruned from FinalDeliverable_Hotel_Cancellation_Forecasting.ipynb
(Sections 11 and 15). This module MUST be importable wherever the saved
joblib is loaded (e.g. the FastAPI container), because the pickled model
references these classes by module path.
"""

import calendar

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 67
OUTLIER_UPPER_QUANTILE = 0.995
RARE_CATEGORY_MIN_COUNT = 100
DECISION_THRESHOLD = 0.45  # chosen on validation by max F0.5 (notebook 15.4)

MONTH_NUM = {name: i for i, name in enumerate(calendar.month_name) if name}

# Columns that leak the outcome or are only known after booking (notebook F6).
LEAKY_OR_POST_BOOKING = [
    "reservation_status", "reservation_status_date", "assigned_room_type",
    "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests",
]

# Numeric columns capped at the training 99.5th percentile.
CLIP_CANDIDATES = [
    "adr", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "babies", "previous_cancellations", "previous_bookings_not_canceled",
    "history_count", "total_nights", "total_guests", "adr_per_guest",
]


# ---------------------------------------------------------------------------
# Feature engineering (notebook 11.3 and 15)
# ---------------------------------------------------------------------------
class BookingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, booking-time feature construction; safe within CV folds."""

    LEAKY_OR_POST_BOOKING = LEAKY_OR_POST_BOOKING

    def __init__(self, drop_deposit_type=False):
        self.drop_deposit_type = drop_deposit_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        month_number = d["arrival_date_month"].map(MONTH_NUM)
        arrival = pd.to_datetime({
            "year": d["arrival_date_year"], "month": month_number,
            "day": d["arrival_date_day_of_month"],
        }, errors="raise")

        d["children"] = d["children"].fillna(0)
        d["country"] = d["country"].fillna("UNKNOWN")
        d["meal"] = d["meal"].replace("Undefined", "SC")
        d["market_segment"] = d["market_segment"].replace("Undefined", "UNKNOWN")
        d["agent"] = d["agent"].map(
            lambda value: "NO_AGENT" if pd.isna(value) else str(int(value))
        )
        d["has_agent"] = (d["agent"] != "NO_AGENT").astype(int)
        d["has_company"] = d["company"].notna().astype(int)

        d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
        d["zero_night_stay"] = (d["total_nights"] == 0).astype(int)
        d["weekend_ratio"] = d["stays_in_weekend_nights"] / d["total_nights"].replace(0, np.nan)
        d["total_guests"] = d["adults"] + d["children"] + d["babies"]
        d["is_family"] = ((d["children"] + d["babies"]) > 0).astype(int)
        d["has_babies"] = (d["babies"] > 0).astype(int)
        d["adr_per_guest"] = d["adr"] / d["total_guests"].replace(0, np.nan)

        d["history_count"] = d["previous_cancellations"] + d["previous_bookings_not_canceled"]
        d["has_history"] = (d["history_count"] > 0).astype(int)
        d["historical_cancel_rate"] = (
            d["previous_cancellations"] / d["history_count"].replace(0, np.nan)
        ).fillna(0)

        d["lead_time_bucket"] = pd.cut(
            d["lead_time"], bins=[-np.inf, 7, 30, 90, 180, 365, np.inf],
            labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"],
        ).astype(str)
        d["lead_time_log"] = np.log1p(d["lead_time"].clip(lower=0))
        d["adr_log"] = np.log1p(d["adr"].clip(lower=0))
        d["total_nights_log"] = np.log1p(d["total_nights"].clip(lower=0))

        d["is_domestic"] = (d["country"] == "PRT").astype(int)
        d["hotel_market_segment"] = d["hotel"].astype(str) + " | " + d["market_segment"].astype(str)
        d["long_lead_group"] = ((d["lead_time"] > 180)
                                & (d["market_segment"] == "Groups")).astype(int)

        d["arrival_dayofweek"] = arrival.dt.dayofweek.astype(str)
        d["is_weekend_arrival"] = (arrival.dt.dayofweek >= 5).astype(int)
        d["month_sin"] = np.sin(2 * np.pi * month_number / 12)
        d["month_cos"] = np.cos(2 * np.pi * month_number / 12)

        drop_cols = self.LEAKY_OR_POST_BOOKING + [
            "is_canceled", "company", "distribution_channel",
            "arrival_date_year", "arrival_date_month", "arrival_date_week_number",
            "arrival_date_day_of_month",
        ]
        if self.drop_deposit_type:
            drop_cols.append("deposit_type")
        d = d.drop(columns=drop_cols, errors="ignore")
        numeric = d.select_dtypes(include="number").columns
        d.loc[:, numeric] = d.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
        return d


class FineTuneStayValueEngineer(BookingFeatureEngineer):
    """V2 features: V0 plus estimated_stay_value."""

    def transform(self, X):
        features = super().transform(X)
        features["estimated_stay_value"] = features["adr"] * features["total_nights"]
        return features


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Upper-tail winsorization; learns limits only when fit() is called."""

    def __init__(self, upper_quantile=0.995):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.upper_bounds_ = np.nanquantile(array, self.upper_quantile, axis=0)
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


def to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


# ---------------------------------------------------------------------------
# Preprocessing (notebook 11.4) and feature versions V0 / V2 (notebook 15)
# ---------------------------------------------------------------------------
def make_preparation(X_train_raw, version):
    """Build the unfitted prepare pipeline for feature version 'V0' or 'V2'.

    Column groups are inferred from the engineered *training* data, as in the
    notebook. V2 drops is_domestic and adds estimated_stay_value.
    """
    if version not in {"V0", "V2"}:
        raise ValueError("version must be 'V0' or 'V2'")

    preview = BookingFeatureEngineer().transform(X_train_raw)
    num_cols = preview.select_dtypes(include="number").columns.tolist()
    cat_cols = preview.select_dtypes(exclude="number").columns.tolist()
    clip_cols = [c for c in CLIP_CANDIDATES if c in num_cols]
    other_num_cols = [c for c in num_cols if c not in clip_cols]

    engineer = BookingFeatureEngineer()
    if version == "V2":
        clip_cols = [c for c in clip_cols if c != "is_domestic"] + ["estimated_stay_value"]
        other_num_cols = [c for c in other_num_cols if c != "is_domestic"]
        cat_cols = [c for c in cat_cols if c != "is_domestic"]
        engineer = FineTuneStayValueEngineer()

    preprocessor = ColumnTransformer([
        ("capped_num", Pipeline([
            ("cap", QuantileCapper(upper_quantile=OUTLIER_UPPER_QUANTILE)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), clip_cols),
        ("other_num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), other_num_cols),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=RARE_CATEGORY_MIN_COUNT)),
        ]), cat_cols),
    ], remainder="drop", verbose_feature_names_out=True)

    return Pipeline([("feature_engineering", engineer), ("preprocessing", preprocessor)])


def make_extra_trees_v2(X_train_raw):
    """Extra Trees V2: original settings were retained after tuning (notebook 15.2)."""
    return Pipeline([
        ("prepare", make_preparation(X_train_raw, "V2")),
        ("model", ExtraTreesClassifier(
            n_estimators=90, max_depth=18, min_samples_leaf=4,
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])


def make_mlp_v0(X_train_raw):
    """MLP V0 with the tuned settings selected in notebook 15.2 (candidate 2)."""
    return Pipeline([
        ("prepare", make_preparation(X_train_raw, "V0")),
        ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
        ("model", MLPClassifier(
            hidden_layer_sizes=(48,),
            alpha=0.01142030649919016,
            learning_rate_init=0.001,
            max_iter=90,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Final model: equal-weight soft-voting ensemble with threshold 0.45
# ---------------------------------------------------------------------------
class CancellationEnsemble(BaseEstimator, ClassifierMixin):
    """Averages the cancellation probabilities of fitted member pipelines.

    predict() labels a booking as canceled when the averaged probability is
    strictly greater than `threshold` (same rule as the notebook).
    """

    def __init__(self, members, threshold=DECISION_THRESHOLD):
        self.members = members  # dict: name -> unfitted sklearn Pipeline
        self.threshold = threshold

    def fit(self, X, y):
        self.fitted_members_ = {name: est.fit(X, y) for name, est in self.members.items()}
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p1 = np.mean(
            [est.predict_proba(X)[:, 1] for est in self.fitted_members_.values()],
            axis=0,
        )
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > self.threshold).astype(int)
