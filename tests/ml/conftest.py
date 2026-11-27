"""Shared fixtures for the ML tests.

They run on a small committed sample of the dataset (tests/data/hotel_bookings_sample.csv:
150 bookings per arrival month, 3,900 rows) so CI needs no download and stays fast.
The model code is imported from frontend/pipeline for now; update the path below if
hotel_model.py moves to a shared package.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "data" / "hotel_bookings_sample.csv"
sys.path.insert(0, str(ROOT / "frontend" / "pipeline"))

from hotel_model import (  # noqa: E402
    CancellationEnsemble,
    make_extra_trees_v2,
    make_mlp_v0,
)
from train import load_and_split  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return pd.read_csv(SAMPLE)


@pytest.fixture(scope="session")
def splits():
    """Same cleaning and chronological 70/15/15 split as train.py."""
    return load_and_split(SAMPLE)


@pytest.fixture(scope="session")
def trained_model(splits):
    """The final model recipe (Extra Trees V2 + MLP V0, threshold 0.45), fit on train + validation."""
    (X_train, y_train), (X_valid, y_valid), _ = splits
    X_dev = pd.concat([X_train, X_valid], ignore_index=True)
    y_dev = pd.concat([y_train, y_valid], ignore_index=True)
    model = CancellationEnsemble(members={
        "Extra Trees V2": make_extra_trees_v2(X_train),
        "MLP V0": make_mlp_v0(X_train),
    })
    return model.fit(X_dev, y_dev)
