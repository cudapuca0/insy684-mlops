"""
Train the final hotel cancellation model and dump it with joblib.

Reproduces the notebook's final Task 2 model (Section 17):
  Extra Trees V2 + MLP V0, equal-weight probability average, threshold 0.45,
  refit on train + validation, evaluated once on the chronological test set.

Usage:
  python train.py --data data/raw/hotel_bookings.csv --out models/cancellation_model.joblib
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)

from hotel_model import (MONTH_NUM, CancellationEnsemble, make_extra_trees_v2,
                         make_mlp_v0)


def load_and_split(csv_path):
    """Cleaning (notebook 11.1) and 70/15/15 chronological split by unique arrival date (11.2)."""
    raw = pd.read_csv(csv_path)

    clean = raw.drop_duplicates().copy()
    guest_count = clean["adults"] + clean["children"].fillna(0) + clean["babies"]
    clean = clean.loc[(guest_count > 0) & ~(clean["adr"] < 0)].reset_index(drop=True)

    clean["_arrival_date"] = pd.to_datetime({
        "year": clean["arrival_date_year"],
        "month": clean["arrival_date_month"].map(MONTH_NUM),
        "day": clean["arrival_date_day_of_month"],
    })
    clean = clean.sort_values("_arrival_date", kind="stable").reset_index(drop=True)

    dates = clean["_arrival_date"].drop_duplicates().to_numpy()
    train_end = pd.Timestamp(dates[int(len(dates) * 0.70) - 1])
    valid_end = pd.Timestamp(dates[int(len(dates) * 0.85) - 1])

    def make_xy(part):
        X = part.drop(columns=["is_canceled", "_arrival_date"]).reset_index(drop=True)
        y = part["is_canceled"].astype(int).reset_index(drop=True)
        return X, y

    train = clean[clean["_arrival_date"] <= train_end]
    valid = clean[(clean["_arrival_date"] > train_end) & (clean["_arrival_date"] <= valid_end)]
    test = clean[clean["_arrival_date"] > valid_end]
    print(f"Rows after cleaning: {len(clean):,} | train {len(train):,} | "
          f"valid {len(valid):,} | test {len(test):,}")
    return make_xy(train), make_xy(valid), make_xy(test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/hotel_bookings.csv")
    parser.add_argument("--out", default="models/cancellation_model.joblib")
    args = parser.parse_args()

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_split(args.data)

    # Final refit on train + validation (notebook Section 17).
    X_dev = pd.concat([X_train, X_valid], ignore_index=True)
    y_dev = pd.concat([y_train, y_valid], ignore_index=True)

    # Column groups are inferred from the original training partition, as in the notebook.
    model = CancellationEnsemble(members={
        "Extra Trees V2": make_extra_trees_v2(X_train),
        "MLP V0": make_mlp_v0(X_train),
    })
    print("Fitting ensemble on train + validation...")
    model.fit(X_dev, y_dev)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }
    print("Test metrics (threshold %.2f):" % model.threshold)
    for k, v in metrics.items():
        print(f"  {k:9s} {v:.4f}")
    print("Notebook reference: roc_auc 0.8040, brier 0.1703, precision 0.6090, "
          "recall 0.5196, f1 0.5607")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out, compress=3)
    print(f"Saved model to {out.resolve()}")

    # Sanity check: reload and predict.
    reloaded = joblib.load(out)
    assert np.allclose(reloaded.predict_proba(X_test.head(5)), model.predict_proba(X_test.head(5)))
    print("Reload check passed.")


if __name__ == "__main__":
    main()
