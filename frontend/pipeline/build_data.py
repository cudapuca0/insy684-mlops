"""
Build the dashboard's data: a look at the hotel bookings + the cancellation model's results.

Usage (from the project root):
    python pipeline/build_data.py

Reads  data/raw/hotel_bookings.csv   (Hotel Booking Demand dataset)
Writes data/processed/*.json         (aggregates only, copied into the website)

The model is the final Task 2 ensemble from train.py / hotel_model.py (copied from the MLOps project):
Extra Trees V2 + MLP V0, averaged, threshold 0.45, chronological 70/15/15 split, refit on train + validation,
scored once on the test period. Its numbers must match the notebook (printed at the end).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from hotel_model import LEAKY_OR_POST_BOOKING, MONTH_NUM, CancellationEnsemble, make_extra_trees_v2, make_mlp_v0
from train import load_and_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUT = ROOT / "data" / "processed"
SEED = 67


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def rate_table(d, col, order=None, min_n=0):
    g = d.groupby(col, observed=True)["is_canceled"].agg(["size", "mean"]).reset_index()
    g = g[g["size"] >= min_n]
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    else:
        g = g.sort_values("size", ascending=False)
    return [{"key": str(k), "n": int(n), "cancel_rate": r(m)} for k, n, m in g.itertuples(index=False)]


def overview():
    raw = pd.read_csv(RAW)
    dedup = raw.drop_duplicates()
    guests = dedup["adults"] + dedup["children"].fillna(0) + dedup["babies"]
    clean = dedup.loc[(guests > 0) & ~(dedup["adr"] < 0)].copy()
    clean["arrival"] = pd.to_datetime({"year": clean["arrival_date_year"],
                                       "month": clean["arrival_date_month"].map(MONTH_NUM),
                                       "day": clean["arrival_date_day_of_month"]})
    clean["month"] = clean["arrival"].dt.strftime("%Y-%m")
    clean["nights"] = clean["stays_in_weekend_nights"] + clean["stays_in_week_nights"]
    clean["lead_bucket"] = pd.cut(clean["lead_time"], [-1, 7, 30, 90, 180, 365, 10_000],
                                  labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"]).astype(str)
    stayed = clean[clean["is_canceled"] == 0]

    monthly = []
    for (m, h), g in clean.groupby(["month", "hotel"]):
        monthly.append({"month": m, "hotel": h, "bookings": int(len(g)), "canceled": int(g["is_canceled"].sum()),
                        "cancel_rate": r(g["is_canceled"].mean()), "adr": r(g.loc[g["is_canceled"] == 0, "adr"].mean(), 2)})
    seasonal = []
    for (mn, h), g in clean.groupby([clean["arrival"].dt.month, "hotel"]):
        seasonal.append({"month": int(mn), "hotel": h, "adr": r(g.loc[g["is_canceled"] == 0, "adr"].mean(), 2),
                         "cancel_rate": r(g["is_canceled"].mean())})
    countries = clean.groupby("country")["is_canceled"].agg(["size", "mean"]).sort_values("size", ascending=False).head(12)
    prev = clean["previous_cancellations"].clip(upper=2).map({0: "0", 1: "1", 2: "2+"})
    req = clean["total_of_special_requests"].clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
    nights = clean["nights"].clip(upper=14)

    return {
        "funnel": {"raw": int(len(raw)), "after_duplicates": int(len(dedup)), "clean": int(len(clean)),
                   "duplicates": int(len(raw) - len(dedup)), "no_guests_or_negative_adr": int(len(dedup) - len(clean))},
        "summary": {
            "bookings": int(len(clean)), "cancel_rate": r(clean["is_canceled"].mean()),
            "first": clean["arrival"].min().strftime("%Y-%m-%d"), "last": clean["arrival"].max().strftime("%Y-%m-%d"),
            "hotels": {h: {"bookings": int(len(g)), "cancel_rate": r(g["is_canceled"].mean()),
                           "adr": r(g.loc[g["is_canceled"] == 0, "adr"].mean(), 2),
                           "nights": r(g.loc[g["is_canceled"] == 0, "nights"].mean(), 2)} for h, g in clean.groupby("hotel")},
            "median_lead_time": r(clean["lead_time"].median(), 0), "countries": int(clean["country"].nunique()),
            "adr": r(stayed["adr"].mean(), 2),
        },
        "monthly": monthly,
        "seasonal": seasonal,
        "by": {
            "lead_time": rate_table(clean, "lead_bucket", ["0-7", "8-30", "31-90", "91-180", "181-365", "366+"]),
            "deposit_type": rate_table(clean, "deposit_type"),
            "market_segment": rate_table(clean, "market_segment", min_n=100),
            "customer_type": rate_table(clean, "customer_type"),
            "repeated_guest": rate_table(clean.assign(g=clean["is_repeated_guest"].map({0: "New guest", 1: "Repeat guest"})), "g"),
            "previous_cancellations": rate_table(clean.assign(p=prev), "p", ["0", "1", "2+"]),
            "special_requests": rate_table(clean.assign(q=req), "q", ["0", "1", "2", "3+"]),
            "hotel": rate_table(clean, "hotel"),
        },
        "countries": [{"code": c, "n": int(n), "cancel_rate": r(m)} for c, (n, m) in countries.iterrows()],
        "nights": [{"nights": int(k), "n": int(v)} for k, v in nights.value_counts().sort_index().items()],
        "lead_time_hist": [{"days": int(k * 30), "n": int(v)} for k, v in (clean["lead_time"].clip(upper=599) // 30).value_counts().sort_index().items()],
    }


def model_results():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_split(RAW)
    X_dev, y_dev = pd.concat([X_train, X_valid], ignore_index=True), pd.concat([y_train, y_valid], ignore_index=True)
    model = CancellationEnsemble(members={"Extra Trees V2": make_extra_trees_v2(X_train), "MLP V0": make_mlp_v0(X_train)})
    model.fit(X_dev, y_dev)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p > model.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    def date_range(X):
        d = pd.to_datetime({"year": X["arrival_date_year"], "month": X["arrival_date_month"].map(MONTH_NUM),
                            "day": X["arrival_date_day_of_month"]})
        return [d.min().strftime("%Y-%m-%d"), d.max().strftime("%Y-%m-%d")]

    fpr, tpr, _ = roc_curve(y_test, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
    prec, rec, _ = precision_recall_curve(y_test, p)
    keep_pr = np.unique(np.linspace(0, len(prec) - 1, 120).astype(int))
    frac, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    sweep = []
    for t in np.round(np.arange(0.1, 0.91, 0.05), 2):
        pr_ = (p > t).astype(int)
        sweep.append({"t": float(t), "precision": r(precision_score(y_test, pr_, zero_division=0)),
                      "recall": r(recall_score(y_test, pr_)), "f1": r(f1_score(y_test, pr_)),
                      "f05": r(fbeta_score(y_test, pr_, beta=0.5)), "flagged": r(pr_.mean())})
    members = {name: r(roc_auc_score(y_test, est.predict_proba(X_test)[:, 1])) for name, est in model.fitted_members_.items()}
    members["Ensemble"] = r(roc_auc_score(y_test, p))

    groups = []
    test = X_test.assign(y=y_test.values, p=p, pred=pred)
    for col in ["hotel", "market_segment", "deposit_type", "customer_type"]:
        for k, g in test.groupby(col):
            if len(g) < 200 or g["y"].nunique() < 2:
                continue
            groups.append({"by": col, "key": str(k), "n": int(len(g)), "actual": r(g["y"].mean()),
                           "predicted": r(g["p"].mean()), "auc": r(roc_auc_score(g["y"], g["p"]))})

    # which booking details the model relies on: drop in test AUC when one input column is shuffled
    sample = X_test.sample(4000, random_state=SEED)
    ys = y_test.loc[sample.index]
    date_cols = ["arrival_date_year", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month"]
    groups_to_shuffle = {c: [c] for c in X_test.columns if c not in LEAKY_OR_POST_BOOKING and c not in date_cols}
    groups_to_shuffle["arrival_date"] = date_cols  # shuffled together so dates stay valid
    base = roc_auc_score(ys, model.predict_proba(sample)[:, 1])
    rng = np.random.default_rng(SEED)
    importance = []
    for name, cs in groups_to_shuffle.items():
        drops = []
        for _ in range(3):
            order = rng.permutation(len(sample))
            shuffled = sample.copy()
            shuffled[cs] = sample[cs].to_numpy()[order]
            drops.append(base - roc_auc_score(ys, model.predict_proba(shuffled)[:, 1]))
        importance.append({"feature": name, "auc_drop": r(np.mean(drops))})
    importance.sort(key=lambda x: -x["auc_drop"])

    hist = []
    bins = np.linspace(0, 1, 21)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        hist.append({"p": r(lo, 2), "kept": int(((y_test == 0) & m).sum()), "canceled": int(((y_test == 1) & m).sum())})

    return {
        "split": {"train": [int(len(X_train)), *date_range(X_train)], "valid": [int(len(X_valid)), *date_range(X_valid)],
                  "test": [int(len(X_test)), *date_range(X_test)]},
        "threshold": model.threshold,
        "test_cancel_rate": r(y_test.mean()),
        "metrics": {"roc_auc": r(roc_auc_score(y_test, p)), "pr_auc": r(average_precision_score(y_test, p)),
                    "brier": r(brier_score_loss(y_test, p)), "brier_baseline": r(brier_score_loss(y_test, np.full(len(p), y_dev.mean()))),
                    "precision": r(precision_score(y_test, pred)), "recall": r(recall_score(y_test, pred)),
                    "f1": r(f1_score(y_test, pred)), "f05": r(fbeta_score(y_test, pred, beta=0.5)),
                    "accuracy": r((tp + tn) / len(pred)), "baseline_accuracy": r(1 - y_test.mean())},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "members_auc": members,
        "roc": [{"fpr": r(fpr[i], 3), "tpr": r(tpr[i], 3)} for i in keep],
        "pr": [{"recall": r(rec[i], 3), "precision": r(prec[i], 3)} for i in keep_pr],
        "calibration": [{"predicted": r(a, 3), "actual": r(b, 3)} for a, b in zip(mean_pred, frac)],
        "threshold_sweep": sweep,
        "groups": groups,
        "importance": importance,
        "prob_hist": hist,
        "excluded": LEAKY_OR_POST_BOOKING,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"overview": overview()}
    print("Training the final ensemble (takes about a minute)...")
    data["model"] = model_results()
    for name, d in data.items():
        (OUT / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    m = data["model"]["metrics"]
    print(f"Test: ROC AUC {m['roc_auc']}, Brier {m['brier']}, precision {m['precision']}, recall {m['recall']}, F1 {m['f1']}")
    print("Notebook reference: ROC AUC 0.8040, Brier 0.1703, precision 0.6090, recall 0.5196, F1 0.5607")


if __name__ == "__main__":
    main()
