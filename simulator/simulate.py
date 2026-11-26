"""Traffic simulator: sends unseen bookings to the model API, optionally with drift.

Unseen = bookings arriving in the test period (from 2017-05-05 by default), which the final
model never trained on. Each one is posted to /predict in the format api/main.py expects.

Examples:
    python simulator/simulate.py --requests 500
    python simulator/simulate.py --requests 500 --drift
    docker compose run --rm simulator --requests 500 --drift      # once it's in docker-compose

Options:
    --drift          shift the inputs (see drift.py) so monitoring should flag data drift
    --error-rate     share of deliberately invalid requests, to exercise the API's error metrics
    --feedback       also send the true outcome to /feedback (if the API offers it)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift import inject_drift  # noqa: E402

FIELDS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "meal",
    "country", "market_segment", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "deposit_type", "agent", "company",
    "customer_type", "adr",
]
INT_FIELDS = {
    "lead_time", "arrival_date_year", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def load_unseen(path: str | Path, since: str) -> pd.DataFrame:
    """Bookings arriving on or after `since`, cleaned like train.py, valid for the API."""
    df = pd.read_csv(path).drop_duplicates()
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df[(guests > 0) & (df["adr"] >= 0) & df["country"].notna()]
    arrival = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": df["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)}),
        "day": df["arrival_date_day_of_month"],
    })
    df = df[arrival >= pd.Timestamp(since)].copy()
    df["children"] = df["children"].fillna(0)
    return df.reset_index(drop=True)


def to_payload(row: pd.Series) -> dict:
    """One CSV row -> the JSON body /predict expects (NaN agent/company become null)."""
    payload = {}
    for f in FIELDS:
        v = row[f]
        if f in ("agent", "company"):
            payload[f] = None if pd.isna(v) else float(v)
        elif f in INT_FIELDS:
            payload[f] = int(v)
        elif f == "adr":
            payload[f] = float(v)
        else:
            payload[f] = str(v)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--data", default=os.environ.get("DATA_PATH", "data/raw/hotel_bookings.csv"))
    parser.add_argument("--since", default="2017-05-05", help="first arrival date counted as unseen")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--rps", type=float, default=5.0, help="requests per second (0 = as fast as possible)")
    parser.add_argument("--drift", action="store_true", help="inject drift into the inputs")
    parser.add_argument("--drift-strength", type=float, default=1.0)
    parser.add_argument("--error-rate", type=float, default=0.02)
    parser.add_argument("--feedback", action="store_true", help="send true outcomes to /feedback")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    data = load_unseen(args.data, args.since)
    if args.drift:
        data = inject_drift(data, seed=seed, strength=args.drift_strength)
    source = "simulator-drift" if args.drift else "simulator"
    print(f"{len(data):,} unseen bookings loaded; sending {args.requests} to {args.api_url} (drift={args.drift})")

    ok = errors = 0
    delay = 1.0 / args.rps if args.rps > 0 else 0.0
    for sent in range(1, args.requests + 1):
        row = data.sample(1, random_state=random.randint(0, 2**31 - 1)).iloc[0]
        payload = to_payload(row)
        if random.random() < args.error_rate:
            payload["lead_time"] = -1  # invalid on purpose -> 422
        try:
            resp = requests.post(f"{args.api_url}/predict", json=payload, timeout=10,
                                 headers={"X-Source": source})
            if resp.ok:
                ok += 1
                pid = resp.json().get("prediction_id")
                if args.feedback and pid is not None:
                    requests.post(f"{args.api_url}/feedback",
                                  json={"prediction_id": pid, "actual": int(row["is_canceled"])}, timeout=10)
            else:
                errors += 1
        except requests.RequestException as exc:
            errors += 1
            print(f"request failed: {exc}")
        if sent % 50 == 0:
            print(f"{sent}/{args.requests} sent: {ok} ok, {errors} errors")
        time.sleep(delay)
    print(f"Done: {ok} ok, {errors} errors")


if __name__ == "__main__":
    main()
