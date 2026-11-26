"""Traffic simulator: replays unseen test rows against the model API and sends
the true label back via /feedback. Use --drift to shift the input
distribution (configured in config.yaml -> drift.inject).

Examples (from the host):
    docker compose run --rm simulator --requests 500
    docker compose run --rm simulator --requests 500 --drift
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

from mlops_core.config import feature_columns, load_config, target_column
from mlops_core.drift import inject_drift

SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/shared"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://api:8000"))
    parser.add_argument("--requests", type=int, default=300, help="number of requests to send")
    parser.add_argument("--rps", type=float, default=5.0, help="requests per second")
    parser.add_argument("--drift", action="store_true", help="inject drift into the inputs")
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.02,
        help="share of deliberately invalid requests (to exercise error metrics)",
    )
    parser.add_argument("--no-feedback", action="store_true", help="don't send ground-truth labels")
    args = parser.parse_args()

    cfg = load_config()
    data = pd.read_csv(SHARED_DIR / "test.csv")
    if args.drift:
        data = inject_drift(data, cfg, seed=int(time.time()))
    feats, target = feature_columns(cfg), target_column(cfg)
    source = "simulator-drift" if args.drift else "simulator"

    sent = ok = errors = 0
    delay = 1.0 / args.rps if args.rps > 0 else 0
    print(f"Sending {args.requests} requests to {args.api_url} (drift={args.drift})")
    while sent < args.requests:
        row = data.sample(1).iloc[0]
        payload = {c: (float(row[c]) if c in cfg["features"]["numeric"] else str(row[c])) for c in feats}
        if random.random() < args.error_rate:
            payload[feats[0]] = -999  # out of bounds -> 422
        try:
            resp = requests.post(
                f"{args.api_url}/predict", json=payload, timeout=10, headers={"X-Source": source}
            )
            if resp.ok:
                ok += 1
                pid = resp.json().get("prediction_id")
                if pid is not None and not args.no_feedback:
                    requests.post(
                        f"{args.api_url}/feedback",
                        json={"prediction_id": pid, "actual": int(row[target])},
                        timeout=10,
                    )
            else:
                errors += 1
        except requests.RequestException as exc:
            errors += 1
            print(f"request failed: {exc}")
        sent += 1
        if sent % 50 == 0:
            print(f"{sent}/{args.requests} sent — {ok} ok, {errors} errors")
        time.sleep(delay)
    print(f"Done: {ok} ok, {errors} errors")


if __name__ == "__main__":
    main()
