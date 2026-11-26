"""Container entrypoint for the training service.

1. Wait for MLflow and Prefect to be reachable.
2. If no model has the `production` alias yet, run the pipeline once so the
   API has something to serve.
3. Serve the flow as a Prefect deployment ("training-pipeline/retrain"), so the
   monitor (or you, from the Prefect UI) can trigger retraining at any time.
"""

from __future__ import annotations

import os
import time

import requests
from mlflow.tracking import MlflowClient

from flow import training_pipeline
from mlops_core.config import load_config


def wait_for(url: str, name: str, timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code < 500:
                print(f"{name} is up")
                return
        except requests.RequestException:
            pass
        print(f"Waiting for {name}...")
        time.sleep(5)
    raise RuntimeError(f"{name} not reachable at {url}")


def production_model_exists() -> bool:
    cfg = load_config()
    try:
        MlflowClient().get_model_version_by_alias(cfg["model"]["registered_name"], cfg["model"]["alias"])
        return True
    except Exception:
        return False


if __name__ == "__main__":
    wait_for(os.environ["MLFLOW_TRACKING_URI"] + "/health", "MLflow")
    wait_for(os.environ["PREFECT_API_URL"] + "/health", "Prefect")

    if not production_model_exists():
        print("No production model registered — running initial training.")
        training_pipeline(trigger="initial")

    training_pipeline.serve(
        name="retrain",
        parameters={"use_feedback": True, "trigger": "drift"},
    )
