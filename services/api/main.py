"""Model API (FastAPI).

POST /predict   -> prediction for one record (validated against config bounds)
POST /feedback  -> attach the true label to an earlier prediction
GET  /health    -> liveness + whether a model is loaded
GET  /model-info-> which registered model version is being served
GET  /metrics   -> Prometheus metrics (request count, latency, errors, predictions)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, create_model

from mlops_core.config import feature_columns, load_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")
CFG = load_config()

# --------------------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
ERRORS = Counter("api_errors_total", "Requests that returned 4xx/5xx", ["endpoint", "status"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["label"])
PROBABILITY = Histogram(
    "model_prediction_probability",
    "Predicted probability of the positive class",
    buckets=[i / 10 for i in range(11)],
)
MODEL_VERSION = Gauge("model_version_info", "Registered model version being served")


# --------------------------------------------------------------------------- input schema
def _build_input_model() -> type[BaseModel]:
    """Pydantic model generated from config.yaml so bad input is rejected with 422."""
    fields: dict[str, Any] = {}
    bounds = CFG["features"].get("bounds", {})
    for col in CFG["features"]["numeric"]:
        lo, hi = bounds.get(col, (None, None))
        fields[col] = (float, Field(..., ge=lo, le=hi))
    for col in CFG["features"]["categorical"]:
        fields[col] = (str, Field(..., min_length=1))
    return create_model("PredictionInput", **fields)


PredictionInput = _build_input_model()


class PredictionOutput(BaseModel):
    prediction_id: int | None
    prediction: int
    label: str
    probability: float
    model_version: str | None


class Feedback(BaseModel):
    prediction_id: int
    actual: int = Field(..., ge=0, le=1)


# --------------------------------------------------------------------------- model holder
class ModelHolder:
    """Loads the model from the MLflow registry (or a local path for tests) and
    reloads it when the `production` alias moves to a new version."""

    def __init__(self) -> None:
        self.model = None
        self.version: str | None = None
        self.lock = threading.Lock()
        self.name = CFG["model"]["registered_name"]
        self.alias = CFG["model"]["alias"]

    def _current_registry_version(self) -> str:
        from mlflow.tracking import MlflowClient

        return MlflowClient().get_model_version_by_alias(self.name, self.alias).version

    def load(self) -> bool:
        import mlflow.sklearn

        local_path = os.environ.get("MODEL_PATH")
        try:
            if local_path:
                model, version = mlflow.sklearn.load_model(local_path), "local"
            else:
                version = self._current_registry_version()
                if version == self.version:
                    return True
                model = mlflow.sklearn.load_model(f"models:/{self.name}/{version}")
            with self.lock:
                self.model, self.version = model, version
            if version.isdigit():
                MODEL_VERSION.set(int(version))
            log.info("Loaded model %s version %s", self.name, version)
            return True
        except Exception as exc:
            log.warning("Model not available yet: %s", exc)
            return False

    def watch(self, interval: int) -> None:
        while True:
            self.load()
            time.sleep(interval)


holder = ModelHolder()


# --------------------------------------------------------------------------- prediction log
class PredictionStore:
    def __init__(self) -> None:
        self.engine = None
        if os.environ.get("PREDICTION_LOGGING", "true").lower() == "true":
            try:
                from sqlalchemy import create_engine

                from mlops_core.db import database_url, ensure_schema

                self.engine = create_engine(database_url(), pool_pre_ping=True)
                ensure_schema(self.engine)
            except Exception as exc:
                log.warning("Prediction logging disabled (database unavailable): %s", exc)
                self.engine = None

    def log(
        self, features: dict, prediction: int, probability: float, version: str | None, source: str
    ) -> int | None:
        if self.engine is None:
            return None
        from sqlalchemy import text

        with self.engine.begin() as conn:
            return conn.execute(
                text(
                    "INSERT INTO predictions (features, prediction, probability, model_version, source) "
                    "VALUES (CAST(:f AS JSONB), :p, :prob, :v, :s) RETURNING id"
                ),
                {"f": json.dumps(features), "p": prediction, "prob": probability, "v": version, "s": source},
            ).scalar_one()

    def feedback(self, prediction_id: int, actual: int) -> bool:
        if self.engine is None:
            return False
        from sqlalchemy import text

        with self.engine.begin() as conn:
            res = conn.execute(
                text("UPDATE predictions SET actual = :a WHERE id = :i"), {"a": actual, "i": prediction_id}
            )
            return res.rowcount > 0


store: PredictionStore | None = None


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_: FastAPI):
    global store
    store = PredictionStore()
    holder.load()
    if not os.environ.get("MODEL_PATH"):
        # Keep polling the registry so a retrained model is picked up automatically.
        interval = int(os.environ.get("MODEL_RELOAD_SECONDS", "30"))
        threading.Thread(target=holder.watch, args=(interval,), daemon=True).start()
    yield


app = FastAPI(title=CFG["project"]["name"], version="1.0", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if endpoint != "/metrics":
            LATENCY.labels(endpoint).observe(time.perf_counter() - start)
            REQUESTS.labels(endpoint, request.method, str(status)).inc()
            if status >= 400:
                ERRORS.labels(endpoint, str(status)).inc()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": holder.model is not None, "model_version": holder.version}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "name": holder.name,
        "alias": holder.alias,
        "version": holder.version,
        "features": feature_columns(CFG),
        "classes": CFG["classes"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request) -> PredictionOutput:  # type: ignore[valid-type]
    if holder.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet — try again shortly.")

    features = payload.model_dump()
    X = pd.DataFrame([features])[feature_columns(CFG)]
    with holder.lock:
        proba = float(holder.model.predict_proba(X)[0, 1])
        version = holder.version
    prediction = int(proba >= 0.5)
    label = CFG["classes"]["positive_label"] if prediction else CFG["classes"]["negative_label"]
    PREDICTIONS.labels(label).inc()
    PROBABILITY.observe(proba)

    source = request.headers.get("X-Source", "api")
    try:
        prediction_id = store.log(features, prediction, proba, version, source) if store else None
    except Exception as exc:  # never fail a prediction because logging failed
        log.error("Failed to log prediction: %s", exc)
        prediction_id = None

    return PredictionOutput(
        prediction_id=prediction_id,
        prediction=prediction,
        label=label,
        probability=round(proba, 4),
        model_version=version,
    )


@app.post("/feedback")
def feedback(body: Feedback) -> dict:
    if store is None or not store.feedback(body.prediction_id, body.actual):
        raise HTTPException(status_code=404, detail="Prediction not found or logging disabled")
    return {"status": "recorded"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
