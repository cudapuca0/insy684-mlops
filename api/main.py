"""Model API (FastAPI) for the hotel cancellation model.

GET  /health   -> is the service up, and is the model loaded?
POST /predict  -> cancellation probability for one booking (also saved to PostgreSQL)
GET  /metrics  -> request counts and latency for Prometheus
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

# The saved model refers to classes in hotel_model.py, so Python must be able to import it.
sys.path.insert(0, os.environ.get("HOTEL_MODEL_DIR", "frontend/pipeline"))

MODEL_PATH = os.environ.get("MODEL_PATH", "model/cancellation_model.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "local-file")  # becomes the registry version later
THRESHOLD = 0.45  # same cut-off the model was tuned with

log = logging.getLogger("api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions by outcome", ["outcome"])

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


class Booking(BaseModel):
    """One hotel booking. Anything outside these rules is rejected with a 422."""

    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(..., ge=0, le=800, description="Days between booking and arrival")
    arrival_date_year: int = Field(..., ge=2015, le=2030)
    arrival_date_month: Literal[tuple(MONTHS)]  # type: ignore[valid-type]
    arrival_date_day_of_month: int = Field(..., ge=1, le=31)
    stays_in_weekend_nights: int = Field(..., ge=0, le=20)
    stays_in_week_nights: int = Field(..., ge=0, le=50)
    adults: int = Field(..., ge=0, le=10)
    children: int = Field(0, ge=0, le=10)
    babies: int = Field(0, ge=0, le=10)
    meal: Literal["BB", "HB", "FB", "SC", "Undefined"]
    country: str = Field("PRT", min_length=2, max_length=4)
    market_segment: str
    is_repeated_guest: Literal[0, 1] = 0
    previous_cancellations: int = Field(0, ge=0, le=30)
    previous_bookings_not_canceled: int = Field(0, ge=0, le=80)
    reserved_room_type: str
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    agent: float | None = None
    company: float | None = None
    customer_type: Literal["Transient", "Contract", "Transient-Party", "Group"]
    adr: float = Field(..., ge=0, le=1000, description="Average daily rate")


class Prediction(BaseModel):
    prediction_id: int | None = None  # row id in PostgreSQL, or None if logging is off
    cancel_probability: float
    will_cancel: bool
    label: str


model = None
engine = None  # database connection, or None when DATABASE_URL is not set


def make_engine():
    """Connect to PostgreSQL if DATABASE_URL is set, for example
    postgresql+psycopg://user:password@postgres:5432/monitoring"""
    url = os.environ.get("DATABASE_URL")
    if not url:
        log.info("DATABASE_URL not set: predictions will not be logged.")
        return None
    return create_engine(url, pool_pre_ping=True)


def log_prediction(features: dict, probability: float, prediction: int, source: str):
    """Save one prediction to the predictions table. Never raises: a database
    problem must not stop the user from getting their answer."""
    if engine is None:
        return None
    try:
        with engine.begin() as conn:
            return conn.execute(
                text(
                    "INSERT INTO predictions "
                    "(features, probability, prediction, threshold, model_version, source) "
                    "VALUES (CAST(:features AS JSONB), :probability, :prediction, "
                    ":threshold, :model_version, :source) RETURNING id"
                ),
                {
                    "features": json.dumps(features),
                    "probability": probability,
                    "prediction": prediction,
                    "threshold": THRESHOLD,
                    "model_version": MODEL_VERSION,
                    "source": source,
                },
            ).scalar_one()
    except Exception as exc:
        log.error("Could not log prediction: %s", exc)
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, engine
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
    engine = make_engine()
    yield
    if engine is not None:
        engine.dispose()


app = FastAPI(title="Hotel Cancellation API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    start, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if request.url.path != "/metrics":
            LATENCY.labels(request.url.path).observe(time.perf_counter() - start)
            REQUESTS.labels(request.url.path, str(status)).inc()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(booking: Booking, request: Request) -> Prediction:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    features = booking.model_dump()
    row = pd.DataFrame([features])
    probability = float(model.predict_proba(row)[0, 1])
    will_cancel = probability > THRESHOLD
    PREDICTIONS.labels("cancel" if will_cancel else "keep").inc()
    # Callers say who they are with an X-Source header, for example "frontend" or "simulator".
    source = request.headers.get("X-Source", "api")
    prediction_id = log_prediction(features, probability, int(will_cancel), source)
    return Prediction(
        prediction_id=prediction_id,
        cancel_probability=round(probability, 4),
        will_cancel=will_cancel,
        label="Likely to cancel" if will_cancel else "Likely to keep booking",
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
