"""FastAPI prediction service with JSON logs and Prometheus instrumentation.

This is an educational demonstration, not a clinical decision-support system.
Do not send patient-identifiable data to this service.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from heart_ml import __version__
from heart_ml.data import FEATURES
from heart_ml.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", __version__)
REQUESTS = Counter("http_requests_total", "HTTP requests handled", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "path"])
PREDICTIONS = Counter("model_predictions_total", "Model predictions by class", ["prediction"])
ERRORS = Counter("api_errors_total", "API errors by category", ["category"])


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("event=application_start model_path=%s model_version=%s", MODEL_PATH, MODEL_VERSION)
    application.state.model = None
    if MODEL_PATH.is_file():
        try:
            # SECURITY: only load model artifacts produced by a trusted build pipeline.
            application.state.model = joblib.load(MODEL_PATH)
            logger.info("event=model_load_success model_version=%s", MODEL_VERSION)
        except Exception:
            logger.exception("event=model_load_failure model_path=%s", MODEL_PATH)
            raise RuntimeError("Model artifact could not be loaded") from None
    else:
        logger.warning("event=model_artifact_missing model_path=%s", MODEL_PATH)
    yield
    logger.info("event=application_shutdown")


app = FastAPI(
    title="Heart Disease Prediction API",
    version=__version__,
    description="Educational demonstration only; not for diagnosis or treatment.",
    lifespan=lifespan,
)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=3, le=7)


class PredictionOutput(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["no_disease", "disease_present"]
    confidence: float
    probabilities: dict[str, float]
    model_version: str


@app.middleware("http")
async def request_telemetry(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    logger.debug("event=request_entry request_id=%s method=%s path=%s",
                 request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        ERRORS.labels("unhandled_exception").inc()
        logger.exception("event=request_unhandled_error request_id=%s path=%s",
                         request_id, request.url.path)
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        REQUESTS.labels(request.method, request.url.path, str(status_code)).inc()
        LATENCY.labels(request.method, request.url.path).observe(duration_ms / 1000)
        logger.info(
            "event=request_exit request_id=%s method=%s path=%s status_code=%d duration_ms=%.3f",
            request_id, request.method, request.url.path, status_code, duration_ms,
        )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    ERRORS.labels("validation").inc()
    request_id = getattr(request.state, "request_id", "unknown")
    # Do not log request body or field values; those may contain sensitive data.
    logger.warning("event=request_validation_failed request_id=%s path=%s error_count=%d",
                   request_id, request.url.path, len(exc.errors()))
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.get("/health")
def health():
    """Liveness probe: process is running; model need not be loaded."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready(request: Request):
    """Readiness probe: returns 503 until the model is loaded."""
    if request.app.state.model is None:
        logger.warning("event=readiness_check_failed reason=model_not_loaded")
        raise HTTPException(status_code=503, detail="Model artifact is not loaded")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request):
    model = request.app.state.model
    request_id = getattr(request.state, "request_id", "unknown")
    if model is None:
        ERRORS.labels("model_unavailable").inc()
        logger.error("event=prediction_rejected request_id=%s reason=model_not_loaded", request_id)
        raise HTTPException(status_code=503, detail="Model is not loaded; train the model first")
    try:
        row = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        probability_map = {
            str(int(label)): float(probability)
            for label, probability in zip(model.classes_, probabilities, strict=True)
        }
        confidence = float(max(probabilities))
        PREDICTIONS.labels(str(prediction)).inc()
        logger.info(
            "event=prediction_complete request_id=%s prediction=%d confidence=%.4f model_version=%s",
            request_id, prediction, confidence, MODEL_VERSION,
        )
        return PredictionOutput(
            prediction=prediction,
            label="disease_present" if prediction else "no_disease",
            confidence=confidence,
            probabilities=probability_map,
            model_version=MODEL_VERSION,
        )
    except HTTPException:
        raise
    except Exception:
        ERRORS.labels("inference").inc()
        logger.exception("event=prediction_failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Prediction failed") from None
