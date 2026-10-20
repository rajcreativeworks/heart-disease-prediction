from contextlib import asynccontextmanager

import numpy as np
from fastapi.testclient import TestClient

from heart_ml.api import app


class DummyModel:
    classes_ = np.array([0, 1])

    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.1, 0.9]])


@asynccontextmanager
async def model_lifespan(application):
    application.state.model = DummyModel()
    yield


def test_health_and_ready():
    original = app.router.lifespan_context
    app.router.lifespan_context = model_lifespan
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
    finally:
        app.router.lifespan_context = original


def test_predict_and_request_id():
    original = app.router.lifespan_context
    app.router.lifespan_context = model_lifespan
    payload = {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 160, "chol": 300, "fbs": 1,
        "restecg": 2, "thalach": 110, "exang": 1, "oldpeak": 4.0,
        "slope": 3, "ca": 2, "thal": 7,
    }
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=payload, headers={"X-Request-ID": "test-123"})
            assert response.status_code == 200
            assert response.json()["prediction"] == 1
            assert response.json()["confidence"] == 0.9
            assert response.headers["X-Request-ID"] == "test-123"
    finally:
        app.router.lifespan_context = original


def test_validation_error_does_not_echo_input():
    original = app.router.lifespan_context
    payload = {"age": -1}
    app.router.lifespan_context = model_lifespan
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            assert response.status_code == 422
            assert "Request validation failed" in response.text
            assert "-1" not in response.text
    finally:
        app.router.lifespan_context = original
