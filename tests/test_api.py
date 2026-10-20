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
            response_data = response.json()
            assert "-1" not in str(response_data.get("detail", ""))
    finally:
        app.router.lifespan_context = original


def test_readiness_prediction_and_metrics_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("heart_ml.api.MODEL_PATH", tmp_path / "missing.joblib")
    payload = {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 160, "chol": 300, "fbs": 1,
        "restecg": 2, "thalach": 110, "exang": 1, "oldpeak": 4.0,
        "slope": 3, "ca": 2, "thal": 7,
    }

    with TestClient(app) as client:
        ready_response = client.get("/ready")
        prediction_response = client.post("/predict", json=payload)
        metrics_response = client.get("/metrics")

    assert ready_response.status_code == 503
    assert prediction_response.status_code == 503
    assert metrics_response.status_code == 200
    assert "http_requests_total" in metrics_response.text


def test_lifespan_loads_existing_model(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr("heart_ml.api.MODEL_PATH", model_path)
    monkeypatch.setattr("heart_ml.api.joblib.load", lambda path: DummyModel())

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200


def test_lifespan_raises_clear_error_for_unreadable_model(monkeypatch, tmp_path):
    model_path = tmp_path / "broken.joblib"
    model_path.touch()
    monkeypatch.setattr("heart_ml.api.MODEL_PATH", model_path)

    def fail_load(path):
        raise ValueError("invalid model")

    monkeypatch.setattr("heart_ml.api.joblib.load", fail_load)

    try:
        with TestClient(app):
            pass
    except RuntimeError as error:
        assert str(error) == "Model artifact could not be loaded"
    else:
        raise AssertionError("Expected model loading to fail")


def test_prediction_failure_returns_sanitized_error(monkeypatch):
    class BrokenModel(DummyModel):
        def predict(self, frame):
            raise RuntimeError("sensitive inference details")

    @asynccontextmanager
    async def broken_model_lifespan(application):
        application.state.model = BrokenModel()
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = broken_model_lifespan
    payload = {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 160, "chol": 300, "fbs": 1,
        "restecg": 2, "thalach": 110, "exang": 1, "oldpeak": 4.0,
        "slope": 3, "ca": 2, "thal": 7,
    }
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "sensitive inference details" not in response.text
    finally:
        app.router.lifespan_context = original
