import numpy as np
import pandas as pd

from heart_ml.data import FEATURES
from heart_ml.train import calculate_metrics, candidates


def sample_frame():
    row_a = {
        "age": 45, "sex": 1, "cp": 1, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0, "oldpeak": 0.0,
        "slope": 1, "ca": 0, "thal": 3,
    }
    row_b = {
        "age": 65, "sex": 0, "cp": 4, "trestbps": 160, "chol": 280,
        "fbs": 1, "restecg": 2, "thalach": 110, "exang": 1, "oldpeak": 3.0,
        "slope": 3, "ca": 2, "thal": 7,
    }
    return pd.DataFrame([row_a, row_b, row_a | {"age": 50}, row_b | {"age": 60}], columns=FEATURES)


def test_candidate_pipeline_predicts():
    x = sample_frame()
    y = np.array([0, 1, 0, 1])
    model = candidates()["logistic_regression"]
    model.fit(x, y)
    assert set(model.predict(x)).issubset({0, 1})
    assert model.predict_proba(x).shape == (4, 2)


def test_calculate_metrics_returns_expected_keys():
    x = sample_frame()
    y = pd.Series([0, 1, 0, 1])
    model = candidates()["logistic_regression"].fit(x, y)
    result = calculate_metrics(model, x, y)
    assert set(result) == {"accuracy", "precision", "recall", "roc_auc"}
