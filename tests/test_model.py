import numpy as np
import pandas as pd
import pytest

from heart_ml import train as train_module
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


@pytest.mark.parametrize("tracking", [False, True])
def test_train_writes_model_metadata_and_reports(monkeypatch, tmp_path, tracking):
    frame = pd.concat([sample_frame()] * 10, ignore_index=True)
    frame["target"] = np.tile([0, 1, 0, 1], 10)
    source = tmp_path / "training.csv"
    output = tmp_path / "models" / "model.joblib"
    frame.to_csv(source, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_module, "candidates", lambda: {
        "logistic_regression": candidates()["logistic_regression"]
    })
    monkeypatch.setattr(train_module, "cross_validate", lambda *args, **kwargs: {
        f"test_{metric}": np.array([0.8, 0.85, 0.9])
        for metric in ("accuracy", "precision", "recall", "roc_auc")
    })
    monkeypatch.setattr(train_module.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(
        train_module.mlflow, "start_run", lambda **kwargs: train_module.nullcontext()
    )
    monkeypatch.setattr(train_module.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(train_module.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(train_module.mlflow, "log_artifact", lambda path: None)

    result = train_module.train(source, output, tracking=tracking, quick=True)

    assert result["selected_model"] == "logistic_regression"
    assert result["test_rows"] == 8
    assert output.is_file()
    assert output.with_suffix(".metadata.json").is_file()
    assert (tmp_path / "reports" / "training" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "training" / "confusion_matrix.png").is_file()
    assert (tmp_path / "reports" / "training" / "roc_curve.png").is_file()


def test_train_rejects_invalid_dataset(tmp_path):
    source = tmp_path / "invalid.csv"
    pd.DataFrame({"bad": [1]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        train_module.train(source, tmp_path / "model.joblib", tracking=False, quick=True)


def test_train_main_passes_cli_options(monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "train", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("sys.argv", ["heart-train", "--data", "input.csv", "--output",
                                     "model.joblib", "--mlflow", "--quick"])

    train_module.main()

    assert calls == [( ("input.csv", "model.joblib"), {"tracking": True, "quick": True})]
