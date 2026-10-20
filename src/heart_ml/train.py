"""Train candidate classifiers, log MLflow runs, and save reproducible artifacts."""
from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_ml.data import FEATURES, TARGET, validate_frame
from heart_ml.logging_config import configure_logging
from heart_ml.path_utils import validate_output_path

logger = logging.getLogger(__name__)
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [column for column in FEATURES if column not in NUMERIC]


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("preprocess", preprocessor()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
    }


def calculate_metrics(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    prediction = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def train(data: str | Path, output: str | Path, tracking: bool = True,
          quick: bool = False) -> dict:
    logger.info("event=training_entry data=%s output=%s tracking=%s quick=%s",
                data, output, tracking, quick)
    try:
        frame = pd.read_csv(data)
        validate_frame(frame)
        x, y = frame[FEATURES], frame[TARGET].astype(int)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=42)
        if tracking:
            mlflow.set_experiment("heart-disease-classification")
        comparisons: dict[str, dict[str, float]] = {}
        best_name, best_score = "", -np.inf
        for name, estimator in candidates().items():
            run_context = mlflow.start_run(run_name=name) if tracking else nullcontext()
            with run_context:
                cv_results = cross_validate(
                    estimator, x_train, y_train, cv=cv,
                    scoring=["accuracy", "precision", "recall", "roc_auc"], n_jobs=1,
                )
                cv_metrics = {
                    f"cv_{metric}": float(np.mean(cv_results[f"test_{metric}"]))
                    for metric in ("accuracy", "precision", "recall", "roc_auc")
                }
                estimator.fit(x_train, y_train)
                if tracking:
                    mlflow.log_params({
                        "model": name, "seed": 42, "test_fraction": 0.2,
                        "cv_folds": cv.get_n_splits(), "dataset_rows": len(frame),
                    })
                    mlflow.log_metrics(cv_metrics)
                comparisons[name] = cv_metrics
                logger.info("event=candidate_evaluated model=%s cv_roc_auc=%.4f",
                            name, cv_metrics["cv_roc_auc"])
                if cv_metrics["cv_roc_auc"] > best_score:
                    best_name, best_score = name, cv_metrics["cv_roc_auc"]

        search_space = (
            {"model__C": [0.1, 1.0, 10.0]}
            if best_name == "logistic_regression"
            else {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3]}
        )
        search = GridSearchCV(
            candidates()[best_name], search_space, cv=cv, scoring="roc_auc", n_jobs=1, refit=True
        )
        search.fit(x_train, y_train)
        final_model = search.best_estimator_
        test_metrics = calculate_metrics(final_model, x_test, y_test)
        output_path = validate_output_path(output, Path.cwd())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, output_path)
        metadata = {
            "selected_model": best_name,
            "best_params": search.best_params_,
            "cv_roc_auc": float(search.best_score_),
            "test_metrics": test_metrics,
            "features": FEATURES,
            "seed": 42,
            "test_rows": len(x_test),
            "warning": "Educational dataset/model; not for diagnosis or treatment.",
        }
        output_path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        report_dir = Path("reports/training")
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "metrics.json").write_text(
            json.dumps(metadata | {"cv_comparison": comparisons}, indent=2), encoding="utf-8"
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(final_model, x_test, y_test, ax=ax)
        fig.tight_layout()
        fig.savefig(report_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(final_model, x_test, y_test, ax=ax)
        fig.tight_layout()
        fig.savefig(report_dir / "roc_curve.png", dpi=160)
        plt.close(fig)
        if tracking:
            with mlflow.start_run(run_name="selected_model_final"):
                mlflow.log_params({"selected_model": best_name, **{
                    key: str(value) for key, value in search.best_params_.items()
                }})
                mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
                mlflow.log_artifact(str(output_path.with_suffix(".metadata.json")))
                mlflow.log_artifact(str(report_dir / "confusion_matrix.png"))
                mlflow.log_artifact(str(report_dir / "roc_curve.png"))
                mlflow.log_artifact(str(output_path))
        logger.info("event=training_exit selected_model=%s test_roc_auc=%.4f output=%s",
                    best_name, test_metrics["roc_auc"], output_path)
        return metadata
    except Exception:
        logger.exception("event=training_error data=%s", data)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/heart_disease.csv")
    parser.add_argument("--output", default="models/model.joblib")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use 3-fold CV for smoke testing.")
    args = parser.parse_args()
    train(args.data, args.output, tracking=args.mlflow, quick=args.quick)


if __name__ == "__main__":
    configure_logging()
    main()
