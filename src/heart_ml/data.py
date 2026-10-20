"""Data acquisition and input validation for the UCI Heart Disease dataset."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from heart_ml.path_utils import validate_output_path

logger = logging.getLogger(__name__)
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"


def fetch_dataset() -> pd.DataFrame:
    """Fetch UCI dataset ID 45 and binarize target: 0=no disease, >0=disease."""
    logger.info("event=data_fetch_start dataset=uci_heart_disease dataset_id=45")
    try:
        dataset = fetch_ucirepo(id=45)
        x = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        x.columns = [str(column).strip().lower() for column in x.columns]
        y.columns = [str(column).strip().lower() for column in y.columns]
        if "num" not in y.columns:
            raise ValueError(f"Expected target column 'num'; received {list(y.columns)}")
        missing = sorted(set(FEATURES) - set(x.columns))
        if missing:
            raise ValueError(f"Dataset is missing expected features: {missing}")
        frame = x[FEATURES].copy()
        for column in FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        target = pd.to_numeric(y["num"], errors="coerce")
        valid = target.notna()
        frame = frame.loc[valid].copy()
        frame[TARGET] = (target.loc[valid] > 0).astype(int)
        frame = frame.reset_index(drop=True)
        validate_frame(frame)
        logger.info(
            "event=data_fetch_complete rows=%d columns=%d missing_values=%d",
            len(frame), len(frame.columns), int(frame.isna().sum().sum()),
        )
        return frame
    except Exception:
        logger.exception("event=data_fetch_failed dataset=uci_heart_disease")
        raise


def validate_frame(frame: pd.DataFrame) -> None:
    """Validate schema and target before training or saving."""
    required = set(FEATURES + [TARGET])
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame[TARGET].isna().any():
        raise ValueError("Target contains missing values")
    target_values = set(frame[TARGET].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError("Target must be binary (0/1)")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Target must contain both classes (0 and 1)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/heart_disease.csv")
    args = parser.parse_args()
    output = validate_output_path(args.output, Path.cwd())
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("event=data_acquisition_entry output=%s", output)
    try:
        frame = fetch_dataset()
        frame.to_csv(output, index=False)
        logger.info("event=data_acquisition_exit output=%s rows=%d", output, len(frame))
        print(f"Saved {len(frame)} rows to {output}")
    except Exception:
        logger.exception("event=data_acquisition_error")
        raise


if __name__ == "__main__":
    from heart_ml.logging_config import configure_logging
    configure_logging()
    main()
