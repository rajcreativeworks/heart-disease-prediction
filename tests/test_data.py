from types import SimpleNamespace

import pandas as pd
import pytest

from heart_ml import data
from heart_ml.data import FEATURES, TARGET, fetch_dataset, validate_frame


def valid_frame():
    row = {feature: 1 for feature in FEATURES}
    return pd.DataFrame([row | {TARGET: 0}, row | {TARGET: 1}])


def test_validate_frame_accepts_binary_target():
    validate_frame(valid_frame())


def test_validate_frame_rejects_missing_feature():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_frame(valid_frame().drop(columns=["age"]))


def test_validate_frame_rejects_single_class():
    frame = valid_frame()
    frame[TARGET] = 0
    with pytest.raises(ValueError, match="both classes"):
        validate_frame(frame)


def test_validate_frame_rejects_empty_frame():
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_frame(valid_frame().iloc[:0])


def test_validate_frame_rejects_missing_target_values():
    frame = valid_frame()
    frame.loc[0, TARGET] = pd.NA
    with pytest.raises(ValueError, match="Target contains missing values"):
        validate_frame(frame)


def test_validate_frame_rejects_nonbinary_target():
    frame = valid_frame()
    frame.loc[0, TARGET] = 2
    with pytest.raises(ValueError, match="Target must be binary"):
        validate_frame(frame)


def test_fetch_dataset_cleans_columns_and_binarizes_target(monkeypatch):
    features = pd.DataFrame([{feature.upper(): 1 for feature in FEATURES}])
    targets = pd.DataFrame({" NUM ": [0]})
    targets.loc[1] = 2
    features.loc[1] = 2
    dataset = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr(data, "fetch_ucirepo", lambda *, id: dataset)

    result = fetch_dataset()

    assert result.columns.tolist() == FEATURES + [TARGET]
    assert result[TARGET].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("features", "targets", "message"),
    [
        (pd.DataFrame([{feature: 1 for feature in FEATURES}]),
         pd.DataFrame({"wrong": [0]}), "Expected target column"),
        (pd.DataFrame({"age": [1]}), pd.DataFrame({"num": [0]}),
         "missing expected features"),
    ],
)
def test_fetch_dataset_rejects_unexpected_schema(monkeypatch, features, targets, message):
    dataset = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr(data, "fetch_ucirepo", lambda *, id: dataset)

    with pytest.raises(ValueError, match=message):
        fetch_dataset()


def test_fetch_dataset_logs_and_reraises_fetch_error(monkeypatch):
    def fail_fetch(*, id):
        raise RuntimeError("offline")

    monkeypatch.setattr(data, "fetch_ucirepo", fail_fetch)

    with pytest.raises(RuntimeError, match="offline"):
        fetch_dataset()


def test_data_main_writes_fetched_frame(monkeypatch, tmp_path, capsys):
    output = tmp_path / "nested" / "dataset.csv"
    monkeypatch.setattr(data, "fetch_dataset", valid_frame)
    monkeypatch.setattr("sys.argv", ["heart-data", "--output", str(output)])

    data.main()

    assert pd.read_csv(output).shape == (2, len(FEATURES) + 1)
    assert "Saved 2 rows" in capsys.readouterr().out


def test_data_main_reraises_acquisition_error(monkeypatch, tmp_path):
    def fail_fetch():
        raise RuntimeError("offline")

    monkeypatch.setattr(data, "fetch_dataset", fail_fetch)
    monkeypatch.setattr("sys.argv", ["heart-data", "--output", str(tmp_path / "data.csv")])

    with pytest.raises(RuntimeError, match="offline"):
        data.main()
