import pandas as pd
import pytest

from heart_ml.data import FEATURES, TARGET, validate_frame


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
