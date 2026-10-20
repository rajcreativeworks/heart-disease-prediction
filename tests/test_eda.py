import pandas as pd
import pytest

from heart_ml.data import FEATURES, TARGET
from heart_ml.eda import generate_eda, main


def valid_frame():
    rows = []
    for index in range(8):
        rows.append({feature: index + offset for offset, feature in enumerate(FEATURES)}
                    | {TARGET: index % 2})
    return pd.DataFrame(rows)


def test_generate_eda_writes_expected_artifacts(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "eda"
    valid_frame().to_csv(source, index=False)

    generate_eda(source, output)

    assert (output / "summary_statistics.csv").is_file()
    assert (output / "missingness.csv").is_file()
    assert (output / "class_balance.csv").is_file()
    assert (output / "class_balance.png").is_file()
    assert (output / "correlation_heatmap.png").is_file()
    assert all((output / f"hist_{feature}.png").is_file() for feature in FEATURES)


def test_generate_eda_reraises_invalid_input(tmp_path):
    source = tmp_path / "invalid.csv"
    pd.DataFrame({"bad": [1]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_eda(source, tmp_path / "eda")


def test_eda_main_uses_cli_paths(monkeypatch, tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "cli-eda"
    valid_frame().to_csv(source, index=False)
    monkeypatch.setattr("sys.argv", ["heart-eda", "--input", str(source), "--output", str(output)])

    main()

    assert (output / "class_balance.png").is_file()