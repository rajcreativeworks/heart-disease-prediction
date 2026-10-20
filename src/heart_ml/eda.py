"""Generate reproducible exploratory-data-analysis artifacts."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_ml.data import FEATURES, TARGET, validate_frame
from heart_ml.logging_config import configure_logging
from heart_ml.path_utils import validate_output_path

logger = logging.getLogger(__name__)


def generate_eda(input_path: str | Path, output_dir: str | Path) -> None:
    logger.info("event=eda_entry input=%s output=%s", input_path, output_dir)
    try:
        frame = pd.read_csv(input_path)
        validate_frame(frame)
        out = validate_output_path(output_dir, Path.cwd())
        out.mkdir(parents=True, exist_ok=True)
        frame.describe(include="all").transpose().to_csv(out / "summary_statistics.csv")
        frame.isna().sum().rename("missing_count").to_csv(out / "missingness.csv")
        frame[TARGET].value_counts().sort_index().rename_axis(
            "target"
        ).to_csv(out / "class_balance.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=frame, x=TARGET, ax=ax)
        ax.set_title("Target class balance")
        fig.tight_layout()
        fig.savefig(out / "class_balance.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(frame.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature correlation heatmap")
        fig.tight_layout()
        fig.savefig(out / "correlation_heatmap.png", dpi=160)
        plt.close(fig)
        for feature in FEATURES:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(data=frame, x=feature, hue=TARGET, kde=False, ax=ax)
            ax.set_title(f"Distribution: {feature}")
            fig.tight_layout()
            fig.savefig(out / f"hist_{feature}.png", dpi=140)
            plt.close(fig)
        logger.info("event=eda_exit rows=%d artifacts=%d output=%s",
                    len(frame), len(list(out.iterdir())), out)
    except Exception:
        logger.exception("event=eda_error input=%s", input_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/heart_disease.csv")
    parser.add_argument("--output", default="reports/eda")
    args = parser.parse_args()
    generate_eda(args.input, args.output)


if __name__ == "__main__":
    configure_logging()
    main()
