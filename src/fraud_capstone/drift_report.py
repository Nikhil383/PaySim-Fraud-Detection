"""Evidently AI data and prediction drift monitoring."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from fraud_capstone.config import DRIFT_REPORT_FILE, RAW_DATA_PATH
from fraud_capstone.data import load_raw_data
from fraud_capstone.features import engineer_features
from fraud_capstone.split import chronological_split

logger = logging.getLogger(__name__)


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Generate an Evidently drift report comparing reference and current feature distributions."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        raise ImportError(
            "Evidently is not installed. Please install it using 'uv sync --extra monitoring'."
        )

    target_path = Path(output_path) if output_path else DRIFT_REPORT_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Engineer features for both datasets
    ref_features = engineer_features(reference_df)
    curr_features = engineer_features(current_df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=curr_features)
    report.save_html(str(target_path))

    logger.info(f"Evidently drift report generated and saved to {target_path}")
    return target_path


def main():
    parser = argparse.ArgumentParser(description="Generate Evidently Drift Report for PaySim")
    parser.add_argument("--data", type=str, default=str(RAW_DATA_PATH), help="Path to PaySim CSV")
    parser.add_argument("--output", type=str, default=str(DRIFT_REPORT_FILE), help="Output HTML file path")
    args = parser.parse_args()

    df = load_raw_data(file_path=args.data)
    # Use training split as reference and test split as current
    train_df, _, test_df = chronological_split(df)
    generate_drift_report(train_df, test_df, output_path=args.output)


if __name__ == "__main__":
    main()
