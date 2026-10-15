"""Exploratory Data Analysis utilities and artifact generation."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_capstone.config import (
    CLASS_BALANCE_PLOT,
    FRAUD_BY_TYPE_CSV,
    REPORTS_DIR,
    TARGET_COLUMN,
)


def compute_class_imbalance(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate fraud and legitimate transaction counts and proportions."""
    total = len(df)
    fraud_count = int((df[TARGET_COLUMN] == 1).sum())
    legit_count = int((df[TARGET_COLUMN] == 0).sum())
    fraud_rate = fraud_count / total if total > 0 else 0.0

    return {
        "total_transactions": total,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_prevalence": fraud_rate,
        "imbalance_ratio": (legit_count / fraud_count) if fraud_count > 0 else float("inf"),
    }


def compute_fraud_by_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze fraud counts, total volume, and fraud rates across transaction types."""
    grouped = (
        df.groupby("type")
        .agg(
            total_count=("step", "count"),
            fraud_count=(TARGET_COLUMN, lambda x: (x == 1).sum()),
            total_amount=("amount", "sum"),
            fraud_amount=("amount", lambda x: x[df.loc[x.index, TARGET_COLUMN] == 1].sum()),
        )
        .reset_index()
    )
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["total_count"]
    grouped["fraud_rate_pct"] = grouped["fraud_rate"] * 100.0
    return grouped


def plot_class_balance(
    df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Save a bar chart illustrating fraud vs legitimate class imbalance."""
    target_path = Path(output_path) if output_path else CLASS_BALANCE_PLOT
    target_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df[TARGET_COLUMN].value_counts()
    labels = ["Legitimate (0)", "Fraudulent (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#2b5c8f", "#d9534f"])
    ax.set_ylabel("Transaction Count")
    ax.set_title("PaySim Transaction Class Distribution")
    ax.set_yscale("log")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(target_path, dpi=200)
    plt.close()
    return target_path


def generate_eda_reports(
    df: pd.DataFrame, reports_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Generate all EDA artifacts: summary statistics, CSVs, and plots."""
    out_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    imbalance_stats = compute_class_imbalance(df)
    fraud_by_type = compute_fraud_by_transaction_type(df)
    csv_path = out_dir / FRAUD_BY_TYPE_CSV.name
    fraud_by_type.to_csv(csv_path, index=False)

    plot_path = out_dir / CLASS_BALANCE_PLOT.name
    plot_class_balance(df, plot_path)

    return {
        "imbalance_stats": imbalance_stats,
        "fraud_by_type_csv": str(csv_path),
        "class_balance_plot": str(plot_path),
    }
