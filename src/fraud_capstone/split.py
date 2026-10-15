"""Chronological data splitting module to prevent temporal leakage."""

from typing import Tuple
import pandas as pd

from fraud_capstone.config import (
    SPLIT_TEST_RATIO,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = SPLIT_TRAIN_RATIO,
    val_ratio: float = SPLIT_VAL_RATIO,
    test_ratio: float = SPLIT_TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically by step.

    Ties are preserved in their stable order.
    Total ratios must sum to approximately 1.0.

    Returns:
        (train_df, val_df, test_df)
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5:
        raise ValueError("Ratios for train, val, and test must sum to 1.0.")

    # Sort strictly chronologically by step, stable sort for ties
    sorted_df = df.sort_values(by="step", kind="mergesort").reset_index(drop=True)

    n = len(sorted_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    return train_df, val_df, test_df
