"""Leakage-safe feature engineering module."""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fraud_capstone.config import (
    FEATURE_SCHEMA_FILE,
    PROHIBITED_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRANSACTION_TYPES,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from raw transaction data without target leakage.

    Features derived:
    - hour_of_day: step % 24
    - day_of_week: (step // 24) % 7
    - balance_orig_diff: (oldbalanceOrg - amount) - newbalanceOrig (error in origin balance)
    - balance_dest_diff: (oldbalanceDest + amount) - newbalanceDest (error in dest balance)
    - amount_to_orig_ratio: amount / (oldbalanceOrg + 1.0)
    - amount_to_dest_ratio: amount / (oldbalanceDest + 1.0)
    - is_zero_orig_balance: 1 if oldbalanceOrg == 0 else 0
    - is_zero_dest_balance: 1 if oldbalanceDest == 0 else 0
    - is_orig_dest_same: 1 if nameOrig == nameDest else 0
    - One-hot encoded transaction types (aligned across all types)
    """
    out = pd.DataFrame(index=df.index)

    # 1. Temporal features
    step = df["step"]
    out["step"] = step
    out["hour_of_day"] = step % 24
    out["day_of_week"] = (step // 24) % 7

    # 2. Transaction Amount
    out["amount"] = df["amount"].astype(float)
    out["log_amount"] = np.log1p(np.maximum(df["amount"], 0))

    # 3. Account balances
    old_orig = df["oldbalanceOrg"].astype(float)
    new_orig = df["newbalanceOrig"].astype(float)
    old_dest = df["oldbalanceDest"].astype(float)
    new_dest = df["newbalanceDest"].astype(float)

    out["oldbalanceOrg"] = old_orig
    out["newbalanceOrig"] = new_orig
    out["oldbalanceDest"] = old_dest
    out["newbalanceDest"] = new_dest

    # Delta & discrepancies
    out["orig_balance_change"] = new_orig - old_orig
    out["dest_balance_change"] = new_dest - old_dest
    out["orig_discrepancy"] = (old_orig - out["amount"]) - new_orig
    out["dest_discrepancy"] = (old_dest + out["amount"]) - new_dest

    # Ratios
    out["amount_to_orig_ratio"] = out["amount"] / (old_orig + 1.0)
    out["amount_to_dest_ratio"] = out["amount"] / (old_dest + 1.0)

    # Flags
    out["is_zero_orig_balance"] = (old_orig == 0).astype(int)
    out["is_zero_dest_balance"] = (old_dest == 0).astype(int)
    if "nameOrig" in df.columns and "nameDest" in df.columns:
        out["is_orig_dest_same"] = (df["nameOrig"] == df["nameDest"]).astype(int)
    else:
        out["is_orig_dest_same"] = 0

    # 4. One-hot encode transaction types deterministically
    for t_type in TRANSACTION_TYPES:
        out[f"type_{t_type}"] = (df["type"] == t_type).astype(int)

    return out


def get_feature_names(engineered_df: pd.DataFrame) -> List[str]:
    """Extract list of feature column names, verifying no prohibited columns are present."""
    cols = list(engineered_df.columns)
    for prohibited in PROHIBITED_FEATURE_COLUMNS:
        if prohibited in cols:
            raise ValueError(f"Prohibited column '{prohibited}' leaked into model features!")
    return cols


def save_feature_schema(
    feature_names: List[str], schema_path: Optional[Union[str, Path]] = None
) -> Path:
    """Save the ordered feature schema as JSON."""
    target_path = Path(schema_path) if schema_path else FEATURE_SCHEMA_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump({"features": feature_names, "count": len(feature_names)}, f, indent=2)
    return target_path


def load_feature_schema(schema_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Load the ordered list of feature column names."""
    target_path = Path(schema_path) if schema_path else FEATURE_SCHEMA_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Feature schema not found at '{target_path}'.")
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["features"]


def align_features(df_features: pd.DataFrame, expected_schema: List[str]) -> pd.DataFrame:
    """Ensure dataframe matches the expected feature schema in exact column order."""
    for col in expected_schema:
        if col not in df_features.columns:
            df_features[col] = 0
    return df_features[expected_schema].copy()
