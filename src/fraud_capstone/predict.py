"""Inference and scoring module for PaySim fraud detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from fraud_capstone.config import (
    FEATURE_SCHEMA_FILE,
    MODEL_FILE,
    THRESHOLD_FILE,
)
from fraud_capstone.evaluate import load_decision_threshold
from fraud_capstone.features import align_features, engineer_features, load_feature_schema


class FraudPredictor:
    """Predictor encapsulating trained model, feature schema, and decision threshold."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        threshold_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else MODEL_FILE
        self.schema_path = Path(schema_path) if schema_path else FEATURE_SCHEMA_FILE
        self.threshold_path = Path(threshold_path) if threshold_path else THRESHOLD_FILE

        self.model = None
        self.feature_schema = []
        self.threshold = 0.5
        self._load_artifacts()

    def _load_artifacts(self):
        """Load artifacts from disk.

        Raises:
            FileNotFoundError: If model or feature schema is missing.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Please train the model first using 'python -m fraud_capstone.train'."
            )
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Feature schema not found at '{self.schema_path}'."
            )

        self.model = joblib.load(self.model_path)
        self.feature_schema = load_feature_schema(self.schema_path)
        self.threshold = load_decision_threshold(self.threshold_path)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch dataframe of raw transactions."""
        X_eng = engineer_features(df)
        X_aligned = align_features(X_eng, self.feature_schema)
        probabilities = self.model.predict_proba(X_aligned)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        results = df.copy()
        results["fraud_probability"] = probabilities
        results["is_fraud_prediction"] = predictions
        results["decision_threshold"] = self.threshold
        return results

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single transaction record."""
        df = pd.DataFrame([transaction])
        scored_df = self.predict_dataframe(df)
        row = scored_df.iloc[0]

        return {
            "fraud_probability": float(row["fraud_probability"]),
            "is_fraud_prediction": int(row["is_fraud_prediction"]),
            "decision_threshold": float(row["decision_threshold"]),
        }
