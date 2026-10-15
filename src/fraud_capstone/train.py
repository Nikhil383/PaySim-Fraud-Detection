"""XGBoost training pipeline with leakage prevention, validation tuning, and MLflow logging."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_capstone.config import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RANDOM_STATE,
    MODEL_FILE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from fraud_capstone.data import load_raw_data, save_data_quality_report, validate_dataset
from fraud_capstone.eda import generate_eda_reports
from fraud_capstone.evaluate import (
    evaluate_predictions,
    find_optimal_threshold,
    plot_confusion_matrix,
    save_decision_threshold,
    save_metrics_report,
)
from fraud_capstone.features import (
    align_features,
    engineer_features,
    get_feature_names,
    save_feature_schema,
)
from fraud_capstone.split import chronological_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline(
    data_path: Optional[Path] = None,
    nrows: Optional[int] = None,
    tracking: bool = False,
    experiment_name: str = "paysim-fraud-detection",
) -> Dict[str, Any]:
    """Execute the end-to-end training pipeline.

    1. Load and validate data
    2. Generate data quality & EDA reports
    3. Chronological split (70% train, 15% val, 15% test)
    4. Leakage-safe feature extraction & schema save
    5. Train XGBoost classifier with scale_pos_weight
    6. Select threshold on validation set
    7. Evaluate on held-out test set
    8. Save model, threshold, schema, and evaluation metrics
    9. Optional MLflow logging
    """
    logger.info("Starting PaySim Fraud Detection Pipeline...")
    df = load_raw_data(file_path=data_path, nrows=nrows)
    logger.info(f"Loaded dataset with {len(df):,} rows.")

    # 1. Validation & Data Quality
    quality_summary = validate_dataset(df)
    save_data_quality_report(quality_summary)
    logger.info("Saved data quality report.")

    # 2. EDA reports
    generate_eda_reports(df)
    logger.info("Generated EDA reports and figures.")

    # 3. Chronological Split
    train_df, val_df, test_df = chronological_split(df)
    logger.info(
        f"Data split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
    )

    # 4. Feature Engineering
    X_train_raw = engineer_features(train_df)
    feature_names = get_feature_names(X_train_raw)
    save_feature_schema(feature_names)
    logger.info(f"Engineered {len(feature_names)} features and saved schema.")

    y_train = train_df[TARGET_COLUMN].values
    X_train = align_features(X_train_raw, feature_names)

    X_val_raw = engineer_features(val_df)
    X_val = align_features(X_val_raw, feature_names)
    y_val = val_df[TARGET_COLUMN].values

    X_test_raw = engineer_features(test_df)
    X_test = align_features(X_test_raw, feature_names)
    y_test = test_df[TARGET_COLUMN].values

    # 5. Class weighting for extreme imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Train XGBoost
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": DEFAULT_RANDOM_STATE,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("Trained XGBoost baseline model.")

    # 6. Threshold selection on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh, val_score = find_optimal_threshold(y_val, val_probs, metric="f1")
    logger.info(f"Selected validation threshold: {best_thresh:.4f} (Val F1: {val_score:.4f})")
    save_decision_threshold(best_thresh)

    # 7. Evaluate on held-out test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_probs, threshold=best_thresh)
    save_metrics_report(test_metrics)
    plot_confusion_matrix(test_metrics["confusion_matrix"])
    logger.info(
        f"Test Results -> PR-AUC: {test_metrics['pr_auc']:.4f}, "
        f"Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}"
    )

    # 8. Save Model
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    logger.info(f"Saved trained model to {MODEL_FILE}")

    # 9. MLflow Tracking (Optional)
    if tracking:
        try:
            import mlflow
            import mlflow.xgboost

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="xgboost_baseline"):
                mlflow.log_params(xgb_params)
                mlflow.log_param("threshold", best_thresh)
                mlflow.log_metrics({
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "val_f1": val_score,
                })
                mlflow.xgboost.log_model(model, "xgboost_model")
                logger.info("Logged run and artifacts to MLflow.")
        except ImportError:
            logger.warning(
                "MLflow is not installed. To log runs, install via 'uv sync --extra tracking'."
            )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    return {
        "model": model,
        "feature_names": feature_names,
        "threshold": best_thresh,
        "metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PaySim XGBoost Fraud Detection Model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Path to PaySim CSV dataset",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of rows for testing/quick iteration",
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    args = parser.parse_args()

    train_pipeline(
        data_path=Path(args.data),
        nrows=args.nrows,
        tracking=args.tracking,
    )


if __name__ == "__main__":
    main()
