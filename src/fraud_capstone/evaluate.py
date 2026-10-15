"""Model evaluation, metrics calculation, and threshold selection."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_capstone.config import (
    CONFUSION_MATRIX_PLOT,
    METRICS_REPORT,
    THRESHOLD_FILE,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate PR-AUC, ROC-AUC, precision, recall, F1, and confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics safely
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    total = len(y_true)
    fraud_count = int(np.sum(y_true))
    fraud_prevalence = fraud_count / total if total > 0 else 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "total_samples": total,
        "fraud_count": fraud_count,
        "fraud_prevalence": fraud_prevalence,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Search for optimal probability threshold on validation set.

    Returns:
        (best_threshold, best_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_thresh = 0.5
    best_score = 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if metric == "f1":
            score = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        else:
            score = p
        if score > best_score:
            best_score = float(score)
            best_thresh = float(t)

    return best_thresh, best_score


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save confusion matrix heatmap plot."""
    target_path = Path(output_path) if output_path else CONFUSION_MATRIX_PLOT
    target_path.parent.mkdir(parents=True, exist_ok=True)

    cm_array = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm_array, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    for i in range(2):
        for j in range(2):
            ax.text(
                x=j,
                y=i,
                s=f"{cm_array[i, j]:,}",
                va="center",
                ha="center",
                size="large",
                weight="bold",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Legit (0)", "Pred Fraud (1)"])
    ax.set_yticklabels(["True Legit (0)", "True Fraud (1)"])
    ax.set_title("PaySim Confusion Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(target_path, dpi=200)
    plt.close()
    return target_path


def save_metrics_report(
    metrics: Dict[str, Any], output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Save evaluation metrics to JSON."""
    target_path = Path(output_path) if output_path else METRICS_REPORT
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return target_path


def save_decision_threshold(
    threshold: float, threshold_path: Optional[Union[str, Path]] = None
) -> Path:
    """Save selected decision threshold to JSON."""
    target_path = Path(threshold_path) if threshold_path else THRESHOLD_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(threshold)}, f, indent=2)
    return target_path


def load_decision_threshold(threshold_path: Optional[Union[str, Path]] = None) -> float:
    """Load selected decision threshold."""
    target_path = Path(threshold_path) if threshold_path else THRESHOLD_FILE
    if not target_path.exists():
        return 0.5
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("threshold", 0.5))
