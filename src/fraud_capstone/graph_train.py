"""Training and evaluation pipeline for Graph Neural Network."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from fraud_capstone.config import MODELS_DIR, RAW_DATA_PATH, REPORTS_DIR
from fraud_capstone.data import load_raw_data
from fraud_capstone.evaluate import evaluate_predictions, find_optimal_threshold
from fraud_capstone.graph_data import build_temporal_graph
from fraud_capstone.graph_model import GraphSAGEFraud

logger = logging.getLogger(__name__)


def train_graph_model(
    data_path: Path,
    sample_size: int = 10000,
    epochs: int = 20,
    lr: float = 0.01,
):
    """Train GraphSAGE on a chronological sample of PaySim transactions."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError(
            "PyTorch and PyTorch Geometric required. Install via 'uv sync --extra graph'."
        )

    logger.info(f"Loading {sample_size} transactions for GraphSAGE...")
    df = load_raw_data(data_path, nrows=sample_size)
    data, _ = build_temporal_graph(df)

    # Class weights for CrossEntropyLoss
    train_y = data.y[data.train_mask]
    n_neg = (train_y == 0).sum().item()
    n_pos = (train_y == 1).sum().item()
    weight = torch.tensor([1.0, (n_neg / n_pos) if n_pos > 0 else 1.0], dtype=torch.float)

    model = GraphSAGEFraud(in_channels=data.x.size(1), hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_pr_auc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        optimizer.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_probs = F.softmax(val_out, dim=-1)[data.val_mask, 1].cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()
            val_metrics = evaluate_predictions(val_y, val_probs)

            if val_metrics["pr_auc"] >= best_val_pr_auc:
                best_val_pr_auc = val_metrics["pr_auc"]
                best_weights = model.state_dict()

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {loss.item():.4f} | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | Val F1: {val_metrics['f1']:.4f}"
            )

    # Load best checkpoint and evaluate on test mask
    if best_weights:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_probs = F.softmax(test_out, dim=-1)[data.test_mask, 1].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()

        # Find best threshold on val
        val_probs = F.softmax(model(data.x, data.edge_index), dim=-1)[data.val_mask, 1].cpu().numpy()
        val_thresh, _ = find_optimal_threshold(data.y[data.val_mask].cpu().numpy(), val_probs)

        test_metrics = evaluate_predictions(test_y, test_probs, threshold=val_thresh)

    logger.info(
        f"GNN Test Results -> PR-AUC: {test_metrics['pr_auc']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}, Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}"
    )

    # Save model weights & metrics
    save_path = MODELS_DIR / "graphsage_fraud.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    metrics_path = REPORTS_DIR / "gnn_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE on PaySim")
    parser.add_argument("--data", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    train_graph_model(Path(args.data), sample_size=args.sample, epochs=args.epochs)


if __name__ == "__main__":
    main()
