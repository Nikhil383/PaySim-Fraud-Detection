"""Temporal transaction graph construction for Graph Neural Networks."""

from typing import Tuple
import numpy as np
import pandas as pd

from fraud_capstone.config import TARGET_COLUMN
from fraud_capstone.features import engineer_features, get_feature_names


def build_temporal_graph(df: pd.DataFrame):
    """Build a temporal PyTorch Geometric Data graph from PaySim transactions.

    Nodes represent individual transactions.
    Directed edges connect earlier transactions to later transactions sharing
    an origin or destination account (chronological directed acyclic graph).
    Labels (isFraud) are strictly isolated as target y and excluded from node features x.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch or PyTorch Geometric is missing. Install via 'uv sync --extra graph'."
        )

    # 1. Sort strictly chronologically by step
    sorted_df = df.sort_values(by="step", kind="mergesort").reset_index(drop=True)

    # 2. Extract node features without leakage
    X_features = engineer_features(sorted_df)
    feature_names = get_feature_names(X_features)
    x = torch.tensor(X_features[feature_names].values, dtype=torch.float)

    # Target labels
    y = torch.tensor(sorted_df[TARGET_COLUMN].values, dtype=torch.long)

    # 3. Construct chronological edges
    # Directed edge (u -> v) only if u < v (earlier to later) and u and v share an account
    edge_sources = []
    edge_targets = []

    account_to_nodes = {}
    for idx, row in sorted_df.iterrows():
        orig = row["nameOrig"]
        dest = row["nameDest"]

        for acc in (orig, dest):
            if acc in account_to_nodes:
                # Add edge from previous transactions on this account to current transaction
                for prev_node in account_to_nodes[acc]:
                    edge_sources.append(prev_node)
                    edge_targets.append(idx)
                account_to_nodes[acc].append(idx)
            else:
                account_to_nodes[acc] = [idx]

    if edge_sources:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 4. Chronological boolean masks
    n = len(sorted_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data, sorted_df
