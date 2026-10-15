"""GraphSAGE neural network model for PaySim fraud detection."""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
except ImportError:
    torch = None
    nn = object
    F = None
    SAGEConv = None


class GraphSAGEFraud(nn.Module if torch else object):
    """2-layer GraphSAGE architecture for binary fraud classification."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        if torch is None:
            raise ImportError(
                "PyTorch and PyTorch Geometric required. Install via 'uv sync --extra graph'."
            )
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x)
        return out
