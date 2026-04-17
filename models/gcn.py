"""Graph Convolutional Network (GCN) model.

Implements the architecture from Kipf & Welling (2017) with configurable
depth.  Supports both node-level and graph-level classification.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    """A multi-layer Graph Convolutional Network.

    The model stacks *num_layers* ``GCNConv`` layers with ReLU activation
    and dropout in between.  For graph-classification tasks an additional
    ``global_mean_pool`` readout followed by a linear classifier is used.

    Args:
        in_channels: Dimensionality of input node features.
        hidden_channels: Width of each hidden layer.
        out_channels: Number of output classes.
        num_layers: Total number of ``GCNConv`` layers (≥ 2).
        dropout: Dropout probability applied after each hidden layer.
        task: ``"node"`` for node classification or ``"graph"`` for
            graph classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        task: str = "node",
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        assert task in ("node", "graph"), f"Unknown task: {task}"

        self.task = task
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        if task == "graph":
            self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        else:
            self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the forward pass.

        Args:
            x: Node feature matrix of shape ``[num_nodes, in_channels]``.
            edge_index: Edge index tensor of shape ``[2, num_edges]``.
            batch: Batch vector of shape ``[num_nodes]`` mapping each node
                to its graph index.  Required when ``task="graph"``.

        Returns:
            Log-softmax prediction of shape ``[num_nodes, out_channels]``
            (node task) or ``[num_graphs, out_channels]`` (graph task).
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph":
            assert batch is not None, "batch tensor is required for graph classification"
            x = global_mean_pool(x, batch)

        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)
