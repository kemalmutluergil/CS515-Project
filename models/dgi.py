"""Deep Graph Infomax (DGI) — unsupervised node/graph representation learning.

Two-phase usage:
  Phase 1 (pre-training, no labels):
    node datasets : model.pretrain_loss(x, edge_index)
    graph datasets: model.pretrain_loss(x, edge_index, batch)
  Phase 2 (linear evaluation, labels used):
    model.forward(x, edge_index[, batch])  — encoder frozen, only classifier trains
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import DeepGraphInfomax, GCNConv, global_mean_pool


class _Encoder(torch.nn.Module):
    """Multi-layer GCN encoder shared across both training phases."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def _corruption(x: Tensor, edge_index: Tensor):
    """Shuffle all node features — negative sample for node-task pre-training."""
    return x[torch.randperm(x.size(0), device=x.device)], edge_index


def _shuffle_within_graphs(x: Tensor, batch: Tensor) -> Tensor:
    """Shuffle node features within each graph independently (graph-task)."""
    perm = torch.arange(x.size(0), device=x.device)
    for g in range(batch.max().item() + 1):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        perm[idx] = idx[torch.randperm(len(idx), device=x.device)]
    return x[perm]


class DGI(torch.nn.Module):
    """Deep Graph Infomax model.

    Learns node/graph representations without supervision by maximising mutual
    information between local node embeddings and a global graph summary.

    Args:
        in_channels: Dimensionality of input node features.
        hidden_channels: Width of each hidden layer.
        out_channels: Number of output classes (for the downstream linear head).
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

        encoder = _Encoder(in_channels, hidden_channels, num_layers, dropout)

        self.dgi = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder=encoder,
            summary=lambda z, *_args, **_kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=_corruption,
        )

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    # ── Phase 1 ────────────────────────────────────────────────────────────

    def pretrain_loss(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the DGI mutual-information loss (no labels required).

        Args:
            x: Node feature matrix ``[num_nodes, in_channels]``.
            edge_index: Edge index ``[2, num_edges]``.
            batch: Node-to-graph assignment ``[num_nodes]``.
                Pass ``None`` for node-classification datasets (single graph).

        Returns:
            Scalar pre-training loss.
        """
        if batch is None:
            # Node task: use PyG's DeepGraphInfomax directly
            pos_z, neg_z, summary = self.dgi(x, edge_index)
            return self.dgi.loss(pos_z, neg_z, summary)

        # Graph task: batch-aware variant
        EPS = 1e-15
        pos_z = self.dgi.encoder(x, edge_index)
        neg_x = _shuffle_within_graphs(x, batch)
        neg_z = self.dgi.encoder(neg_x, edge_index)

        # Per-graph summary: [num_graphs, H]
        summary = torch.sigmoid(global_mean_pool(pos_z, batch))
        # Expand to each node's own graph summary: [total_nodes, H]
        node_summary = summary[batch]

        # Bilinear score: z_i^T · W · s_i  = dot(z_i, W^T s_i)
        pos_score = torch.sigmoid(
            (pos_z * (node_summary @ self.dgi.weight.t())).sum(dim=-1)
        )
        neg_score = torch.sigmoid(
            (neg_z * (node_summary @ self.dgi.weight.t())).sum(dim=-1)
        )
        return (-torch.log(pos_score + EPS) - torch.log(1 - neg_score + EPS)).mean()

    # ── Phase 2 ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Downstream classification forward pass (encoder is frozen in Phase 2).

        Args:
            x: Node feature matrix ``[num_nodes, in_channels]``.
            edge_index: Edge index ``[2, num_edges]``.
            batch: Node-to-graph assignment (graph task only).

        Returns:
            Log-softmax predictions ``[num_nodes, out_channels]`` (node task)
            or ``[num_graphs, out_channels]`` (graph task).
        """
        z = self.dgi.encoder(x, edge_index)
        if self.task == "graph":
            assert batch is not None, "batch tensor required for graph task"
            z = global_mean_pool(z, batch)
        return F.log_softmax(self.classifier(z), dim=-1)

    @torch.no_grad()
    def embed(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Return encoder embeddings (no dropout).

        Args:
            x: Node feature matrix ``[num_nodes, in_channels]``.
            edge_index: Edge index tensor ``[2, num_edges]``.
            batch: Batch vector (graph task only).

        Returns:
            ``[num_nodes, hidden_channels]`` (node) or
            ``[num_graphs, hidden_channels]`` (graph).
        """
        self.eval()
        z = self.dgi.encoder(x, edge_index)
        if self.task == "graph":
            assert batch is not None
            z = global_mean_pool(z, batch)
        return z

    @torch.no_grad()
    def anomaly_score(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """DGI discriminator-based anomaly score (higher = more anomalous).

        Nodes/graphs that poorly match the global summary receive a high score.

        Args:
            x: Node feature matrix ``[num_nodes, in_channels]``.
            edge_index: Edge index tensor ``[2, num_edges]``.
            batch: Batch vector (graph task only).

        Returns:
            Scores ``[num_nodes]`` (node task) or ``[num_graphs]`` (graph task).
        """
        self.eval()
        z = self.dgi.encoder(x, edge_index)

        if batch is None:
            summary = torch.sigmoid(z.mean(dim=0))
            scores = self.dgi.discriminate(z, summary, sigmoid=True)
        else:
            summary = torch.sigmoid(global_mean_pool(z, batch))
            node_summary = summary[batch]
            node_scores = torch.sigmoid(
                (z * (node_summary @ self.dgi.weight.t())).sum(dim=-1)
            )
            scores = global_mean_pool(node_scores.unsqueeze(-1), batch).squeeze(-1)

        return 1.0 - scores
