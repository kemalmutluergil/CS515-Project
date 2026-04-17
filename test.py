"""Evaluation routines for node-level and graph-level classification.

This module provides a unified ``evaluate`` dispatcher as well as the
underlying task-specific helpers.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_node_classification(
    model: torch.nn.Module,
    data: Data,
    mask: Tensor,
) -> Dict[str, float]:
    """Evaluate a model on the given node mask.

    Args:
        model: The GNN model (set to eval mode internally).
        data: A single PyG :class:`Data` object containing the graph.
        mask: Boolean mask selecting the nodes to evaluate.

    Returns:
        A dictionary with ``"accuracy"`` and ``"loss"`` keys.
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=-1)
    correct = (pred == data.y[mask]).sum().item()
    total = mask.sum().item()
    loss = F.nll_loss(out[mask], data.y[mask]).item()
    return {"accuracy": correct / total, "loss": loss}


# ---------------------------------------------------------------------------
# Graph classification
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_graph_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a model on a graph-classification data loader.

    Args:
        model: The GNN model (set to eval mode internally).
        loader: A PyG :class:`DataLoader` yielding batched graphs.
        device: Device to place tensors on.

    Returns:
        A dictionary with ``"accuracy"`` and ``"loss"`` keys.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
        total_loss += F.nll_loss(out, batch.y).item() * batch.num_graphs

    return {"accuracy": correct / total, "loss": total_loss / total}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    task: str,
    device: torch.device,
    data: Data | None = None,
    mask: Tensor | None = None,
    loader: DataLoader | None = None,
) -> Dict[str, float]:
    """Dispatch evaluation based on the task type.

    Args:
        model: The GNN model.
        task: ``"node"`` or ``"graph"``.
        device: Device to place tensors on.
        data: The full graph (node classification only).
        mask: Boolean evaluation mask (node classification only).
        loader: DataLoader for batched graphs (graph classification only).

    Returns:
        A dictionary with ``"accuracy"`` and ``"loss"`` keys.

    Raises:
        ValueError: If *task* is not ``"node"`` or ``"graph"``.
    """
    if task == "node":
        assert data is not None and mask is not None
        return test_node_classification(model, data, mask)
    elif task == "graph":
        assert loader is not None
        return test_graph_classification(model, loader, device)
    else:
        raise ValueError(f"Unknown task type: {task}")
