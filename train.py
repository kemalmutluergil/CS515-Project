"""Training routines for node-level and graph-level classification.

This module provides a unified ``train_one_epoch`` dispatcher as well as the
underlying task-specific helpers, plus DGI unsupervised pre-training functions.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from parameters import TrainParams


# ---------------------------------------------------------------------------
# DGI unsupervised pre-training
# ---------------------------------------------------------------------------

def train_dgi_pretrain(
    model: torch.nn.Module,
    data: Data,
    optimizer: Optimizer,
) -> float:
    """One unsupervised pre-training epoch for DGI on a single graph.

    Args:
        model: The DGI model.
        data: A single PyG :class:`Data` object (node classification dataset).
        optimizer: The optimizer instance.

    Returns:
        The pre-training loss for this epoch.
    """
    model.train()
    optimizer.zero_grad()
    loss = model.pretrain_loss(data.x, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_dgi_pretrain_graph(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """One unsupervised pre-training epoch for DGI on batched graphs.

    Args:
        model: The DGI model.
        loader: A PyG :class:`DataLoader` yielding batched graphs.
        optimizer: The optimizer instance.
        device: Device to place tensors on.

    Returns:
        The mean pre-training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.pretrain_loss(batch.x, batch.edge_index, batch.batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / total_graphs


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def train_node_classification(
    model: torch.nn.Module,
    data: Data,
    optimizer: Optimizer,
    train_mask: Tensor,
) -> float:
    """Run one training epoch for node classification (full-batch).

    Args:
        model: The GNN model.
        data: A single PyG :class:`Data` object containing the graph.
        optimizer: The optimizer instance.
        train_mask: Boolean mask selecting training nodes.

    Returns:
        The mean training loss for this epoch.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Graph classification
# ---------------------------------------------------------------------------

def train_graph_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch for graph classification (mini-batch).

    Args:
        model: The GNN model.
        loader: A PyG :class:`DataLoader` yielding batched graphs.
        optimizer: The optimizer instance.
        device: Device to place tensors on.

    Returns:
        The mean training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        num_graphs += batch.num_graphs

    return total_loss / num_graphs


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    task: str,
    optimizer: Optimizer,
    device: torch.device,
    data: Data | None = None,
    train_mask: Tensor | None = None,
    train_loader: DataLoader | None = None,
) -> float:
    """Dispatch a single training epoch based on the task type.

    Args:
        model: The GNN model.
        task: ``"node"`` or ``"graph"``.
        optimizer: The optimizer instance.
        device: Device to place tensors on.
        data: The full graph (node classification only).
        train_mask: Boolean training mask (node classification only).
        train_loader: DataLoader for batched graphs (graph classification only).

    Returns:
        The mean training loss for this epoch.

    Raises:
        ValueError: If *task* is not ``"node"`` or ``"graph"``.
    """
    if task == "node":
        assert data is not None and train_mask is not None
        return train_node_classification(model, data, optimizer, train_mask)
    elif task == "graph":
        assert train_loader is not None
        return train_graph_classification(model, train_loader, optimizer, device)
    else:
        raise ValueError(f"Unknown task type: {task}")
