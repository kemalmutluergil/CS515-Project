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
from test import evaluate


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


def train_model_full(
    model: torch.nn.Module,
    task: str,
    train_params: TrainParams,
    device: torch.device,
    data: Data | None = None,
    train_mask: Tensor | None = None,
    val_mask: Tensor | None = None,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, float]:
    """Train a GNN model fully, including DGI pre-training and standard training with early stopping.

    Args:
        model (torch.nn.Module): The GNN model to train.
        task (str): The classification task ("node" or "graph").
        train_params (TrainParams): Hyperparameters for training.
        device (torch.device): Device to place tensors and model on.
        data (Optional[Data]): A single graph Data object (node task only).
        train_mask (Optional[Tensor]): Boolean training mask (node task only).
        val_mask (Optional[Tensor]): Boolean validation mask (node task only).
        train_loader (Optional[DataLoader]): Mini-batch loader for training graphs (graph task only).
        val_loader (Optional[DataLoader]): Mini-batch loader for validating graphs (graph task only).
        verbose (bool): Whether to print epoch training logs (default is True).

    Returns:
        Tuple[torch.nn.Module, float]: The trained model with best validation weights and the best validation accuracy.
    """
    is_dgi = hasattr(model, "dgi")

    # ── Phase 1: DGI unsupervised pre-training ───────────────────────
    if is_dgi:
        if verbose:
            print("--- Phase 1: DGI Pre-training (unsupervised) ---")
        pretrain_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
        )
        for epoch in range(1, train_params.pretrain_epochs + 1):
            if task == "node":
                assert data is not None
                pt_loss = train_dgi_pretrain(model, data, pretrain_optimizer)
            else:
                assert train_loader is not None
                pt_loss = train_dgi_pretrain_graph(
                    model, train_loader, pretrain_optimizer, device
                )
            if verbose and (epoch % 50 == 0 or epoch == 1):
                print(f"Epoch {epoch:>4d}  |  Pre-train Loss: {pt_loss:.4f}")

        # Freeze encoder weights before linear evaluation
        for param in model.dgi.encoder.parameters():
            param.requires_grad = False
        model.dgi.weight.requires_grad = False

        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
        )
        if verbose:
            print("\n--- Phase 2: Linear Evaluation ---")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
        )

    # ── Training loop (Phase 2 for DGI; full training for others) ────
    best_val_acc = 0.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, train_params.epochs + 1):
        # Train
        loss = train_one_epoch(
            model=model,
            task=task,
            optimizer=optimizer,
            device=device,
            data=data,
            train_mask=train_mask,
            train_loader=train_loader,
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            task=task,
            device=device,
            data=data,
            mask=val_mask,
            loader=val_loader,
        )
        val_acc = val_metrics["accuracy"]

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:>4d}  |  "
                f"Train Loss: {loss:.4f}  |  "
                f"Val Acc: {val_acc:.4f}"
            )

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if train_params.patience > 0 and epochs_without_improvement >= train_params.patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (patience={train_params.patience}).")
            break

    # ── Restore best model ───────────────────────────────────────────
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val_acc
