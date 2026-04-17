"""Entry point for the graph learning pipeline.

Usage examples::

    python main.py --model gcn --dataset cora --epochs 200 --lr 0.01
    python main.py --model graphsage --dataset enzymes --epochs 100 --batch_size 32
    python main.py --model gcn --dataset karate --epochs 200
    python main.py --model graphsage --dataset proteins --epochs 100

Run ``python main.py --help`` for the full list of arguments.
"""

import argparse
import random
from typing import Any, Dict, Tuple

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.datasets import KarateClub, Planetoid, TUDataset
from torch_geometric.loader import DataLoader

# PyTorch ≥ 2.6 defaults to weights_only=True in torch.load, which rejects
# PyG's custom classes during dataset deserialization.  Register them here.
torch.serialization.add_safe_globals([
    Data,
    DataEdgeAttr,
    DataTensorAttr,
    GlobalStorage,
])

from models import GCN, GraphSAGE
from parameters import DataParams, ModelParams, TrainParams
from test import evaluate
from train import train_one_epoch


# ── Argument parsing ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return a :class:`~argparse.Namespace`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNN models on graph benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn", "graphsage"],
        help="GNN architecture to use.",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Width of hidden layers.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of message-passing layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability.",
    )

    # Training
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="L2 regularization."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Maximum training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Mini-batch size (graph tasks)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (0 = disabled).",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "karate", "enzymes", "proteins"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Dataset cache directory."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (graph / Karate splits).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (graph / Karate splits).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def build_params(
    args: argparse.Namespace,
) -> Tuple[ModelParams, TrainParams, DataParams]:
    """Construct typed dataclass instances from raw CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple of ``(ModelParams, TrainParams, DataParams)``.
    """
    model_params = ModelParams(
        model_name=args.model,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_params = TrainParams(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    data_params = DataParams(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    return model_params, train_params, data_params


# ── Seed ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset loading ──────────────────────────────────────────────────────

def _make_karate_masks(
    data: Data,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random train / val / test node masks for the Karate Club graph.

    Args:
        data: The Karate Club :class:`Data` object.
        train_ratio: Fraction of nodes for training.
        val_ratio: Fraction of nodes for validation.

    Returns:
        A tuple ``(train_mask, val_mask, test_mask)`` of boolean tensors.
    """
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train : n_train + n_val]] = True
    test_mask[indices[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


def load_dataset(
    data_params: DataParams,
) -> Dict[str, Any]:
    """Load a dataset and prepare task-specific data structures.

    Args:
        data_params: Data-related parameters.

    Returns:
        A dictionary with the following keys:

        * ``"task"`` — ``"node"`` or ``"graph"``.
        * ``"num_features"`` — Input feature dimensionality.
        * ``"num_classes"`` — Number of target classes.

        For **node classification** tasks:

        * ``"data"`` — The single graph :class:`Data` object.
        * ``"train_mask"``, ``"val_mask"``, ``"test_mask"`` — Boolean masks.

        For **graph classification** tasks:

        * ``"train_loader"``, ``"val_loader"``, ``"test_loader"`` —
          :class:`DataLoader` instances.
    """
    name = data_params.dataset_name.lower()

    if name == "cora":
        dataset = Planetoid(root=data_params.data_root, name="Cora")
        data = dataset[0]
        return {
            "task": "node",
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "data": data,
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
        }

    elif name == "karate":
        dataset = KarateClub()
        data = dataset[0]
        train_mask, val_mask, test_mask = _make_karate_masks(
            data,
            train_ratio=data_params.train_ratio,
            val_ratio=data_params.val_ratio,
        )
        return {
            "task": "node",
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "data": data,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

    elif name in ("enzymes", "proteins"):
        dataset = TUDataset(
            root=data_params.data_root,
            name=name.upper(),
            use_node_attr=True,
        )

        # Random train / val / test split at the graph level
        num_graphs = len(dataset)
        indices = torch.randperm(num_graphs)
        n_train = int(num_graphs * data_params.train_ratio)
        n_val = int(num_graphs * data_params.val_ratio)

        train_dataset = dataset[indices[:n_train]]
        val_dataset = dataset[indices[n_train : n_train + n_val]]
        test_dataset = dataset[indices[n_train + n_val :]]

        return {
            "task": "graph",
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
        }

    else:
        raise ValueError(f"Unknown dataset: {name}")


# ── Model factory ────────────────────────────────────────────────────────

def build_model(
    model_params: ModelParams,
    num_features: int,
    num_classes: int,
    task: str,
) -> torch.nn.Module:
    """Instantiate a GNN model from parameters.

    Args:
        model_params: Architecture parameters.
        num_features: Dimensionality of input node features.
        num_classes: Number of target classes.
        task: ``"node"`` or ``"graph"``.

    Returns:
        An initialised :class:`torch.nn.Module`.

    Raises:
        ValueError: If ``model_params.model_name`` is unknown.
    """
    kwargs = dict(
        in_channels=num_features,
        hidden_channels=model_params.hidden_channels,
        out_channels=num_classes,
        num_layers=model_params.num_layers,
        dropout=model_params.dropout,
        task=task,
    )

    if model_params.model_name == "gcn":
        return GCN(**kwargs)
    elif model_params.model_name == "graphsage":
        return GraphSAGE(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_params.model_name}")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full training and evaluation pipeline."""
    args = parse_args()
    model_params, train_params, data_params = build_params(args)

    set_seed(data_params.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ────────────────────────────────────────────────────
    ds_info = load_dataset(data_params)
    task: str = ds_info["task"]
    num_features: int = ds_info["num_features"]
    num_classes: int = ds_info["num_classes"]

    print(f"Dataset : {data_params.dataset_name}")
    print(f"Task    : {task} classification")
    print(f"Features: {num_features}  |  Classes: {num_classes}")

    # Prepare data / loaders depending on task
    data: Data | None = None
    train_mask = val_mask = test_mask = None
    train_loader = val_loader = test_loader = None

    if task == "node":
        data = ds_info["data"].to(device)
        train_mask = ds_info["train_mask"].to(device)
        val_mask = ds_info["val_mask"].to(device)
        test_mask = ds_info["test_mask"].to(device)
    else:
        train_loader = DataLoader(
            ds_info["train_dataset"],
            batch_size=train_params.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_info["val_dataset"],
            batch_size=train_params.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            ds_info["test_dataset"],
            batch_size=train_params.batch_size,
            shuffle=False,
        )

    # ── Build model ──────────────────────────────────────────────────
    model = build_model(model_params, num_features, num_classes, task).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params.lr,
        weight_decay=train_params.weight_decay,
    )

    print(f"\nModel   : {model_params.model_name.upper()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {total_params:,}")
    print(f"Device  : {device}\n")

    # ── Training loop ────────────────────────────────────────────────
    best_val_acc = 0.0
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

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4d}  |  "
                f"Train Loss: {loss:.4f}  |  "
                f"Val Acc: {val_acc:.4f}"
            )

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if train_params.patience > 0 and epochs_without_improvement >= train_params.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={train_params.patience}).")
            break

    # ── Restore best model & test ────────────────────────────────────
    model.load_state_dict(best_state)

    test_metrics = evaluate(
        model=model,
        task=task,
        device=device,
        data=data,
        mask=test_mask,
        loader=test_loader,
    )

    print("\n" + "=" * 50)
    print(f"Best Val Accuracy  : {best_val_acc:.4f}")
    print(f"Test Accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"Test Loss          : {test_metrics['loss']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
