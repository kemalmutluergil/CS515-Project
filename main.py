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

from models import GCN, GraphSAGE, DGI, GIN
from parameters import (
    DataParams,
    ModelParams,
    TrainParams,
    AnomalyParams,
    ExplainParams,
    OversmoothingParams,
    SweepParams,
    RobustnessParams,
    TopologyParams,
)
from test import (
    evaluate,
    run_anomaly_analysis,
    run_explainability_analysis,
    run_oversmoothing_analysis,
    run_robustness_analysis,
    run_topology_analysis,
)
from train import train_model_full


# ── Argument parsing ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return a :class:`~argparse.Namespace`.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Graph learning pipeline.")

    # ── Pipeline Mode ────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "anomaly", "explain", "oversmoothing", "sweep", "robustness", "topology"],
        help="Pipeline execution mode.",
    )

    # ── Model configuration ──────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn", "graphsage", "dgi", "gin"],
        help="GNN model architecture.",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Dimension of hidden embeddings.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of GNN conv layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability.",
    )

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 penalty).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Max number of downstream training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (graph task only).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (0 to disable).",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=100,
        help="Number of pre-training epochs (DGI only).",
    )

    # ── Dataset parameters ───────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer", "pubmed", "karate", "enzymes", "proteins"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for saving datasets.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training split ratio (graph / Karate splits).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (graph / Karate splits).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # ── Anomaly Detection parameters ─────────────────────────────────────────
    parser.add_argument(
        "--anomaly_contamination",
        type=float,
        default=0.1,
        help="Contamination parameter for Isolation Forest.",
    )
    parser.add_argument(
        "--anomaly_top_k",
        type=int,
        default=10,
        help="Number of top anomalous items to print.",
    )

    # ── Explainability parameters ────────────────────────────────────────────
    parser.add_argument(
        "--explain_epochs",
        type=int,
        default=100,
        help="Number of epochs to train GNNExplainer masks.",
    )

    # ── Oversmoothing parameters ─────────────────────────────────────────────
    parser.add_argument(
        "--oversmoothing_depths",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="List of model depths to analyze for oversmoothing.",
    )

    # ── Parameter Sweep parameters ───────────────────────────────────────────
    parser.add_argument(
        "--sweep_param",
        type=str,
        default="lr",
        choices=["lr", "hidden_channels", "num_layers", "dropout", "pretrain_epochs"],
        help="The hyperparameter to sweep over.",
    )
    parser.add_argument(
        "--sweep_values",
        type=str,
        nargs="+",
        default=["0.0001", "0.001", "0.01", "0.1"],
        help="Values to sweep over.",
    )

    # ── Robustness parameters ────────────────────────────────────────────────
    parser.add_argument(
        "--robustness_pert_type",
        type=str,
        default="edge_drop",
        choices=["edge_drop", "edge_add", "feature_noise"],
        help="Perturbation type for robustness analysis.",
    )
    parser.add_argument(
        "--robustness_pert_values",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4],
        help="Proportions or noise levels to sweep.",
    )

    # ── Topology parameters ──────────────────────────────────────────────────
    parser.add_argument(
        "--topology_degree_bins",
        type=str,
        nargs="+",
        default=["1", "2", "3-5", "6-10", "11+"],
        help="Bins for degree grouping in topology analysis.",
    )
    parser.add_argument(
        "--topology_homophily_bins",
        type=str,
        nargs="+",
        default=["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        help="Bins for homophily grouping in topology analysis.",
    )

    return parser.parse_args()


def build_params(
    args: argparse.Namespace,
) -> Tuple[
    ModelParams,
    TrainParams,
    DataParams,
    AnomalyParams,
    ExplainParams,
    OversmoothingParams,
    SweepParams,
    RobustnessParams,
    TopologyParams,
]:
    """Construct typed dataclass instances from raw CLI arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple: A tuple of all configuration parameter dataclasses.
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
        pretrain_epochs=args.pretrain_epochs,
    )
    data_params = DataParams(
        dataset_name=args.dataset,
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    anomaly_params = AnomalyParams(
        contamination=args.anomaly_contamination,
        top_k=args.anomaly_top_k,
    )
    explain_params = ExplainParams(
        epochs=args.explain_epochs,
    )
    oversmoothing_params = OversmoothingParams(
        depths=args.oversmoothing_depths,
    )

    raw_values = args.sweep_values
    parsed_values = []
    for val in raw_values:
        if args.sweep_param in ["lr", "dropout"]:
            parsed_values.append(float(val))
        elif args.sweep_param in ["hidden_channels", "num_layers", "pretrain_epochs"]:
            parsed_values.append(int(val))
        else:
            parsed_values.append(val)

    sweep_params = SweepParams(
        param_name=args.sweep_param,
        values=parsed_values,
    )
    robustness_params = RobustnessParams(
        pert_type=args.robustness_pert_type,
        pert_values=args.robustness_pert_values,
    )
    topology_params = TopologyParams(
        degree_bins=args.topology_degree_bins,
        homophily_bins=args.topology_homophily_bins,
    )

    return (
        model_params,
        train_params,
        data_params,
        anomaly_params,
        explain_params,
        oversmoothing_params,
        sweep_params,
        robustness_params,
        topology_params,
    )


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
    elif model_params.model_name == "dgi":
        return DGI(**kwargs)
    elif model_params.model_name == "gin":
        return GIN(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_params.model_name}")


# ── Main ─────────────────────────────────────────────────────────────────

def run_sweep(
    sweep_params: SweepParams,
    model_params: ModelParams,
    train_params: TrainParams,
    data_params: DataParams,
    device: torch.device,
) -> None:
    """Run a hyperparameter sweep over a chosen parameter and print results.

    Args:
        sweep_params (SweepParams): The sweep settings.
        model_params (ModelParams): Base model parameters.
        train_params (TrainParams): Base training parameters.
        data_params (DataParams): Dataset parameters.
        device (torch.device): Device to run on.
    """
    print(f"\n--- Hyperparameter Sweep for {model_params.model_name.upper()} ---")
    print(f"Sweeping parameter: {sweep_params.param_name}")
    print(f"{sweep_params.param_name:<15} | {'Best Val Acc':<15} | {'Test Acc':<15}")
    print("-" * 50)

    for val in sweep_params.values:
        m_params = ModelParams(
            model_name=model_params.model_name,
            hidden_channels=model_params.hidden_channels,
            num_layers=model_params.num_layers,
            dropout=model_params.dropout,
        )
        t_params = TrainParams(
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
            epochs=train_params.epochs,
            batch_size=train_params.batch_size,
            patience=train_params.patience,
            pretrain_epochs=train_params.pretrain_epochs,
        )

        if sweep_params.param_name == "lr":
            t_params.lr = val
        elif sweep_params.param_name == "weight_decay":
            t_params.weight_decay = val
        elif sweep_params.param_name == "epochs":
            t_params.epochs = val
        elif sweep_params.param_name == "batch_size":
            t_params.batch_size = val
        elif sweep_params.param_name == "patience":
            t_params.patience = val
        elif sweep_params.param_name == "pretrain_epochs":
            t_params.pretrain_epochs = val
        elif sweep_params.param_name == "hidden_channels":
            m_params.hidden_channels = val
        elif sweep_params.param_name == "num_layers":
            m_params.num_layers = val
        elif sweep_params.param_name == "dropout":
            m_params.dropout = val

        set_seed(data_params.seed)

        ds_info = load_dataset(data_params)
        task = ds_info["task"]

        data = None
        train_mask = val_mask = test_mask = None
        train_loader = val_loader = test_loader = None

        if task == "node":
            data = ds_info["data"].to(device)
            train_mask = ds_info["train_mask"].to(device)
            val_mask = ds_info["val_mask"].to(device)
            test_mask = ds_info["test_mask"].to(device)
        else:
            train_loader = DataLoader(ds_info["train_dataset"], batch_size=t_params.batch_size, shuffle=True)
            val_loader = DataLoader(ds_info["val_dataset"], batch_size=t_params.batch_size, shuffle=False)
            test_loader = DataLoader(ds_info["test_dataset"], batch_size=t_params.batch_size, shuffle=False)

        model = build_model(m_params, ds_info["num_features"], ds_info["num_classes"], task).to(device)

        trained_model, best_val_acc = train_model_full(
            model=model,
            task=task,
            train_params=t_params,
            device=device,
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=False,
        )

        test_metrics = evaluate(
            model=trained_model,
            task=task,
            device=device,
            data=data,
            mask=test_mask,
            loader=test_loader,
        )
        test_acc = test_metrics["accuracy"]
        print(f"{val:<15} | {best_val_acc:<15.4f} | {test_acc:<15.4f}")


def main() -> None:
    """Run the pipeline in the specified mode."""
    args = parse_args()
    (
        model_params,
        train_params,
        data_params,
        anomaly_params,
        explain_params,
        oversmoothing_params,
        sweep_params,
        robustness_params,
        topology_params,
    ) = build_params(args)

    set_seed(data_params.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "sweep":
        run_sweep(sweep_params, model_params, train_params, data_params, device)
        return

    ds_info = load_dataset(data_params)
    task: str = ds_info["task"]
    num_features: int = ds_info["num_features"]
    num_classes: int = ds_info["num_classes"]

    data = None
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

    if args.mode == "oversmoothing":
        run_oversmoothing_analysis(
            model_name=model_params.model_name,
            ds_info=ds_info,
            device=device,
            train_params=train_params,
            params=oversmoothing_params,
        )
        return

    if args.mode == "robustness":
        run_robustness_analysis(
            model_name=model_params.model_name,
            ds_info=ds_info,
            device=device,
            train_params=train_params,
            params=robustness_params,
        )
        return

    print(f"Dataset : {data_params.dataset_name}")
    print(f"Task    : {task} classification")
    print(f"Features: {num_features}  |  Classes: {num_classes}")

    model = build_model(model_params, num_features, num_classes, task).to(device)

    print(f"\nModel   : {model_params.model_name.upper()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {total_params:,}")
    print(f"Device  : {device}\n")

    model, best_val_acc = train_model_full(
        model=model,
        task=task,
        train_params=train_params,
        device=device,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        train_loader=train_loader,
        val_loader=val_loader,
        verbose=True,
    )

    if args.mode == "train":
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

    elif args.mode == "anomaly":
        run_anomaly_analysis(
            model=model,
            task=task,
            device=device,
            data=data,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            params=anomaly_params,
        )

    elif args.mode == "explain":
        assert data is not None, "Explainability requires node task."
        run_explainability_analysis(
            model=model,
            device=device,
            data=data,
            params=explain_params,
        )

    elif args.mode == "topology":
        assert data is not None, "Topology analysis requires node task."
        run_topology_analysis(
            model=model,
            task=task,
            device=device,
            data=data,
            params=topology_params,
        )


if __name__ == "__main__":
    main()
