"""Parameter dataclasses for the graph learning pipeline.

Each dataclass groups a logically related set of hyperparameters so that
functions receive only the parameters they need.
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class ModelParams:
    """Parameters that define the architecture of a graph neural network.

    Attributes:
        model_name: Which GNN architecture to use (``gcn`` or ``graphsage``).
        hidden_channels: Width of every hidden layer.
        num_layers: Total number of message-passing layers.
        dropout: Dropout probability applied after each hidden layer.
    """

    model_name: str = "gcn"
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5


@dataclass
class TrainParams:
    """Parameters that control the training procedure.

    Attributes:
        lr: Learning rate for the optimizer.
        weight_decay: L2 regularization coefficient.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size (used for graph-classification datasets).
        patience: Number of epochs without validation improvement before
            early stopping triggers.  Set to ``0`` to disable.
        pretrain_epochs: Number of unsupervised pre-training epochs (DGI only).
    """

    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    batch_size: int = 32
    patience: int = 20
    pretrain_epochs: int = 300


@dataclass
class DataParams:
    """Parameters that specify which dataset to load and how to split it.

    Attributes:
        dataset_name: Dataset identifier (``cora``, ``karate``, ``enzymes``,
            or ``proteins``).
        data_root: Root directory where downloaded data is cached.
        train_ratio: Fraction of data used for training (graph-level or
            Karate Club node-level splits).
        val_ratio: Fraction of data used for validation.
        seed: Random seed for reproducibility.
    """

    dataset_name: str = "cora"
    data_root: str = "./data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42


@dataclass
class AnomalyParams:
    """Parameters for anomaly detection.

    Attributes:
        contamination: Proportion of outliers in the dataset.
        top_k: Number of most anomalous nodes/graphs to print.
    """

    contamination: float = 0.1
    top_k: int = 10


@dataclass
class ExplainParams:
    """Parameters for GNNExplainer.

    Attributes:
        epochs: Number of epochs to train the explanation mask.
    """

    epochs: int = 100


@dataclass
class OversmoothingParams:
    """Parameters for oversmoothing analysis.

    Attributes:
        depths: List of model depths to evaluate.
    """

    depths: List[int] = field(default_factory=lambda: [2, 4, 8, 16])


@dataclass
class SweepParams:
    """Parameters for hyperparameter sweeps.

    Attributes:
        param_name: Hyperparameter to sweep over (e.g. lr, hidden_channels).
        values: List of values to test during the sweep.
    """

    param_name: str = "lr"
    values: List[Any] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.1])


@dataclass
class RobustnessParams:
    """Parameters for robustness evaluations under perturbations.

    Attributes:
        pert_type: Type of perturbation ("edge_drop", "edge_add", or "feature_noise").
        pert_values: List of perturbation values/intensities to test.
    """

    pert_type: str = "edge_drop"
    pert_values: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.4]
    )


@dataclass
class TopologyParams:
    """Parameters for graph topological analysis.

    Attributes:
        degree_bins: Bins for grouping node degrees.
        homophily_bins: Bins for grouping neighborhood homophily.
    """

    degree_bins: List[str] = field(
        default_factory=lambda: ["1", "2", "3-5", "6-10", "11+"]
    )
    homophily_bins: List[str] = field(
        default_factory=lambda: [
            "0.0-0.2",
            "0.2-0.4",
            "0.4-0.6",
            "0.6-0.8",
            "0.8-1.0",
        ]
    )
