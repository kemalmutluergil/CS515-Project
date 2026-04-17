"""Parameter dataclasses for the graph learning pipeline.

Each dataclass groups a logically related set of hyperparameters so that
functions receive only the parameters they need.
"""

from dataclasses import dataclass, field


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
    """

    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    batch_size: int = 32
    patience: int = 20


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
