"""Graph neural network model implementations.

This package exposes:
* :class:`GCN` — Graph Convolutional Network
* :class:`GraphSAGE` — GraphSAGE
"""

from models.gcn import GCN
from models.graphsage import GraphSAGE

__all__ = ["GCN", "GraphSAGE"]
