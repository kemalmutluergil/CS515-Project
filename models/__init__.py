"""Graph neural network model implementations.

This package exposes:
* :class:`GCN` — Graph Convolutional Network
* :class:`GraphSAGE` — GraphSAGE
* :class:`DGI` — Deep Graph Infomax (unsupervised)
* :class:`GIN` — Graph Isomorphism Network
"""

from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.dgi import DGI
from models.gin import GIN

__all__ = ["GCN", "GraphSAGE", "DGI", "GIN"]
