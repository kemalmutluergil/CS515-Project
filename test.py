"""Evaluation routines for node-level and graph-level classification.

This module provides a unified ``evaluate`` dispatcher as well as the
underlying task-specific helpers.
"""

from typing import Dict, Optional, List, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from parameters import AnomalyParams, ExplainParams, OversmoothingParams, RobustnessParams, TopologyParams, TrainParams


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


def compute_isolation_forest_scores(
    embeddings: np.ndarray,
    contamination: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Fit IsolationForest and return anomaly scores (higher = more anomalous).

    Args:
        embeddings (np.ndarray): Representation matrix of shape (num_samples, hidden_channels).
        contamination (float): Proportion of outliers.
        seed (int): Random seed.

    Returns:
        np.ndarray: Anomaly scores for each sample.
    """
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=contamination, random_state=seed)
    iso.fit(embeddings)
    return -iso.score_samples(embeddings)


def run_anomaly_analysis(
    model: torch.nn.Module,
    task: str,
    device: torch.device,
    data: Data | None,
    train_loader: DataLoader | None,
    val_loader: DataLoader | None,
    test_loader: DataLoader | None,
    params: AnomalyParams,
) -> None:
    """Run anomaly detection using Isolation Forest and DGI native scores.

    Args:
        model (torch.nn.Module): GNN model.
        task (str): Task type ("node" or "graph").
        device (torch.device): Device to place tensors on.
        data (Optional[Data]): A single graph Data object (node task only).
        train_loader (Optional[DataLoader]): Training DataLoader (graph task only).
        val_loader (Optional[DataLoader]): Validation DataLoader (graph task only).
        test_loader (Optional[DataLoader]): Testing DataLoader (graph task only).
        params (AnomalyParams): Anomaly detection hyperparameters.
    """
    model.eval()

    print("\n--- Anomaly Detection Analysis ---")

    if task == "node":
        assert data is not None
        with torch.no_grad():
            emb = model.embed(data.x, data.edge_index).cpu().numpy()

        if_scores = compute_isolation_forest_scores(emb, contamination=params.contamination, seed=42)
        top_if_idx = np.argsort(-if_scores)[:params.top_k]

        print(f"\nTop-{params.top_k} Most Anomalous Nodes (Isolation Forest):")
        for rank, idx in enumerate(top_if_idx, 1):
            lbl = int(data.y[idx].item())
            print(f"  Rank {rank:>2d} | Node {idx:>5d} | Class: {lbl:>2d} | Anomaly Score: {if_scores[idx]:.4f}")

        if hasattr(model, "anomaly_score"):
            with torch.no_grad():
                disc_scores = model.anomaly_score(data.x, data.edge_index).cpu().numpy()
            top_disc_idx = np.argsort(-disc_scores)[:params.top_k]

            print(f"\nTop-{params.top_k} Most Anomalous Nodes (DGI Discriminator):")
            for rank, idx in enumerate(top_disc_idx, 1):
                lbl = int(data.y[idx].item())
                print(f"  Rank {rank:>2d} | Node {idx:>5d} | Class: {lbl:>2d} | Discriminator Score: {disc_scores[idx]:.4f}")

            corr = np.corrcoef(if_scores, disc_scores)[0, 1]
            print(f"\nPearson correlation between IF and DGI discriminator scores: {corr:.4f}")

    else:
        assert train_loader is not None and val_loader is not None and test_loader is not None
        embs_list, labels_list = [], []
        with torch.no_grad():
            for loader in [train_loader, val_loader, test_loader]:
                for batch in loader:
                    batch = batch.to(device)
                    z = model.embed(batch.x, batch.edge_index, batch.batch)
                    embs_list.append(z.cpu().numpy())
                    labels_list.append(batch.y.cpu().numpy())

        emb = np.vstack(embs_list)
        labels = np.concatenate(labels_list)

        if_scores = compute_isolation_forest_scores(emb, contamination=params.contamination, seed=42)
        top_if_idx = np.argsort(-if_scores)[:params.top_k]

        print(f"\nTop-{params.top_k} Most Anomalous Graphs (Isolation Forest):")
        for rank, idx in enumerate(top_if_idx, 1):
            lbl = int(labels[idx])
            print(f"  Rank {rank:>2d} | Graph {idx:>5d} | Class: {lbl:>2d} | Anomaly Score: {if_scores[idx]:.4f}")

        if hasattr(model, "anomaly_score"):
            disc_scores_list = []
            with torch.no_grad():
                for loader in [train_loader, val_loader, test_loader]:
                    for batch in loader:
                        batch = batch.to(device)
                        scores = model.anomaly_score(batch.x, batch.edge_index, batch.batch)
                        disc_scores_list.append(scores.cpu().numpy())
            disc_scores = np.concatenate(disc_scores_list)
            top_disc_idx = np.argsort(-disc_scores)[:params.top_k]

            print(f"\nTop-{params.top_k} Most Anomalous Graphs (DGI Discriminator):")
            for rank, idx in enumerate(top_disc_idx, 1):
                lbl = int(labels[idx])
                print(f"  Rank {rank:>2d} | Graph {idx:>5d} | Class: {lbl:>2d} | Discriminator Score: {disc_scores[idx]:.4f}")

            corr = np.corrcoef(if_scores, disc_scores)[0, 1]
            print(f"\nPearson correlation between IF and DGI discriminator scores: {corr:.4f}")


def run_explainability_analysis(
    model: torch.nn.Module,
    device: torch.device,
    data: Data,
    params: ExplainParams,
) -> None:
    """Run GNNExplainer to find node feature importances.

    Args:
        model (torch.nn.Module): The trained GNN model.
        device (torch.device): Device to execute on.
        data (Data): Graph data.
        params (ExplainParams): GNNExplainer parameters.
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    model.eval()
    with torch.no_grad():
        emb = model.embed(data.x, data.edge_index).cpu().numpy()

    scores = compute_isolation_forest_scores(emb, contamination=0.1, seed=42)
    normal_node = int(np.argmin(scores))
    anomalous_node = int(np.argmax(scores))

    print(f"\n--- Explainability Analysis ---")
    print(f"Normal Node Selected   : {normal_node} (Score={scores[normal_node]:.4f})")
    print(f"Anomalous Node Selected: {anomalous_node} (Score={scores[anomalous_node]:.4f})")

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=params.epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )

    print(f"\nExplaining normal node {normal_node}...")
    exp_normal = explainer(data.x, data.edge_index, index=normal_node)
    feat_norm = exp_normal.node_mask.mean(dim=0).cpu().numpy()

    top_k = 10
    top_norm_idx = np.argsort(feat_norm)[-top_k:][::-1]
    print(f"Top {top_k} Feature Importances for Normal Node {normal_node}:")
    for rank, idx in enumerate(top_norm_idx, 1):
        print(f"  Rank {rank:>2d} | Feature {idx:>4d} | Importance: {feat_norm[idx]:.4f}")

    print(f"\nExplaining anomalous node {anomalous_node}...")
    exp_anomalous = explainer(data.x, data.edge_index, index=anomalous_node)
    feat_anom = exp_anomalous.node_mask.mean(dim=0).cpu().numpy()

    top_anom_idx = np.argsort(feat_anom)[-top_k:][::-1]
    print(f"Top {top_k} Feature Importances for Anomalous Node {anomalous_node}:")
    for rank, idx in enumerate(top_anom_idx, 1):
        print(f"  Rank {rank:>2d} | Feature {idx:>4d} | Importance: {feat_anom[idx]:.4f}")


def compute_dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Compute Normalized Dirichlet Energy of node representations.

    Args:
        x (torch.Tensor): Representation matrix [num_nodes, hidden_channels].
        edge_index (torch.Tensor): Edge index tensor.

    Returns:
        float: Normalized Dirichlet Energy.
    """
    row, col = edge_index
    diff = x[row] - x[col]
    tr_x_l_x = 0.5 * torch.sum(diff * diff).item()
    tr_x_x = torch.sum(x * x).item()
    return tr_x_l_x / (tr_x_x + 1e-12)


def compute_mean_cosine_similarity(x: torch.Tensor) -> float:
    """Compute the mean pairwise cosine similarity among nodes.

    Args:
        x (torch.Tensor): Node representation matrix.

    Returns:
        float: Mean pairwise cosine similarity.
    """
    num_nodes = x.size(0)
    if num_nodes > 1000:
        indices = torch.randperm(num_nodes)[:1000]
        x_sample = x[indices]
    else:
        x_sample = x

    norm = torch.norm(x_sample, p=2, dim=1, keepdim=True)
    x_norm = x_sample / (norm + 1e-12)
    sim_matrix = torch.matmul(x_norm, x_norm.t())
    return torch.mean(sim_matrix).item()


@torch.no_grad()
def extract_layer_representations(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> List[torch.Tensor]:
    """Forward pass to extract intermediate layer outputs (after conv + relu).

    Args:
        model (torch.nn.Module): The model to extract layer outputs from.
        x (torch.Tensor): Input feature matrix.
        edge_index (torch.Tensor): Edge index tensor.

    Returns:
        List[torch.Tensor]: Representation matrices at each layer depth.
    """
    model.eval()
    reps = [x.cpu()]
    h = x
    for conv in model.convs:
        h = conv(h, edge_index)
        h = F.relu(h)
        reps.append(h.cpu())
    return reps


def run_oversmoothing_analysis(
    model_name: str,
    ds_info: dict,
    device: torch.device,
    train_params: TrainParams,
    params: OversmoothingParams,
) -> None:
    """Analyze Dirichlet energy and cosine similarity across depths.

    Args:
        model_name (str): Model name.
        ds_info (dict): Loaded dataset dictionary.
        device (torch.device): Device to place tensors on.
        train_params (TrainParams): Training parameters.
        params (OversmoothingParams): Oversmoothing analysis configuration parameters.
    """
    from main import build_model, set_seed
    from train import train_model_full

    task = ds_info["task"]
    if task != "node":
        print("Oversmoothing analysis is only supported for node classification tasks.")
        return

    data = ds_info["data"].to(device)
    train_mask = ds_info["train_mask"].to(device)
    val_mask = ds_info["val_mask"].to(device)
    test_mask = ds_info["test_mask"].to(device)

    print(f"\n--- Oversmoothing Analysis for {model_name.upper()} ---")
    print(f"{'Depth':<6} | {'Test Acc':<10} | {'Layer':<5} | {'Dirichlet Energy':<18} | {'Cosine Sim':<12}")
    print("-" * 60)

    for depth in params.depths:
        from parameters import ModelParams
        model_params = ModelParams(
            model_name=model_name,
            hidden_channels=64,
            num_layers=depth,
            dropout=0.5,
        )
        set_seed(train_params.seed if hasattr(train_params, "seed") else 42)
        model = build_model(model_params, ds_info["num_features"], ds_info["num_classes"], task).to(device)

        trained_model, _ = train_model_full(
            model=model,
            task=task,
            train_params=train_params,
            device=device,
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            verbose=False,
        )

        test_acc = evaluate(trained_model, task, device, data=data, mask=test_mask)["accuracy"]

        layer_reps = extract_layer_representations(trained_model, data.x, data.edge_index)
        for layer_idx, rep in enumerate(layer_reps):
            energy = compute_dirichlet_energy(rep, data.edge_index.cpu())
            cos_sim = compute_mean_cosine_similarity(rep)
            print(f"{depth:<6d} | {test_acc:<10.4f} | {layer_idx:<5d} | {energy:<18.6e} | {cos_sim:<12.6f}")


def perturb_edges_delete(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly delete a fraction p of edges.

    Args:
        edge_index (torch.Tensor): Edge indices.
        p (float): Proportion of edges to delete.

    Returns:
        torch.Tensor: Perturbed edge indices.
    """
    if p == 0.0:
        return edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > p
    return edge_index[:, keep_mask]


def perturb_edges_add(
    edge_index: torch.Tensor,
    num_nodes: int,
    p: float,
) -> torch.Tensor:
    """Randomly add a fraction p of spurious edges.

    Args:
        edge_index (torch.Tensor): Edge indices.
        num_nodes (int): Total number of nodes.
        p (float): Proportion of edges to add.

    Returns:
        torch.Tensor: Perturbed edge indices.
    """
    if p == 0.0:
        return edge_index
    num_edges = edge_index.size(1)
    num_to_add = int(num_edges * p)
    new_edges = torch.randint(0, num_nodes, (2, num_to_add), device=edge_index.device)
    return torch.cat([edge_index, new_edges], dim=1)


def perturb_features_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise to feature matrix.

    Args:
        x (torch.Tensor): Input feature matrix.
        std (float): Standard deviation of Gaussian noise.

    Returns:
        torch.Tensor: Perturbed feature matrix.
    """
    if std == 0.0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise


def run_robustness_analysis(
    model_name: str,
    ds_info: dict,
    device: torch.device,
    train_params: TrainParams,
    params: RobustnessParams,
) -> None:
    """Evaluate accuracy and anomaly stability under noise.

    Args:
        model_name (str): Model name.
        ds_info (dict): Loaded dataset dictionary.
        device (torch.device): Device to place tensors on.
        train_params (TrainParams): Training parameters.
        params (RobustnessParams): Robustness analysis parameters.
    """
    from scipy.stats import spearmanr
    from main import build_model, set_seed
    from train import train_model_full
    from torch_geometric.data import Data as PyGData

    task = ds_info["task"]
    if task != "node":
        print("Robustness analysis is currently only supported for node classification tasks.")
        return

    clean_data = ds_info["data"].to(device)
    train_mask = ds_info["train_mask"].to(device)
    val_mask = ds_info["val_mask"].to(device)
    test_mask = ds_info["test_mask"].to(device)

    from parameters import ModelParams
    model_params = ModelParams(
        model_name=model_name,
        hidden_channels=64,
        num_layers=2,
        dropout=0.5,
    )
    set_seed(train_params.seed if hasattr(train_params, "seed") else 42)
    baseline_model = build_model(model_params, ds_info["num_features"], ds_info["num_classes"], task).to(device)

    print(f"\n--- Robustness Auditing for {model_name.upper()} ---")
    print("Training baseline model on clean data...")
    baseline_model, _ = train_model_full(
        model=baseline_model,
        task=task,
        train_params=train_params,
        device=device,
        data=clean_data,
        train_mask=train_mask,
        val_mask=val_mask,
        verbose=False,
    )

    baseline_model.eval()
    with torch.no_grad():
        clean_emb = baseline_model.embed(clean_data.x, clean_data.edge_index).cpu().numpy()

    clean_anom_scores = compute_isolation_forest_scores(clean_emb, contamination=0.1, seed=42)
    clean_anom_ranks = np.argsort(-clean_anom_scores)

    print(f"\nPerturbation Type: {params.pert_type}")
    print(f"{'Intensity':<10} | {'Test Accuracy':<15} | {'Spearman Rank Corr':<20}")
    print("-" * 50)

    for val in params.pert_values:
        p_x = clean_data.x.clone()
        p_edge = clean_data.edge_index.clone()

        if params.pert_type == "edge_drop":
            p_edge = perturb_edges_delete(p_edge, val)
        elif params.pert_type == "edge_add":
            p_edge = perturb_edges_add(p_edge, clean_data.num_nodes, val)
        elif params.pert_type == "feature_noise":
            p_x = perturb_features_noise(p_x, val)

        set_seed(train_params.seed if hasattr(train_params, "seed") else 42)
        perturbed_model = build_model(model_params, ds_info["num_features"], ds_info["num_classes"], task).to(device)

        perturbed_data = PyGData(x=p_x, edge_index=p_edge, y=clean_data.y).to(device)
        perturbed_model, _ = train_model_full(
            model=perturbed_model,
            task=task,
            train_params=train_params,
            device=device,
            data=perturbed_data,
            train_mask=train_mask,
            val_mask=val_mask,
            verbose=False,
        )

        perturbed_model.eval()
        with torch.no_grad():
            pred_out = perturbed_model(p_x, p_edge)
            pred = pred_out.argmax(dim=-1)
            acc = (pred[test_mask] == clean_data.y[test_mask]).sum().item() / test_mask.sum().item()

            pert_emb = perturbed_model.embed(p_x, p_edge).cpu().numpy()

        pert_anom_scores = compute_isolation_forest_scores(pert_emb, contamination=0.1, seed=42)
        pert_anom_ranks = np.argsort(-pert_anom_scores)

        corr, _ = spearmanr(clean_anom_ranks, pert_anom_ranks)
        print(f"{val:<10.2f} | {acc:<15.4f} | {corr:<20.4f}")


def compute_neighborhood_homophily(
    edge_index: torch.Tensor,
    y: torch.Tensor,
) -> np.ndarray:
    """Compute neighborhood homophily for each node in the graph.

    Args:
        edge_index (torch.Tensor): Edge indices.
        y (torch.Tensor): Labels of nodes.

    Returns:
        np.ndarray: Homophily scores.
    """
    num_nodes = y.size(0)
    homophilies = torch.zeros(num_nodes)
    row, col = edge_index

    for i in range(num_nodes):
        neighbors = col[row == i]
        if len(neighbors) == 0:
            homophilies[i] = 1.0
        else:
            neighbor_classes = y[neighbors]
            same_class = (neighbor_classes == y[i]).sum().item()
            homophilies[i] = same_class / len(neighbors)

    return homophilies.cpu().numpy()


def run_topology_analysis(
    model: torch.nn.Module,
    task: str,
    device: torch.device,
    data: Data,
    params: TopologyParams,
) -> None:
    """Analyze classification accuracy and anomaly scores vs node degree and homophily.

    Args:
        model (torch.nn.Module): The model to analyze.
        task (str): The classification task type.
        device (torch.device): Device to place tensors on.
        data (Data): Graph dataset.
        params (TopologyParams): Topology parameters.
    """
    from torch_geometric.utils import degree
    import pandas as pd

    if task != "node":
        print("Topology analysis is only supported for node classification tasks.")
        return

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1).cpu().numpy()
        emb = model.embed(data.x, data.edge_index).cpu().numpy()

    y_true = data.y.cpu().numpy()
    correct = (pred == y_true)

    anomaly_scores = compute_isolation_forest_scores(emb, contamination=0.1, seed=42)
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
    homophily = compute_neighborhood_homophily(data.edge_index, data.y)

    df = pd.DataFrame({
        "degree": deg,
        "homophily": homophily,
        "anomaly_score": anomaly_scores,
        "correct": correct,
        "test_mask": data.test_mask.cpu().numpy(),
    })

    def bin_degree(d: float) -> str:
        if d == 1:
            return "1"
        elif d == 2:
            return "2"
        elif 3 <= d <= 5:
            return "3-5"
        elif 6 <= d <= 10:
            return "6-10"
        return "11+"

    df["degree_bin"] = df["degree"].apply(bin_degree)

    def bin_homophily(h: float) -> str:
        if h <= 0.2:
            return "0.0-0.2"
        elif h <= 0.4:
            return "0.2-0.4"
        elif h <= 0.6:
            return "0.4-0.6"
        elif h <= 0.8:
            return "0.6-0.8"
        return "0.8-1.0"

    df["homophily_bin"] = df["homophily"].apply(bin_homophily)

    print("\n--- Topological Auditing ---")

    print("\nDegree Bin Grouping:")
    print(f"{'Degree Bin':<12} | {'Count':<6} | {'Test Accuracy':<15} | {'Mean Anomaly Score':<20}")
    print("-" * 60)
    for d_bin in params.degree_bins:
        sub = df[df["degree_bin"] == d_bin]
        test_sub = sub[sub["test_mask"]]
        cnt = len(sub)
        acc = test_sub["correct"].mean() if len(test_sub) > 0 else 0.0
        avg_anom = sub["anomaly_score"].mean() if cnt > 0 else 0.0
        print(f"{d_bin:<12} | {cnt:<6d} | {acc:<15.4f} | {avg_anom:<20.4f}")

    print("\nHomophily Bin Grouping:")
    print(f"{'Homophily Bin':<14} | {'Count':<6} | {'Test Accuracy':<15} | {'Mean Anomaly Score':<20}")
    print("-" * 62)
    for h_bin in params.homophily_bins:
        sub = df[df["homophily_bin"] == h_bin]
        test_sub = sub[sub["test_mask"]]
        cnt = len(sub)
        acc = test_sub["correct"].mean() if len(test_sub) > 0 else 0.0
        avg_anom = sub["anomaly_score"].mean() if cnt > 0 else 0.0
        print(f"{h_bin:<14} | {cnt:<6d} | {acc:<15.4f} | {avg_anom:<20.4f}")

    corr = df[["degree", "homophily", "anomaly_score", "correct"]].corr()
    print("\nCorrelation Matrix (Continuous Properties):")
    print(corr.to_string())
