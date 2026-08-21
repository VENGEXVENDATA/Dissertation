"""
src/community/alternative_algorithms.py
========================================
Implements comparative community detection algorithms across distinct theoretical families:
  1. Modularity Maximization: Louvain, Leiden
  2. Information Theory / Flow-Based: Infomap
  3. Local Consensus / Dynamics: Label Propagation Algorithm (LPA)
  4. Spectral / Linear Algebra: Spectral Clustering
  5. Hierarchical / Distance-Based: Hierarchical Agglomerative Clustering (HAC)

Generates standardized partition ledgers and serial ARI temporal stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def nx_to_igraph(nx_graph: nx.Graph) -> tuple[ig.Graph, list[str]]:
    """Converts a NetworkX graph to an igraph.Graph preserving node order and edge weights."""
    nodes = sorted(list(nx_graph.nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    edges = []
    weights = []
    
    for u, v, data in nx_graph.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))
        
    ig_graph = ig.Graph(n=len(nodes), edges=edges, edge_attrs={"weight": weights})
    return ig_graph, nodes


# =============================================================================
# ALGORITHM IMPLEMENTATIONS BY FAMILY
# =============================================================================

def run_infomap(nx_graph: nx.Graph) -> dict[str, int]:
    """Family 1: Information Theory & Flow-Based (Infomap)."""
    ig_graph, nodes = nx_to_igraph(nx_graph)
    partition = ig_graph.community_infomap(edge_weights="weight")
    return {nodes[idx]: comm_id for idx, comm_id in enumerate(partition.membership)}


def run_label_propagation(nx_graph: nx.Graph) -> dict[str, int]:
    """Family 2: Local Consensus & Dynamics (LPA)."""
    ig_graph, nodes = nx_to_igraph(nx_graph)
    partition = ig_graph.community_label_propagation(weights="weight")
    return {nodes[idx]: comm_id for idx, comm_id in enumerate(partition.membership)}


def run_spectral_clustering(nx_graph: nx.Graph, n_clusters: int = 5, seed: int = 42) -> dict[str, int]:
    """Family 3: Spectral Decomposition & Linear Algebra."""
    nodes = sorted(list(nx_graph.nodes))
    adj_matrix = nx.to_numpy_array(nx_graph, nodelist=nodes, weight="weight")
    
    # Bound n_clusters to not exceed N nodes
    actual_k = min(n_clusters, len(nodes) - 1)
    
    model = SpectralClustering(
        n_clusters=actual_k,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans"
    )
    labels = model.fit_predict(adj_matrix)
    return {nodes[idx]: int(labels[idx]) for idx in range(len(nodes))}


def run_hierarchical_clustering(nx_graph: nx.Graph, n_clusters: int = 5) -> dict[str, int]:
    """Family 5: Hierarchical & Connectivity-Based (HAC on Shortest Path Metric Space)."""
    nodes = sorted(list(nx_graph.nodes))
    
    # Compute shortest-path distance matrix on PMFG graph
    length_dict = dict(nx.all_pairs_dijkstra_path_length(nx_graph, weight="weight"))
    dist_matrix = np.zeros((len(nodes), len(nodes)))
    
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            dist_matrix[i, j] = length_dict[u].get(v, 0.0)
            
    actual_k = min(n_clusters, len(nodes) - 1)
    model = AgglomerativeClustering(n_clusters=actual_k, metric="precomputed", linkage="average")
    labels = model.fit_predict(dist_matrix)
    return {nodes[idx]: int(labels[idx]) for idx in range(len(nodes))}


# =============================================================================
# UNIFIED TIMELINE TRACKER
# =============================================================================

ALGORITHM_DISPATCH = {
    "infomap": run_infomap,
    "lpa": run_label_propagation,
    "spectral": run_spectral_clustering,
    "hac": run_hierarchical_clustering
}


def track_alternative_timeline_stability(
    windows_data: list[tuple[str, nx.Graph]], 
    algorithm_name: str
) -> pd.DataFrame:
    """
    Executes specified community algorithm across temporal graph windows
    and measures consecutive window Adjusted Rand Index (ARI) stability.
    """
    algo_key = algorithm_name.lower().strip()
    if algo_key not in ALGORITHM_DISPATCH:
        raise ValueError(
            f"Unsupported algorithm '{algorithm_name}'. "
            f"Available alternative choices: {list(ALGORITHM_DISPATCH.keys())}"
        )
        
    algo_func = ALGORITHM_DISPATCH[algo_key]
    sorted_windows = sorted(windows_data, key=lambda x: x[0])
    
    records = []
    prev_partition = None
    prev_nodes = None
    
    for idx, (date_str, graph) in enumerate(sorted_windows):
        market_name = graph.graph.get("market", "unknown")
        current_nodes = sorted(list(graph.nodes))
        
        # Partition graph using target algorithm
        current_partition = algo_func(graph)
        
        if prev_partition is not None:
            shared_nodes = sorted(list(set(prev_nodes) & set(current_nodes)))
            if len(shared_nodes) < 2:
                ari_score = np.nan
            else:
                prev_labels = [prev_partition[node] for node in shared_nodes]
                current_labels = [current_partition[node] for node in shared_nodes]
                ari_score = adjusted_rand_score(prev_labels, current_labels)
        else:
            ari_score = 1.0  # Initial baseline anchor
            
        for node in current_nodes:
            records.append({
                "date": date_str,
                "market": market_name,
                "ticker": node,
                "community": current_partition[node],
                "ari_stability": ari_score,
                "algorithm": algo_key
            })
            
        prev_partition = current_partition
        prev_nodes = current_nodes
        
    return pd.DataFrame(records)