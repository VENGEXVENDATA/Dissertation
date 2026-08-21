"""
src/community/consensus_leiden.py
==================================
Implements consensus community detection using the Leiden algorithm (python-igraph & leidenalg).
Constructs an N x N co-occurrence association matrix C_ij over independent runs 
to resolve stochastic non-determinism, then tracks temporal ARI stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from sklearn.metrics import adjusted_rand_score
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def nx_to_igraph(nx_graph: nx.Graph) -> tuple[ig.Graph, list[str]]:
    """
    Converts a NetworkX Graph into an igraph.Graph object,
    preserving node ordering and edge weight attributes.
    """
    nodes = sorted(list(nx_graph.nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    edges = []
    weights = []
    
    for u, v, data in nx_graph.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))
        
    ig_graph = ig.Graph(n=len(nodes), edges=edges, edge_attrs={"weight": weights})
    return ig_graph, nodes


def extract_stable_leiden_partition(graph: nx.Graph, num_runs: int = 100, seed: int = 42) -> dict[str, int]:
    """
    Builds a consensus partition using the Leiden algorithm via an N x N 
    co-occurrence matrix tracking node co-membership frequency.
    """
    ig_graph, nodes = nx_to_igraph(graph)
    num_nodes = len(nodes)
    
    # Pre-allocate association matrix C_ij
    co_occurrence = np.zeros((num_nodes, num_nodes), dtype=float)
    
    # 1. Execute num_runs independent Leiden partitions with varying seeds
    for run in range(num_runs):
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            seed=seed + run
        )
        
        # Map node indices to their community assignment
        communities: dict[int, list[int]] = {}
        for node_idx, comm_id in enumerate(partition.membership):
            communities.setdefault(comm_id, []).append(node_idx)
            
        # Update co-occurrence matrix C_ij
        for comm_members in communities.values():
            for i in comm_members:
                for j in comm_members:
                    co_occurrence[i, j] += 1.0
                    
    # 2. Normalize association matrix to co-membership probability [0, 1]
    co_occurrence /= num_runs
    
    # 3. Construct consensus igraph object weighted by co-occurrence probabilities
    consensus_edges = []
    consensus_weights = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = co_occurrence[i, j]
            if prob > 0.10:  # Noise reduction threshold
                consensus_edges.append((i, j))
                consensus_weights.append(prob)
                
    consensus_ig = ig.Graph(n=num_nodes, edges=consensus_edges, edge_attrs={"weight": consensus_weights})
    
    # 4. Final deterministic Leiden pass on the consensus association network
    final_part = leidenalg.find_partition(
        consensus_ig,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=seed
    )
    
    consensus_dict = {nodes[idx]: comm_id for idx, comm_id in enumerate(final_part.membership)}
    return consensus_dict


def track_leiden_timeline_stability(windows_data: list[tuple[str, nx.Graph]]) -> pd.DataFrame:
    """
    Computes Adjusted Rand Index (ARI) scores between consecutive 100-run 
    consensus Leiden partitions across a chronologically aligned timeline.
    """
    config = get_config()
    comm_config = config.get("community", {})
    consensus_runs = comm_config.get("consensus_runs", 100)
    seed = comm_config.get("random_seed_start", 42)
    
    sorted_windows = sorted(windows_data, key=lambda x: x[0])
    
    records = []
    prev_partition = None
    prev_nodes = None
    
    for idx, (date_str, graph) in enumerate(sorted_windows):
        market_name = graph.graph["market"]
        current_nodes = sorted(list(graph.nodes))
        
        current_partition = extract_stable_leiden_partition(
            graph, num_runs=consensus_runs, seed=seed
        )
        
        if prev_partition is not None:
            shared_nodes = sorted(list(set(prev_nodes) & set(current_nodes)))
            
            if len(shared_nodes) < 2:
                logger.warning("Insufficient shared nodes at %s for Leiden ARI calculation.", date_str)
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
                "algorithm": "leiden"
            })
            
        prev_partition = current_partition
        prev_nodes = current_nodes
        
    return pd.DataFrame(records)