"""
src/community/consensus_louvain.py
===================================
Constructs a stable consensus partition using co-occurrence (association) 
matrices across independent runs to resolve stochastic integer label arbitrary assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain
from sklearn.metrics import adjusted_rand_score
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_stable_partition(graph: nx.Graph, num_runs: int = 100, seed: int = 42) -> dict[str, int]:
    """
    Builds a consensus partition via an N x N association matrix,
    measuring node co-membership frequency across independent runs.
    """
    nodes = sorted(list(graph.nodes))
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Pre-allocate co-occurrence matrix C_ij
    co_occurrence = np.zeros((num_nodes, num_nodes), dtype=float)
    
    for run in range(num_runs):
        part = community_louvain.best_partition(graph, weight="weight", random_state=seed + run)
        
        # Group node indices by community assignment
        communities: dict[int, list[int]] = {}
        for node, comm_id in part.items():
            communities.setdefault(comm_id, []).append(node_to_idx[node])
            
        # Increment co-occurrence count for nodes placed in the same community
        for comm_members in communities.values():
            for i in comm_members:
                for j in comm_members:
                    co_occurrence[i, j] += 1.0
                    
    # Normalize association matrix into co-membership probability [0, 1]
    co_occurrence /= num_runs
    
    # Construct a consensus graph where edge weights equal co-membership probability
    consensus_graph = nx.Graph()
    consensus_graph.add_nodes_from(nodes)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = co_occurrence[i, j]
            if prob > 0.10:  # Threshold noise to retain dominant co-membership structures
                consensus_graph.add_edge(nodes[i], nodes[j], weight=prob)
                
    # Final Louvain pass on the deterministic co-occurrence consensus network
    final_partition = community_louvain.best_partition(consensus_graph, weight="weight", random_state=seed)
    return final_partition


def track_timeline_stability(windows_data: list[tuple[str, nx.Graph]]) -> pd.DataFrame:
    config = get_config()
    consensus_runs = config["community"]["consensus_runs"]
    seed = config["community"]["random_seed_start"]
    
    sorted_windows = sorted(windows_data, key=lambda x: x[0])
    
    records = []
    prev_partition = None
    prev_nodes = None
    
    for idx, (date_str, graph) in enumerate(sorted_windows):
        market_name = graph.graph["market"]
        current_nodes = sorted(list(graph.nodes))
        
        current_partition = extract_stable_partition(graph, num_runs=consensus_runs, seed=seed)
        
        if prev_partition is not None:
            shared_nodes = sorted(list(set(prev_nodes) & set(current_nodes)))
            
            if len(shared_nodes) < 2:
                logger.warning("Insufficient shared nodes at %s to calculate meaningful ARI.", date_str)
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
                "ari_stability": ari_score
            })
            
        prev_partition = current_partition
        prev_nodes = current_nodes
        
    return pd.DataFrame(records)