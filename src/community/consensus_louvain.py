"""
src/community/consensus_louvain.py
===================================
Runs 100 independent Louvain iterations per slice to construct a stable,
reproducible consensus partition, then computes historical ARI vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_stable_partition(graph: nx.Graph, num_runs: int = 100, seed: int = 42) -> dict[str, int]:
    """
    Builds a consensus partition over 100 independent runs to guarantee reproducibility.
    """
    nodes = sorted(list(graph.nodes))
    num_nodes = len(nodes)
    
    # Store memberships across all 100 iterations
    run_matrix = np.zeros((num_nodes, num_runs), dtype=int)
    for run in range(num_runs):
        part = community_louvain.best_partition(graph, weight="weight", random_state=seed + run)
        for idx, node in enumerate(nodes):
            run_matrix[idx, run] = part[node]
            
    # Resolve consensus using majority voting per node node
    consensus = {}
    for idx, node in enumerate(nodes):
        counts = np.bincount(run_matrix[idx, :])
        consensus[node] = int(np.argmax(counts))
        
    return consensus


def track_timeline_stability(windows_data: list[tuple[str, nx.Graph]]) -> pd.DataFrame:
    """
    Computes ARI scores between consecutive 100-run consensus partitions across time.
    """
    records = []
    prev_partition = None
    
    for idx, (date_str, graph) in enumerate(windows_data):
        market_name = graph.graph["market"]
        nodes = sorted(list(graph.nodes))
        
        # 1. Compute stable consensus partition
        current_partition = extract_stable_partition(graph, num_runs=100)
        current_labels = [current_partition[node] for node in nodes]
        
        # 2. Compare against previous time-slice using ARI
        if prev_partition is not None:
            prev_labels = [prev_partition[node] for node in nodes]
            ari_score = adjusted_rand_score(prev_labels, current_labels)
        else:
            ari_score = 1.0  # Initial baseline setting
            
        # Append rows tracking structural properties
        for node in nodes:
            records.append({
                "date": date_str,
                "market": market_name,
                "ticker": node,
                "community": current_partition[node],
                "ari_stability": ari_score
            })
            
        prev_partition = current_partition
        
    return pd.DataFrame(records)
