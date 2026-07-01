"""
src/community/community_engine.py
==================================
Implements consensus Louvain community detection and computes temporal 
stability profiles using the Adjusted Rand Index (ARI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_consensus_partition(graph: nx.Graph, num_runs: int = 10, seed: int = 42) -> dict[str, int]:
    """
    Runs Louvain multiple times and extracts a consensus partition to handle non-determinism.
    Nodes are grouped based on their modal community assignment across runs.
    """
    nodes = list(graph.nodes)
    run_results = np.zeros((len(nodes), num_runs), dtype=int)
    
    # Run Louvain independently num_runs times
    for run in range(num_runs):
        partition = community_louvain.best_partition(
            graph, weight="weight", random_state=seed + run
        )
        for node_idx, node in enumerate(nodes):
            run_results[node_idx, run] = partition[node]
            
    # Compile consensus via majority vote per node
    consensus_partition = {}
    for node_idx, node in enumerate(nodes):
        counts = np.bincount(run_results[node_idx, :])
        consensus_partition[node] = int(np.argmax(counts))
        
    return consensus_partition


def calculate_temporal_stability(graphs: list[nx.Graph]) -> pd.DataFrame:
    """
    Computes the Adjusted Rand Index (ARI) between consecutive network partitions.
    """
    stability_records = []
    prev_partition = None
    
    for idx, graph in enumerate(graphs):
        date_str = graph.graph.get("date", f"Step_{idx}")
        market_name = graph.graph.get("market", "unknown")
        
        # 1. Generate consensus community assignments for window t
        current_partition = get_consensus_partition(graph, num_runs=10)
        
        # 2. Extract communities as an ordered array for matching
        nodes = sorted(list(graph.nodes))
        current_labels = [current_partition[node] for node in nodes]
        
        # Record membership tracking for our Power BI relational model (Sankey/Ribbon charts)
        for node in nodes:
            stability_records.append({
                "date": date_str,
                "market": market_name,
                "ticker": node,
                "community": current_partition[node],
                "ari_stability": np.nan if idx == 0 else None
            })
            
        # 3. Calculate ARI against window t-1
        if prev_partition is not None:
            prev_labels = [prev_partition[node] for node in nodes]
            ari_score = adjusted_rand_score(prev_labels, current_labels)
            
            # Apply the computed ARI to all node records on this specific timestamp
            for record in stability_records[-len(nodes):]:
                record["ari_stability"] = ari_score
        else:
            # First window in timeline has no historical anchor baseline
            ari_score = 1.0
            for record in stability_records[-len(nodes):]:
                record["ari_stability"] = 1.0
                
        prev_partition = current_partition
        
    return pd.DataFrame(stability_records)
