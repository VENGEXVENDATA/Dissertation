"""
src/community/consensus_louvain.py
===================================
Runs independent Louvain iterations per window slice to construct a stable,
reproducible consensus partition, then computes historical ARI vectors.
Ensures chronological sorting and dynamic node intersection across time.
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
    Builds a stable consensus partition over multiple independent runs 
    to guarantee structural reproducibility against stochastic non-determinism.
    """
    nodes = sorted(list(graph.nodes))
    num_nodes = len(nodes)
    
    # Pre-allocate matrix to store memberships across all independent runs
    run_matrix = np.zeros((num_nodes, num_runs), dtype=int)
    for run in range(num_runs):
        part = community_louvain.best_partition(graph, weight="weight", random_state=seed + run)
        for idx, node in enumerate(nodes):
            run_matrix[idx, run] = part[node]
            
    # Resolve consensus using majority voting per individual node
    consensus = {}
    for idx, node in enumerate(nodes):
        counts = np.bincount(run_matrix[idx, :])
        consensus[node] = int(np.argmax(counts))
        
    return consensus


def track_timeline_stability(windows_data: list[tuple[str, nx.Graph]]) -> pd.DataFrame:
    """
    Computes Adjusted Rand Index (ARI) scores between consecutive 100-run 
    consensus partitions across a chronologically aligned timeline.
    """
    config = get_config()
    consensus_runs = config["community"]["consensus_runs"]
    seed = config["community"]["random_seed_start"]
    
    # CRITICAL: Enforce strict chronological alignment to correct asynchronous multi-core output
    sorted_windows = sorted(windows_data, key=lambda x: x[0])
    
    records = []
    prev_partition = None
    prev_nodes = None
    
    for idx, (date_str, graph) in enumerate(sorted_windows):
        market_name = graph.graph["market"]
        current_nodes = sorted(list(graph.nodes))
        
        # 1. Compute stable consensus partition for window t
        current_partition = extract_stable_partition(graph, num_runs=consensus_runs, seed=seed)
        
        # 2. Compare against previous time-slice (t-1) using intersection safety guards
        if prev_partition is not None:
            # Find the intersection of tickers present in both consecutive intervals
            shared_nodes = sorted(list(set(prev_nodes) & set(current_nodes)))
            
            if len(shared_nodes) < 2:
                logger.warning("Insufficient shared nodes at %s to calculate meaningful ARI.", date_str)
                ari_score = np.nan
            else:
                # Align matching historical and current labels for identical entities
                prev_labels = [prev_partition[node] for node in shared_nodes]
                current_labels = [current_partition[node] for node in shared_nodes]
                ari_score = adjusted_rand_score(prev_labels, current_labels)
        else:
            ari_score = 1.0  # Establish initial historical baseline anchor
            
        # Append rows tracking structural properties to pass to the relational model
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