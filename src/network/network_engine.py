"""
src/network/network_engine.py
==============================
Constructs temporal distance networks from stock log-returns.
Implements Mantegna distance conversions and transitions matrices into graph structures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_mantegna_distances(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a Pearson correlation matrix into Euclidean distances 
    using Mantegna's formula: d_ij = sqrt(2 * (1 - rho_ij))
    """
    # Any NaN correlation entries due to remaining zero-variance edge cases are filled with 0.0
    corr_filled = corr_matrix.fillna(0.0)
    
    # Clip values defensively to guard against floating-point precision anomalies outside [-1, 1]
    rho = np.clip(corr_filled.values, -1.0, 1.0)
    distances = np.sqrt(2.0 * (1.0 - rho))
    return pd.DataFrame(distances, index=corr_matrix.index, columns=corr_matrix.columns)


def build_mst_network(distance_matrix: pd.DataFrame) -> nx.Graph:
    """
    Constructs a sparse network graph from a distance matrix using a Minimum Spanning Tree.
    Inverts distances (1 / d_ij) ensuring smaller distances (stronger correlations)
    correspond to higher network edge weights for Louvain modularity optimization.
    """
    tickers = distance_matrix.columns.tolist()
    full_graph = nx.Graph()
    
    # Generate complete graph connections
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            d_ij = distance_matrix.loc[t1, t2]
            
            if np.isnan(d_ij):
                logger.warning("NaN distance encountered between %s and %s. Defaulting to standard independent metric.", t1, t2)
                d_ij = np.sqrt(2.0)
            
            # Invert distance for modularity optimization mapping
            weight_ij = 1.0 / d_ij if d_ij > 1e-6 else 1e6
            full_graph.add_edge(t1, t2, distance=d_ij, weight=weight_ij)
            
    # Extract the Minimum Spanning Tree
    mst_graph = nx.minimum_spanning_tree(full_graph, weight="distance")
    return mst_graph


def convert_window_to_network(
    date_str: str, 
    window_data: pd.DataFrame, 
    market_name: str
) -> nx.Graph:
    """
    Processes a single pre-sliced window chunk into a network graph topology.
    This acts as the target function for your multi-core parallel engine.
    """
    config = get_config()
    network_method = config["network"]["method_primary"]
    
    # 1. Compute pairwise Pearson correlation matrix
    corr_matrix = window_data.corr(method="pearson")
    
    # 2. Map correlations to Euclidean metric space
    dist_matrix = compute_mantegna_distances(corr_matrix)
    
    # 3. Filter network using the approved topology strategy
    if network_method == "mst":
        graph = build_mst_network(dist_matrix)
    elif network_method == "pmfg":
        from src.network.pmfg_engine import build_pmfg_network_fast
        graph = build_pmfg_network_fast(dist_matrix)
    else:
        raise ValueError(f"Unknown network construction method: {network_method}")
        
    # Attach spatial context attributes directly to the network header
    graph.graph["date"] = date_str
    graph.graph["market"] = market_name
    
    return graph