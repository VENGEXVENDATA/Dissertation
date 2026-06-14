"""
src/network/network_engine.py
==============================
Constructs temporal rolling distance networks from stock log-returns.
Implements Mantegna distance conversions with defensive NaN handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_mantegna_distances(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a Pearson correlation matrix into Euclidean distances 
    using Mantegna's formula: d_ij = sqrt(2 * (1 - rho_ij))
    Fills any NaN correlation entries (caused by zero variance) with 0.0.
    """
    # Defensive fix: replace NaN values (division-by-zero artifacts from flat periods) with 0.0
    corr_filled = corr_matrix.fillna(0.0)
    
    # Clip values defensively to guard against floating-point precision anomalies outside [-1, 1]
    rho = np.clip(corr_filled.values, -1.0, 1.0)
    distances = np.sqrt(2.0 * (1.0 - rho))
    return pd.DataFrame(distances, index=corr_matrix.index, columns=corr_matrix.columns)


def build_mst_network(distance_matrix: pd.DataFrame) -> nx.Graph:
    """
    Constructs a sparse network graph from a distance matrix using a Minimum Spanning Tree.
    Edge weights are inverted (1 / d_ij) ensuring smaller distances (stronger correlations)
    correspond to higher network edge weights for Louvain modularity optimization.
    """
    tickers = distance_matrix.columns.tolist()
    full_graph = nx.Graph()
    
    # Generate complete graph connections
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            d_ij = distance_matrix.loc[t1, t2]
            
            # Additional assertion safeguard
            if np.isnan(d_ij):
                logger.warning("NaN distance encountered between %s and %s. Defaulting to standard independent metric.", t1, t2)
                d_ij = np.sqrt(2.0)
            
            # Prevent divide-by-zero errors on perfect correlation edge-cases
            weight_ij = 1.0 / d_ij if d_ij > 1e-6 else 1e6
            full_graph.add_edge(t1, t2, distance=d_ij, weight=weight_ij)
            
    # Extract the Minimum Spanning Tree based on the distance metric constraint
    mst_graph = nx.minimum_spanning_tree(full_graph, weight="distance")
    return mst_graph


def generate_temporal_network_sequence(
    market_name: str, 
    processed_dir: str = "data/processed", 
    window_size: int = 60
) -> list[nx.Graph]:
    """
    Slides a rolling window over log-returns to construct a sequence of sparse graphs.
    """
    returns_path = Path(processed_dir) / f"{market_name}_clean_returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError(f"Clean returns missing for {market_name}. Run preprocessing first.")
        
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    total_steps = len(returns_df) - window_size + 1
    
    logger.info("Generating network sequence for %s with window w=%d (Total slices: %d)", 
                market_name, window_size, total_steps)
                
    network_sequence = []
    
    for start_idx in range(total_steps):
        end_idx = start_idx + window_size
        window_data = returns_df.iloc[start_idx:end_idx]
        
        # 1. Compute pairwise rolling Pearson correlation matrix
        corr_matrix = window_data.corr(method="pearson")
        
        # 2. Map correlations to Euclidean metric space with custom NaN handler
        dist_matrix = compute_mantegna_distances(corr_matrix)
        
        # 3. Filter network using the approved MST topology strategy
        mst_net = build_mst_network(dist_matrix)
        
        # Attach spatial context attributes directly to the network header
        mst_net.graph["date"] = returns_df.index[end_idx - 1].strftime("%Y-%m-%d")
        mst_net.graph["market"] = market_name
        
        network_sequence.append(mst_net)
        
    logger.info("Successfully constructed %d temporal graphs for %s.", len(network_sequence), market_name)
    return network_sequence
