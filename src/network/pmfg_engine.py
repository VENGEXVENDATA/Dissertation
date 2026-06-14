"""
src/network/pmfg_engine.py
===========================
Implements topological filtering via Planar Maximally Filtered Graphs (PMFG)
and falls back to Minimum Spanning Trees (MST) for robustness checking.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_pmfg_network(distance_matrix: pd.DataFrame) -> nx.Graph:
    """
    Constructs a PMFG network from a Mantegna distance matrix.
    Edges are evaluated sequentially from smallest distance (highest correlation).
    An edge is kept only if the graph remains planar.
    """
    tickers = distance_matrix.columns.tolist()
    num_nodes = len(tickers)
    max_edges = int(3 * (num_nodes - 2))
    
    # Collect and sort all possible edges by distance (ascending)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            t1, t2 = tickers[i], tickers[j]
            d_ij = distance_matrix.loc[t1, t2]
            edges.append((d_ij, t1, t2))
            
    edges.sort(key=lambda x: x[0])  # Smallest distance first
    
    pmfg_graph = nx.Graph()
    pmfg_graph.add_nodes_from(tickers)
    
    edge_count = 0
    for d_ij, t1, t2 in edges:
        if edge_count >= max_edges:
            break
            
        # Add candidate edge tentatively
        pmfg_graph.add_edge(t1, t2, distance=d_ij, weight=1.0 / d_ij if d_ij > 1e-6 else 1e6)
        
        # Check planarity using the NetworkX Boyer-Myrvold algorithm
        is_planar, _ = nx.check_planarity(pmfg_graph)
        
        if is_planar:
            edge_count += 1
        else:
            # Reject edge if it violates planarity
            pmfg_graph.remove_edge(t1, t2)
            
    return pmfg_graph


def build_mst_network(distance_matrix: pd.DataFrame) -> nx.Graph:
    """
    Constructs an MST fallback graph for validation checks.
    """
    tickers = distance_matrix.columns.tolist()
    full_graph = nx.Graph()
    
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            d_ij = distance_matrix.loc[t1, t2]
            full_graph.add_edge(t1, t2, distance=d_ij, weight=1.0 / d_ij if d_ij > 1e-6 else 1e6)
            
    return nx.minimum_spanning_tree(full_graph, weight="distance")
