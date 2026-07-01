"""
src/network/pmfg_engine.py
===========================
Topological filtering via Planar Maximally Filtered Graphs (PMFG).
Uses a mathematically sound MST-seeding strategy to reduce planarity checks
by 33% while preserving strict topological validity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.planarity import check_planarity
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_pmfg_network_fast(distance_matrix: pd.DataFrame) -> nx.Graph:
    """
    Constructs a true PMFG network by seeding the graph with a guaranteed-planar
    Minimum Spanning Tree, then evaluating the remaining candidate edges.
    """
    tickers = distance_matrix.columns.tolist()
    num_nodes = len(tickers)
    max_edges = 3 * (num_nodes - 2)

    # 1. Extract upper-triangular edge candidates to avoid duplicates
    tri_upper_idx = np.triu_indices(num_nodes, k=1)
    dist_values = distance_matrix.values[tri_upper_idx]

    edge_candidates = sorted(
        zip(dist_values, tri_upper_idx[0], tri_upper_idx[1]),
        key=lambda x: x[0]
    )

    # 2. Initialize the graph container
    pmfg_graph = nx.Graph()
    pmfg_graph.add_nodes_from(tickers)
    
    # 3. Create a temporary full graph to build an MST seed
    # Since any tree has E = V - 1 and 0 cycles, it is guaranteed planar.
    full_graph = nx.Graph()
    for d_ij, u_idx, v_idx in edge_candidates:
        full_graph.add_edge(tickers[u_idx], tickers[v_idx], weight=d_ij)
        
    mst_edges = nx.minimum_spanning_tree(full_graph, weight="weight").edges(data=True)
    
    # 4. Seed MST edges into the PMFG unconditionally (Saves 33% of planarity tests safely)
    edge_count = 0
    seeded_pairs = set()
    
    for u, v, data in mst_edges:
        d_ij = data["weight"]
        weight_ij = 1.0 / d_ij if d_ij > 1e-6 else 1e6
        pmfg_graph.add_edge(u, v, distance=d_ij, weight=weight_ij)
        seeded_pairs.add(tuple(sorted([u, v])))
        edge_count += 1

    # 5. Evaluate the remaining candidate edges with strict planarity tracking
    for d_ij, u_idx, v_idx in edge_candidates:
        if edge_count >= max_edges:
            break

        t1, t2 = tickers[u_idx], tickers[v_idx]
        
        # Skip if the edge was already added during the MST initialization step
        if tuple(sorted([t1, t2])) in seeded_pairs:
            continue

        # Add the edge tentatively
        weight_ij = 1.0 / d_ij if d_ij > 1e-6 else 1e6
        pmfg_graph.add_edge(t1, t2, distance=d_ij, weight=weight_ij)
        
        # Must execute the full planarity check to prevent sub-graph non-planarity
        is_planar, _ = check_planarity(pmfg_graph)

        if is_planar:
            edge_count += 1
        else:
            # Roll back immediately if planarity is violated
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
            weight_ij = 1.0 / d_ij if d_ij > 1e-6 else 1e6
            full_graph.add_edge(t1, t2, distance=d_ij, weight=weight_ij)

    return nx.minimum_spanning_tree(full_graph, weight="distance")