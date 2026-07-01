"""
src/visualisation/network_plots.py
===================================
Extracts high-resolution spatial network graph snapshots for specific
historical dates, coloring nodes dynamically by their Louvain consensus community.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Enforce repository mapping stability across cluster shell limits
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import get_config
from src.network.network_engine import compute_mantegna_distances
from src.network.pmfg_engine import build_pmfg_network_fast

def generate_network_snapshot(
    market_key: str, 
    target_date: str, 
    stability_csv: Path | str, 
    raw_prices_parquet: Path | str,
    output_png: Path | str
) -> None:
    """
    Constructs the exact PMFG network for a specific window, matches nodes
    with their calculated community assignments, and exports a clean visualization.
    """
    # 1. Load your compiled consensus community assignments
    stability_df = pd.read_csv(stability_csv)
    stability_df['date'] = pd.to_datetime(stability_df['date'])
    
    window_meta = stability_df[stability_df['date'] == pd.to_datetime(target_date)]
    if window_meta.empty:
        raise ValueError(f"No compiled tracking records found for date: {target_date}")
        
    # Map tickers to their specific community ID
    community_map = dict(zip(window_meta['ticker'], window_meta['community']))
    active_tickers = list(community_map.keys())
    
    # 2. Reconstruct the log-returns matrix for the specific historical window frame
    prices_df = pd.read_parquet(raw_prices_parquet)
    prices_df.index = pd.to_datetime(prices_df.index)
    
    # Locate the chronological location of the window end-date
    end_idx = prices_df.index.get_loc(pd.to_datetime(target_date))
    config = get_config()
    window_size = config["windows"]["primary"] # 126 days
    
    window_prices = prices_df.iloc[max(0, end_idx - window_size + 1):end_idx + 1][active_tickers]
    window_prices = window_prices.ffill().bfill()
    
    # Stationarize log-returns and compute Euclidean distance metrics
    log_returns = np.log(window_prices / window_prices.shift(1)).dropna()
    corr_matrix = log_returns.corr(method="pearson")
    dist_matrix = compute_mantegna_distances(corr_matrix)
    
    # 3. Filter network using the approved PMFG planarity constraints
    print(f"Filtering PMFG layout for {market_key.upper()} on {target_date}...")
    graph = build_pmfg_network_fast(dist_matrix)
    
    # 4. Configure visual elements for academic publication
    plt.figure(figsize=(10, 9))
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    
    # Scale node size by degree centrality
    degrees = dict(graph.degree())
    node_sizes = [v * 40 for v in degrees.values()]
    
    # Map community integers to a discrete color palette
    unique_communities = sorted(list(set(community_map.values())))
    color_palette = sns.color_palette("hls", len(unique_communities)).as_hex()
    node_colors = [color_palette[unique_communities.index(community_map[node])] for node in graph.nodes]
    
    # Calculate force-directed layout
    pos = nx.spring_layout(graph, k=1.5 / np.sqrt(len(graph.nodes)), iterations=50, seed=42)
    
    # Draw graph layout components
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, 
                           edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color="dimgray")
    
    plt.title(f"PMFG Network Topology Structure: {market_key.upper()} ({target_date})\n"
              f"Nodes Colored by 100-Run Consensus Community (Size Proportional to Degree Centrality)", 
              loc="left", fontsize=12, fontweight="bold")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Network architecture layout saved to: {output_png}")

if __name__ == "__main__":
    # Example execution targets representing clear baseline vs crisis windows
    stability_file = "data/processed/sp500_methodology_stability.csv"
    raw_data = "data/raw/prices_sp500.parquet"
    
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract Tranquil baseline layout
    generate_network_snapshot("sp500", "2017-07-07", stability_file, raw_data, output_dir / "sp500_topology_tranquil.png")
    
    # Extract Peak Market Stress layout
    generate_network_snapshot("sp500", "2020-04-08", stability_file, raw_data, output_dir / "sp500_topology_covid.png")