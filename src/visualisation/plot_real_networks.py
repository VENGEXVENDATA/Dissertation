"""
src/visualisation/plot_real_networks.py
=======================================
Generates un-capped, full-scale asset network pipeline diagrams for 
S&P 500, NIFTY 50, and Bovespa alongside master analytical trajectories.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def generate_complete_universe_suite(
    raw_dir: str | Path,
    trends_path: str | Path,
    output_dir: str | Path
) -> None:
    """
    Extracts complete, un-capped timeline correlations for all three markets, 
    filters them, and creates the definitive multi-panel visual flow for the thesis.
    """
    raw_path = Path(raw_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    market_files = {
        'sp500': 'prices_sp500.parquet',
        'nifty50': 'prices_nifty50.parquet',
        'bovespa': 'prices_bovespa.parquet'
    }
    
    # -------------------------------------------------------------------------
    # PART 1: THE FULL UNIVERSE NETWORK VISUALIZATIONS (NO CAP)
    # -------------------------------------------------------------------------
    for market, filename in market_files.items():
        file_path = raw_path / filename
        if not file_path.exists():
            print(f"[WARNING] Skipping network rendering for {market}: {file_path.name} not found.")
            continue
            
        print(f"[DATA] Ingesting complete un-capped timeline history for: {market.upper()}...")
        prices_df = pd.read_parquet(file_path).ffill().bfill()
        
        # Clean out flatline tokens dynamically
        prices_df = prices_df.loc[:, prices_df.std() > 0.001]
        
        # NOTE: Node capping has been entirely removed to plot every single stock ticker
        print(f"-> Processing complete network matrix for {market.upper()} containing {prices_df.shape[1]} active asset nodes.")
            
        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        corr_matrix = log_returns.corr().abs()
        
        # Build network containers
        G_raw = nx.from_pandas_adjacency(corr_matrix)
        G_filtered = nx.maximum_spanning_tree(G_raw, weight='weight')
        
        # Render Side-by-Side Panels
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        pos = nx.kamada_kawai_layout(G_raw)
        
        # Panel A: Complete Dense Structure
        nx.draw_networkx_nodes(G_raw, pos, ax=axes[0], node_size=20, node_color='#1f77b4', alpha=0.7)
        nx.draw_networkx_edges(G_raw, pos, ax=axes[0], alpha=0.03, edge_color='gray')
        axes[0].set_title(f"A. Un-Capped Dense Asset Network: {market.upper()}\n({len(G_raw.nodes)} Assets | Total Timeline Co-Movement Pairs)", fontsize=13, fontweight='bold')
        axes[0].axis('off')
        
        # Panel B: Filtered Architecture
        nx.draw_networkx_nodes(G_filtered, pos, ax=axes[1], node_size=25, node_color='#d62728', alpha=0.8)
        nx.draw_networkx_edges(G_filtered, pos, ax=axes[1], alpha=0.5, edge_color='#2c3e50', width=1.0)
        
        # Adjust label visibility dynamically based on node density to keep things structured
        labels = {node: str(node)[:5] for node in G_filtered.nodes}
        f_size = 5 if len(G_filtered.nodes) < 100 else 3
        nx.draw_networkx_labels(G_filtered, pos, labels, ax=axes[1], font_size=f_size, font_weight='bold')
        axes[1].set_title(f"B. Full Sparsified Topological Filter Architecture\n(Maximum Spanning Tree Backbone Container)", fontsize=13, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        out_name = out_dir / f"network_universe_{market}_complete.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Complete asset network architecture saved: {out_name}")

    # -------------------------------------------------------------------------
    # PART 2: THE METRIC & ECONOMETRIC TREND PLOTS (ARI & DUAL-AXIS SUMMARY)
    # -------------------------------------------------------------------------
    print("\n[PLOTTING] Generating multi-market empirical trend summaries...")
    trends_df = pd.read_csv(trends_path)
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    trends_df.columns = trends_df.columns.str.strip()
    
    # Apply baseline truncation patch
    corrupt_mask = (trends_df['market'] == 'sp500') & (trends_df['date'] < pd.to_datetime('2014-01-01'))
    df_clean = trends_df[~corrupt_mask].reset_index(drop=True)
    
    # 1. Consolidated ARI Comparative Graph
    plt.figure(figsize=(14, 5.5))
    colors = {'sp500': '#1f77b4', 'nifty50': '#ff7f0e', 'bovespa': '#2ca02c'}
    for m in df_clean['market'].unique():
        m_subset = df_clean[df_clean['market'] == m].drop_duplicates('date').sort_values('date')
        plt.plot(m_subset['date'], m_subset['ari_stability'], label=m.upper(), color=colors[m], alpha=0.85, linewidth=1.5)
    
    plt.xlabel("Timeline Horizons (2010 - 2024)")
    plt.ylabel("Adjusted Rand Index ($I_{AR}$)")
    plt.title("Evolutionary Network Community Stability ($I_{AR}$) Across Global Ecosystems", fontsize=13, fontweight='bold', loc='left')
    plt.legend(frameon=True, facecolor='white', loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(out_dir / "master_trajectory_ari_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Complete Tri-Market Volatility Overlay Panel
    vol_col = 'market_volatility_x' if 'market_volatility_x' in df_clean.columns else 'market_volatility'
    if vol_col in df_clean.columns:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        for idx, m in enumerate(['sp500', 'nifty50', 'bovespa']):
            m_sub = df_clean[df_clean['market'] == m].drop_duplicates('date').sort_values('date')
            if m_sub.empty: 
                continue
            
            ax1 = axes[idx]
            ax1.plot(m_sub['date'], m_sub['ari_stability'], color='#1f77b4', alpha=0.75, label='Network Stability (ARI)')
            ax1.set_ylabel(f"{m.upper()}\nStability ($I_{{AR}}$)", color='#1f77b4', fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='#1f77b4')
            ax1.grid(True, linestyle=':', alpha=0.4)
            
            ax2 = ax1.twinx()
            ax2.plot(m_sub['date'], m_sub[vol_col], color='#d62728', alpha=0.65, linestyle='--', label='Asset Price Volatility')
            ax2.set_ylabel("Volatility ($\\sigma_t$)", color='#d62728', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#d62728')
            
            if idx == 0:
                ax1.set_title("Master Analytical Flow Matrix: Network Structure Dissolution vs Price Volatility", fontsize=14, fontweight='bold', loc='left')
        
        plt.xlabel("Timeline Horizon")
        plt.tight_layout()
        plt.savefig(out_dir / "master_volatility_dual_axis_panel.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[SUCCESS] All un-capped dissertation visual flow trajectories have been generated.")
    else:
        print("[WARNING] Volatility data column missing from dataset. Dual-axis panel compilation bypassed.")

if __name__ == "__main__":
    generate_complete_universe_suite(
        raw_dir="data/raw",
        trends_path="data/processed/master_methodology_trends.csv",
        output_dir="data/processed/plots"
    )