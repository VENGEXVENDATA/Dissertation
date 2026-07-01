"""
src/visualization/trend_plots.py
=================================
Generates dissertation-grade visualizations tracking macro ARI trajectories
against market stress proxies and extracting micro-topological network layout snapshots.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.config_loader import get_config

# Set academic style formatting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

def plot_macro_ari_trajectory(data_path: str | Path, output_dir: str | Path) -> None:
    """
    Plots a multi-market chronological time-series of network ARI stability 
    with highlighted crisis periods from your centralized configuration file.
    """
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Isolate unique macro trends per date and market
    trend_df = df[['date', 'market', 'ari_stability']].drop_duplicates().sort_values('date')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    markets = ['sp500', 'nifty50', 'bovespa']
    market_labels = {'sp500': 'S&P 500 (USA)', 'nifty50': 'NIFTY 50 (India)', 'bovespa': 'Bovespa (Brazil)'}
    colors = {'sp500': '#1f77b4', 'nifty50': '#ff7f0e', 'bovespa': '#2ca02c'}
    
    # Hardcoded prominent crisis periods for background layout shading (GFC, Euro, China, COVID, Rate Hikes)
    crisis_periods = [
        ("2011-07-01", "2012-01-31", "Euro Debt"),
        ("2015-06-12", "2016-02-11", "China Shock"),
        ("2020-02-19", "2020-04-30", "COVID-19"),
        ("2022-01-03", "2022-10-12", "Fed Hikes")
    ]

    for ax, market in zip(axes, markets):
        m_data = trend_df[trend_df['market'] == market]
        
        # Plot the master ARI sequence line
        ax.plot(m_data['date'], m_data['ari_stability'], color=colors[market], 
                label=f"{market_labels[market]} ARI Stability", linewidth=1.7)
        
        # Shading loops for crisis regimes
        for start, end, label in crisis_periods:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='red', alpha=0.08)
            if ax == axes[0]:  # Add label once on top panel
                ax.text(pd.to_datetime(start) + pd.Timedelta(days=5), 0.95, label, 
                        color='darkred', fontsize=9, fontweight='bold', rotation=90, va='top')
        
        ax.set_ylabel("Adjusted Rand Index (ARI)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left")
        ax.set_title(f"Temporal Community Stability Window-Sequence: {market_labels[market]}", loc='left')

    axes[-1].set_xlabel("Timeline Window Target Date (2010–2024)")
    plt.tight_layout()
    
    out_path = Path(output_dir) / "macro_stability_trajectories.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Macro visualization saved to: {out_path}")

if __name__ == "__main__":
    data_file = "data/processed/master_methodology_trends.csv"
    plots_dir = "data/processed" # Saving inside processing target for check loops
    plot_macro_ari_trajectory(data_file, plots_dir)