"""
pipelines/run_analysis.py
==========================
Downstream econometrics and diagnostic script for dissertation research.
Loads master trends, generates publication-quality timeline plots,
and runs Augmented Dickey-Fuller (ADF) stationarity tests.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def run_stationarity_test(series, market_name):
    """Performs and prints Augmented Dickey-Fuller unit root results."""
    print(f"\n--- ADF Stationarity Test: {market_name.upper()} ---")
    # Drop any NaN values safely before running ADF
    clean_series = series.dropna()
    result = adfuller(clean_series)
    
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4e}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
        
    if result[1] <= 0.05:
        print("Conclusion: Stationary (Reject Null Hypothesis of Unit Root) 🎉")
    else:
        print("Conclusion: Non-Stationary (Fail to Reject Null). Needs Differencing! ⚠️")

def main():
    # Paths
    input_file = Path("data/processed/master_methodology_trends.csv")
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"[ERROR] Target file {input_file} not found. Run the parallel pipeline first.")
        return

    # 1. Load Data
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Successfully loaded master trends dataset. Shape: {df.shape}")
    
    # 2. Generate Publication-Quality Time Series Plot
    print("Generating structural stability timeline charts...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot tracking trends for each market
    sns.lineplot(data=df, x='date', y='ari_stability', hue='market', linewidth=1.8, marker='o', markersize=4)
    
    plt.title("Temporal Topological Network Stability ($ARI$) Across Global Markets", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Timeline Windows (21-Day Step)", fontsize=12)
    plt.ylabel("Adjusted Rand Index ($ARI$) Stability", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(title="Asset Universe", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    # Save chart directly to figures directory
    plot_path = output_dir / "market_methodology_stability_timeline.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Publication figure saved to: {plot_path}")
    
    # 3. Run Stationarity Diagnostic Pre-Tests
    for market in df['market'].unique():
        market_data = df[df['market'] == market].sort_values('date')
        run_stationarity_test(market_data['ari_stability'], market)

if __name__ == "__main__":
    main()
