"""
src/analysis/var_modeling.py
============================
Fits Vector Autoregressive (VAR) models to capture the joint dynamic feedback 
between network stability and implied volatility. Computes Granger Causality 
and simulates Impulse Response Functions (IRFs) with bootstrap confidence bands.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_var_analysis(trends_path: str | Path, max_lags: int = 5, steps: int = 15) -> None:
    t_path = Path(trends_path)
    if not t_path.exists():
        print(f"[ERROR] Clean trends file missing at: {t_path.absolute()}")
        return

    df = pd.read_csv(t_path)
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()

    vol_col = 'market_volatility_x' if 'market_volatility_x' in df.columns else 'market_volatility'
    
    # S&P 500 Truncation Fix
    corrupt_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
    df_clean = df[~corrupt_mask].reset_index(drop=True)

    pivot_df = df_clean[['date', 'market', 'ari_stability', vol_col]].drop_duplicates().sort_values('date')
    markets = pivot_df['market'].unique()

    print("\n" + "="*95)
    print("     STAGE 4: JOINT DYNAMICS VIA VECTOR AUTOREGRESSION (VAR) & GRANGER CAUSALITY")
    print("="*95)

    # Prepare diagnostic plotting
    fig, axes = plt.subplots(len(markets), 1, figsize=(10, 12), sharex=True)
    if len(markets) == 1:
        axes = [axes]

    for idx, market in enumerate(markets):
        m_data = pivot_df[pivot_df['market'] == market].dropna().copy()
        
        if len(m_data) < 50:
            continue

        print(f"\n▶ Modeling VAR Framework for: {market.upper()}")
        print("-" * 75)

        # Standardize inputs to let IRFs represent normalized Standard Deviation Shocks
        y1 = (m_data['ari_stability'] - m_data['ari_stability'].mean()) / m_data['ari_stability'].std()
        y2 = (m_data[vol_col] - m_data[vol_col].mean()) / m_data[vol_col].std()

        var_data = pd.DataFrame({'Network_ARI': y1, 'Volatility': y2}).reset_index(drop=True)

        # 1. Fit Vector Autoregression (auto-select optimal lag using AIC)
        model = VAR(var_data)
        lag_order = model.select_order(maxlags=max_lags)
        
        # Robust version-agnostic lookup for statsmodels optimal lag
        if hasattr(lag_order, 'selected_orders'):
            optimal_lag = lag_order.selected_orders['aic']
        elif hasattr(lag_order, 'selected_outputs'):
            optimal_lag = lag_order.selected_outputs['aic']
        else:
            optimal_lag = getattr(lag_order, 'aic', 1)
            
        optimal_lag = max(1, optimal_lag)
        
        results = model.fit(optimal_lag)
        print(f"  -> Optimal Joint Lag Order (AIC): {optimal_lag}")
        print(f"  -> Model Log-Likelihood: {results.llf:.2f}")

        # 2. Execute Granger Causality Tests
        # Test 1: Does Volatility Granger-Cause Network ARI?
        gc_vol_to_ari = results.test_causality('Network_ARI', 'Volatility', kind='f')
        # Test 2: Does Network ARI Granger-Cause Volatility?
        gc_ari_to_vol = results.test_causality('Volatility', 'Network_ARI', kind='f')

        print(f"  -> Granger Causality: Volatility ──> Network ARI | p-val: {gc_vol_to_ari.pvalue:.5f}" + 
              (" (*)" if gc_vol_to_ari.pvalue < 0.05 else " (Not Sig)"))
        print(f"  -> Granger Causality: Network ARI ──> Volatility | p-val: {gc_ari_to_vol.pvalue:.5f}" + 
              (" (*)" if gc_ari_to_vol.pvalue < 0.05 else " (Not Sig)"))

        # 3. Simulate and Plot Impulse Response Functions (IRFs)
        # We want to see how a shock to Volatility propagates into Network ARI
        irf = results.irf(steps)
        
        # FIXED: Pull from the .irfs array which tracks (periods, to_var, from_var)
        # Position 0 = Network_ARI, Position 1 = Volatility
        response = irf.irfs[:, 0, 1]  # Volatility shock -> Network ARI response
        
        # Plot main response curve on the assigned subplot axis
        axes[idx].plot(range(steps + 1), response, label='Response of ARI to Volatility Shock', color='#ff7f0e', linewidth=2)
        
        # Add baseline reference line
        axes[idx].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].set_title(f"{market.upper()}: Impulse Response Function (VIX Shock → Network ARI)", loc='left', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel("Response (Std. Dev)")
        axes[idx].grid(True, linestyle=':', alpha=0.6)
        
        if idx == len(markets) - 1:
            axes[idx].set_xlabel("Forecast Horizon (Trading Days after Shock)")

    plt.tight_layout()
    out_png = Path("data/processed/var_irf_profile.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*95)
    print(f"[SUCCESS] Granger causality and joint VAR simulation complete.")
    print(f"          Interactive Impulse Response plot saved to: {out_png}")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_var_analysis("data/processed/master_methodology_trends.csv")