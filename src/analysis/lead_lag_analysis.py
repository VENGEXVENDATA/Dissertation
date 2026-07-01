"""
src/analysis/lead_lag_analysis.py
==================================
Calculates the Cross-Correlation Function (CCF) between rolling network 
stability changes (d_ari) and integrated true price-based volatility indices.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Enforce repository stability across nested environment layers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def calculate_cross_correlation(
    trends_path: str | Path, 
    max_lag: int = 30
) -> None:
    """
    Computes cross-correlation coefficients across a spectrum of timeline lags
    to locate where network structural shifts exhibit peak tracking on true volatility.
    """
    t_path = Path(trends_path)
    if not t_path.exists():
        print(f"[ERROR] Target tracking file not found at: {t_path.absolute()}")
        return

    df = pd.read_csv(t_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean string columns of any trailing white spaces from csv serialization
    df.columns = df.columns.str.strip()
    
    # Diagnostic Check: If the target column isn't found, check alternatives or alert columns
    target_vol_col = 'market_volatility'
    if target_vol_col not in df.columns:
        print(f"[WARNING] Exact column '{target_vol_col}' missing from file: {t_path.absolute()}")
        print(f"Available columns found inside this CSV: {list(df.columns)}")
        
        # Defensive fallback if the integration script used an adjusted variable name
        alternatives = [c for c in df.columns if 'vol' in c.lower()]
        if alternatives:
            target_vol_col = alternatives[0]
            print(f"-> Automatically falling back to detected alternative column: '{target_vol_col}'")
        else:
            print("[CRITICAL ERROR] No volatility metrics column detected. Please run 'python -m src.analysis.volatility_integration' again to ensure data is synchronized.")
            return

    # Apply truncation fix to remove early S&P 500 initialization leakage
    corrupt_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
    df_clean = df[~corrupt_mask].reset_index(drop=True)
    
    # Isolate unique macro trends per window
    pivot_df = df_clean[['date', 'market', 'ari_stability', target_vol_col]].drop_duplicates().sort_values('date')
    markets = pivot_df['market'].unique()
    
    print("\n" + "="*75)
    print("    CROSS-CORRELATION FUNCTION (CCF) PROFILE: LEAD-LAG PEAK OPTIMIZATION")
    print("===========================================================================")
    
    plt.figure(figsize=(12, 8))
    colors = {'sp500': '#1f77b4', 'nifty50': '#ff7f0e', 'bovespa': '#2ca02c'}
    
    for market in markets:
        m_data = pivot_df[pivot_df['market'] == market].copy()
        
        # Calculate change in ARI stability (first-difference for time-series stationarity)
        m_data['d_ari'] = m_data['ari_stability'].diff()
        
        # Drop initial NaN row created by differencing, and drop missing volatility records
        m_data = m_data.dropna(subset=['d_ari', target_vol_col])
        
        if len(m_data) < max_lag * 2:
            print(f"[WARNING] Skipping {market.upper()}: Insufficient valid observations.")
            continue
            
        lags = np.arange(-max_lag, max_lag + 1)
        ccf_profile = []
        
        # Calculate correlation explicitly per lag step to guarantee perfect date index matching
        for lag in lags:
            if lag < 0:
                shifted_ari = m_data['d_ari'].shift(-lag)
                volatility = m_data[target_vol_col]
            elif lag > 0:
                shifted_ari = m_data['d_ari']
                volatility = m_data[target_vol_col].shift(lag)
            else:
                shifted_ari = m_data['d_ari']
                volatility = m_data[target_vol_col]
                
            combined = pd.DataFrame({'x': shifted_ari, 'y': volatility}).dropna()
            corr_val = np.corrcoef(combined['x'], combined['y'])[0, 1] if len(combined) > 5 else 0.0
            ccf_profile.append(corr_val)
            
        ccf_profile = np.array(ccf_profile)
        
        # Isolate peak absolute lag position
        peak_idx = np.argmax(np.abs(ccf_profile))
        peak_lag = lags[peak_idx]
        peak_corr = ccf_profile[peak_idx]
        
        print(f"Market: {market.upper():<8} | Peak Predictive Lag: {peak_lag:<3} trading days | Correlation: {peak_corr:.4f}")
        
        plt.plot(lags, ccf_profile, label=f"{market.upper()} (Peak Lag: {peak_lag}d, r={peak_corr:.2f})", 
                 color=colors[market], linewidth=1.8)
        
    plt.axvline(0, color='black', linestyle='--', alpha=0.6)
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel("Lag Parameter $\\tau$ (Days: Negative implies Network leads Volatility)")
    plt.ylabel("Cross-Correlation Coefficient $R(\\tau)$")
    plt.title("Cross-Correlation Function (CCF) Trajectory: Network Instability vs Price Volatility", loc="left", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor='white')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_png = Path("data/processed/lead_lag_ccf_profile.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print("===========================================================================")
    print(f"[SUCCESS] True Lead-Lag structural chart saved to: {out_png}\n")

if __name__ == "__main__":
    calculate_cross_correlation("data/processed/master_methodology_trends.csv")