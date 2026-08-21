"""
src/analysis/inter_crisis_analysis.py
=====================================
Isolates and evaluates inter-crisis tranquil recovery regimes across markets.
Computes structural variance metrics, Hurst Exponents for long-memory tracking,
and builds an internal distribution profile for the intervals between shocks.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def load_crisis_config(root_dir: Path) -> dict:
    crisis_path = root_dir / "config" / "crisis_periods.yaml"
    if not crisis_path.exists():
        return {}
    with open(crisis_path, "r") as f:
        return yaml.safe_load(f) or {}

def calculate_hurst_exponent(time_series: np.ndarray) -> float:
    """Calculates the Hurst Exponent (H) to verify long-memory persistence."""
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    # FIXED: poly[0] is directly the Hurst Exponent (H)
    return float(poly[0])

def run_inter_crisis_audit(trends_path: str | Path) -> None:
    t_path = Path(trends_path)
    if not t_path.exists():
        print(f"[ERROR] Target file not found at: {t_path.absolute()}")
        return

    df = pd.read_csv(t_path)
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()
    
    vol_col = 'market_volatility' if 'market_volatility' in df.columns else 'market_volatility_x'
    
    corrupt_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
    df_clean = df[~corrupt_mask].reset_index(drop=True)
    
    pivot_df = df_clean[['date', 'market', 'ari_stability', vol_col]].drop_duplicates().sort_values('date')
    
    script_dir = Path(__file__).resolve().parent
    root_dir = next((p for p in [script_dir] + list(script_dir.parents) if (p / "config").is_dir()), script_dir)
    crisis_config = load_crisis_config(root_dir)
    
    crises = sorted(crisis_config.get('crises', []), key=lambda x: pd.to_datetime(x['crisis_start']))
    
    print("\n" + "="*95)
    print("      STAGE 3: INTER-CRISIS STRUCTURAL PROFILE & LONG-MEMORY EXCURSIONS")
    print("="*95)
    
    markets = pivot_df['market'].unique()
    
    for market in markets:
        m_data = pivot_df[pivot_df['market'] == market].copy().reset_index(drop=True)
        
        is_tranquil = np.ones(len(m_data), dtype=bool)
        for crisis in crises:
            c_start = pd.to_datetime(crisis['crisis_start'])
            c_end = pd.to_datetime(crisis['crisis_end'])
            is_tranquil = is_tranquil & ~((m_data['date'] >= c_start) & (m_data['date'] <= c_end))
            
        m_tranquil = m_data[is_tranquil].copy()
        ari_vals = m_tranquil['ari_stability'].dropna().values
        
        if len(ari_vals) < 10:
            print(f"\n[WARNING] Insufficient inter-crisis rows for market: {market.upper()}")
            continue
            
        print(f"\n▶ Analyzing Tranquil Inter-Crisis Windows for: {market.upper()}")
        print("-" * 75)
        
        mean_ari = np.mean(ari_vals)
        std_ari = np.std(ari_vals)
        coef_variation = std_ari / mean_ari if mean_ari != 0 else 0.0
        
        print(f"  -> Inter-Crisis Mean ARI Stability : {mean_ari:.4f}")
        print(f"  -> Inter-Crisis Structural Variance: {std_ari:.4f}")
        print(f"  -> Coefficient of Variation (CV)   : {coef_variation:.4f}")
        
        try:
            hurst = calculate_hurst_exponent(ari_vals)
            print(f"  -> Structural Memory (Hurst, H)    : {hurst:.4f}", end="")
            if hurst > 0.55:
                print(" (Persistent Structure Drift) 📈")
            elif hurst < 0.45:
                print(" (Mean-Reverting Structural Shifting) 📉")
            else:
                print(" (Standard Brownian Walk) 🎲")
        except Exception:
            print("  -> Structural Memory (Hurst, H)    : Estimation Error")
            
        skew = stats.skew(ari_vals)
        kurt = stats.kurtosis(ari_vals)
        print(f"  -> Distributional Typography       : Skewness = {skew:.2f} | Ex. Kurtosis = {kurt:.2f}")
        
    print("\n" + "="*95 + "\n")

if __name__ == "__main__":
    run_inter_crisis_audit("data/processed/master_methodology_trends.csv")