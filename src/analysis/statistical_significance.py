"""
src/analysis/statistical_significance.py
=========================================
Executes directional non-parametric Mann-Whitney U tests to evaluate 
the significance of ARI distribution drops across baseline and calendar-crisis regimes,
remediating serial autocorrelation bias via adaptive downsampling.
"""

from __future__ import annotations

import yaml
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path

def load_crisis_config() -> dict:
    """Locates and combines settings.yaml and crisis_periods.yaml dynamically."""
    script_dir = Path(__file__).resolve().parent
    root_dir = None
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / "config").is_dir():
            root_dir = parent
            break
    if root_dir is None:
        root_dir = Path("/home/s2843292/Dissertation/Dissertation")
        
    settings_path = root_dir / "config" / "settings.yaml"
    crisis_path = root_dir / "config" / "crisis_periods.yaml"
    
    with open(settings_path, "r") as f:
        config = yaml.safe_load(f) or {}
    with open(crisis_path, "r") as f:
        crisis_data = yaml.safe_load(f) or {}
        
    config.update(crisis_data)
    return config

def assign_crisis_phase(window_date, config: dict) -> str:
    """Labels a specific network timeline date relative to crisis events."""
    dt = pd.to_datetime(window_date)
    for crisis in config.get('crises', []):
        c_start = pd.to_datetime(crisis['crisis_start'])
        c_end = pd.to_datetime(crisis['crisis_end'])
        pre_start = c_start - pd.DateOffset(months=3)
        
        if c_start <= dt <= c_end:
            return f"In-Crisis ({crisis['short_name']})"
        elif pre_start <= dt < c_start:
            return f"Pre-Crisis ({crisis['short_name']})"
            
    for tranq in config.get('tranquil_periods', []):
        t_start = pd.to_datetime(tranq['start'])
        t_end = pd.to_datetime(tranq['end'])
        if t_start <= dt <= t_end:
            return "Tranquil Recovery"
            
    return "Baseline"

def evaluate_hypothesis_significance(data_path: str | Path) -> None:
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Apply structural data leak truncation filter
    corrupt_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
    df_clean = df[~corrupt_mask].reset_index(drop=True)
    
    # 2. Map calendar phases dynamically to generate the 'regime' column
    config = load_crisis_config()
    df_clean['regime'] = df_clean['date'].apply(lambda x: assign_crisis_phase(x, config))
    
    pivot_df = df_clean[['date', 'market', 'regime', 'ari_stability']].drop_duplicates()
    markets = pivot_df['market'].unique()
    results = []
    
    for market in markets:
        market_df = pivot_df[pivot_df['market'] == market].sort_values('date').reset_index(drop=True)
        
        # --- CRITICAL CORRECTION: Check and apply downsampling patch ---
        ari_raw = market_df['ari_stability'].dropna().values
        lag_1 = np.corrcoef(ari_raw[1:], ari_raw[:-1])[0, 1]
        
        if lag_1 > 0.50:
            print(f"[REMEDIATION] Autocorrelation severe ({lag_1:.4f}) for {market.upper()}. Applying downsampling (stride=4)...")
            market_df_remediated = market_df.iloc[::4].reset_index(drop=True)
        else:
            market_df_remediated = market_df
            
        # Isolate the independent historic baseline distribution
        baseline_ari = market_df_remediated[market_df_remediated['regime'] == 'Baseline']['ari_stability'].dropna()
        regimes = [r for r in market_df_remediated['regime'].unique() if r != 'Baseline']
        
        for regime in regimes:
            regime_ari = market_df_remediated[market_df_remediated['regime'] == regime]['ari_stability'].dropna()
            
            # Require at least 5 windows to ensure a meaningful test profile
            if len(regime_ari) < 5 or len(baseline_ari) < 5:
                continue
                
            # Directional Test: Is the crisis/pre-crisis distribution stochastically
            # SMALLER (dissolution) than the baseline distribution?
            stat, p_val = mannwhitneyu(regime_ari, baseline_ari, alternative='less')
            
            results.append({
                "Market": market.upper(),
                "Regime Segment": regime,
                "Obs.": len(regime_ari),
                "U-Stat": stat,
                "p-value": p_val,
                "Significance (Alpha=0.05)": "STAT. SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            })
            
    results_df = pd.DataFrame(results)
    print("\n" + "="*85)
    print("    REMEDIATED MANN-WHITNEY U TEST: REGIME STABILITY VS BASELINE")
    print("="*85)
    if not results_df.empty:
        print(results_df.to_string(index=False, formatters={"p-value": "{:.5f}".format}))
    else:
        print("[WARNING] No regimes met the minimum sample size constraint (N >= 5) after downsampling.")
    print("="*85 + "\n")

if __name__ == "__main__":
    evaluate_hypothesis_significance("data/processed/master_methodology_trends.csv")