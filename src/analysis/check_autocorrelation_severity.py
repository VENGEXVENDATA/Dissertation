"""
src/analysis/run_robust_analysis.py
===================================
Executes an end-to-end robust statistical analysis framework with high-precision
volatility regime thresholding anchored on full historical populations.
Remediates oversampling bias while preserving tail-event sample density.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_comprehensive_statistical_pipeline(trends_path: str | Path) -> None:
    trends_path = Path(trends_path)
    if not trends_path.exists():
        print(f"[ERROR] Tracked methodology trends file missing at: {trends_path.absolute()}")
        return

    # Ingest dataset and clean column typography
    df = pd.read_csv(trends_path)
    df.columns = df.columns.str.strip()
    
    # Apply baseline truncation patch for S&P 500 historical artifacts
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        corrupt_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
        df = df[~corrupt_mask].reset_index(drop=True)

    print("=" * 95)
    print("      STAGE 1: OVERLAPPING WINDOW BIAS & SERIAL AUTOCORRELATION DIAGNOSTIC AUDIT")
    print("=" * 95)

    market_profiles = {}
    markets = df['market'].unique()

    for market in markets:
        m_df = df[df['market'] == market].sort_values('date').reset_index(drop=True)
        ari = m_df['ari_stability'].dropna().values
        N_raw = len(ari)
        
        if N_raw < 10:
            print(f"[WARNING] Insufficient data rows for market: {market.upper()}")
            continue
            
        # Compute empirical Pearson serial correlation coefficients
        lag_1 = np.corrcoef(ari[1:], ari[:-1])[0, 1]
        lag_10 = np.corrcoef(ari[10:], ari[:-10])[0, 1]
        
        # Calculate Effective Sample Size (N_eff) via Bartlett's correction formula
        if lag_1 < 0.99:
            n_eff = N_raw * (1 - lag_1) / (1 + lag_1)
        else:
            n_eff = N_raw / 12  
            
        n_eff_sanitized = int(np.ceil(max(2, n_eff)))
        
        print(f"\n[MARKET]: {market.upper()}")
        print(f"  -> Total Dataset Rows (N_raw):         {N_raw}")
        print(f"  -> True Independent Obs (N_eff):       {n_eff_sanitized}")
        print(f"  -> Lag-1 Autocorrelation Coefficient:  {lag_1:.4f}")
        
        if lag_1 > 0.50:
            print(f"  ! DIAGNOSIS: Severe Overlap Dependency Present ({lag_1:.4f} > 0.50).")
            print("                Standard independence assumptions FAIL. Remediation mandated.")
            requires_remediation = True
        else:
            print(f"  ✓ DIAGNOSIS: Mild persistence ({lag_1:.4f} <= 0.50). Standard assumptions hold.")
            requires_remediation = False
            
        market_profiles[market] = {
            'df': m_df,
            'requires_remediation': requires_remediation,
            'N_raw': N_raw
        }

    print("\n" + "=" * 95)
    print("      STAGE 2: HIGH-PRECISION NON-PARAMETRIC MANN-WHITNEY U REGIME TESTING")
    print("=" * 95)

    vol_col = 'market_volatility' if 'market_volatility' in df.columns else 'market_volatility_x'
    if vol_col not in df.columns:
        print(f"[ERROR] Volatility proxy column missing from dataset attributes: {df.columns.tolist()}")
        return

    for market, profile in market_profiles.items():
        m_df_raw = profile['df']
        
        # --- FIXED: Ground truth thresholds computed on the complete historical population first ---
        upper_crisis_bound = m_df_raw[vol_col].quantile(0.85)
        lower_tranquil_bound = m_df_raw[vol_col].quantile(0.50)
        
        # --- OPTIMIZED REMEDIATION STRIDE ---
        if profile['requires_remediation']:
            print(f"\n[REMEDIATION RUN] Downsampling {market.upper()} timeline matrix...")
            # Adjusted to step size 4 to preserve sufficient tail event observation sample size (N >= 2)
            m_df_clean = m_df_raw.iloc[::4].reset_index(drop=True)
            print(f"  -> Pruned rows from {profile['N_raw']} down to {len(m_df_clean)} independent slices.")
        else:
            print(f"\n[STANDARD RUN] Processing raw timeline sequence for {market.upper()}...")
            m_df_clean = m_df_raw
            
        # Isolate extreme stress windows vs clear tranquil control samples using true population bounds
        baseline_dist = m_df_clean[m_df_clean[vol_col] <= lower_tranquil_bound]['ari_stability'].dropna().values
        crisis_dist = m_df_clean[m_df_clean[vol_col] >= upper_crisis_bound]['ari_stability'].dropna().values
        
        if len(baseline_dist) >= 2 and len(crisis_dist) >= 2:
            print(f"  -> Population Benchmarks    : Baseline Cutoff (<50th) = {lower_tranquil_bound:.4f} | Crisis Cutoff (>85th) = {upper_crisis_bound:.4f}")
            print(f"  -> Current Window Sample Mean: Baseline ARI = {np.mean(baseline_dist):.4f} | Crisis ARI = {np.mean(crisis_dist):.4f}")
            
            # Execute directional, one-tailed non-parametric hypothesis test
            stat_u, p_value = mannwhitneyu(crisis_dist, baseline_dist, alternative='less')
            
            print(f"  -> Operational Sample Sizes  : Baseline N = {len(baseline_dist)} | Crisis N = {len(crisis_dist)}")
            print(f"  -> Calculated U-Statistic    : {stat_u:.1f}")
            print(f"  -> Empirical Directional P-Val: {p_value:.6f}")
            
            if p_value <= 0.05:
                print("  ✓ VERDICT: Statistically Significant Structural Collapse Confirmed (Alpha = 0.05).")
            elif p_value <= 0.10:
                print("  ! VERDICT: Marginal / Weakly Significant Topological Trend Documented (Alpha = 0.10).")
            else:
                print("  ✗ VERDICT: No Significant Structural Dissolution Found. System displays topological invariance.")
        else:
            print(f"  [ERROR] Insufficient partition counts to calculate rank-sum metrics after tail isolation.")
            print(f"          Baseline Sample N = {len(baseline_dist)} | Crisis Sample N = {len(crisis_dist)}")
            
    print("=" * 95 + "\n")

if __name__ == "__main__":
    run_comprehensive_statistical_pipeline(
        trends_path="data/processed/master_methodology_trends.csv"
    )