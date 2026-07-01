"""
src/analysis/ardl_modeling.py
=============================
Estimates a Distributed Lag model using OLS to evaluate short-run and 
predictive impacts between network stability and price-based volatility.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_distributed_lag_pipeline(trends_path: str | Path) -> None:
    """
    Executes unit root testing and fits an explicit distributed lag regression 
    individually for each target stock market index.
    """
    df = pd.read_csv(trends_path)
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()
    
    vol_col = 'market_volatility_x' if 'market_volatility_x' in df.columns else 'market_volatility'
    if vol_col not in df.columns:
        print("[ERROR] Volatility data column missing. Run volatility integration first.")
        return

    # S&P 500 Truncation Fix
    corrupt_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
    df_clean = df[~corrupt_mask].reset_index(drop=True)
    
    pivot_df = df_clean[['date', 'market', 'ari_stability', vol_col]].drop_duplicates().sort_values('date')
    markets = pivot_df['market'].unique()
    
    print("\n" + "="*85)
    print("      DISSERTATION ECONOMETRIC PIPELINE: DISTRIBUTED LAG EVALUATION")
    print("=====================================================================================")
    
    for market in markets:
        m_data = pivot_df[pivot_df['market'] == market].dropna().copy()
        
        if len(m_data) < 50:
            continue
            
        print(f"\n📈 ESTIMATING DISTRIBUTED LAG REGRESSION FOR MARKET: {market.upper()}")
        print("-" * 75)
        
        # 1. Confirm stationarity via ADF p-values
        adf_y = adfuller(m_data[vol_col])[1]
        adf_x = adfuller(m_data['ari_stability'])[1]
        
        # 2. Build explicit lag matrix features for your network metric (Hypothesis 2)
        regression_df = pd.DataFrame({
            'Volatility': m_data[vol_col],
            'Network_ARI': m_data['ari_stability']
        })
        
        # Generate 5 consecutive lags matching our short-term lead horizons
        for lag in range(1, 6):
            regression_df[f'Network_ARI_Lag_{lag}'] = regression_df['Network_ARI'].shift(lag)
            
        regression_df = regression_df.dropna()
        
        # Isolate target variables and add regression constant
        X = regression_df[[c for c in regression_df.columns if c != 'Volatility']]
        X = sm.add_constant(X)
        y = regression_df['Volatility']
        
        # Fit Ordinary Least Squares
        model = sm.OLS(y, X)
        res = model.fit()
        
        # Output clean concise results block
        print(f"ADF Check: Volatility p={adf_y:.4f} | Network p={adf_x:.4f}")
        print(f"Model Fit Summary: R-squared = {res.rsquared:.4f} | Adj. R-squared = {res.rsquared_adj:.4f}")
        print("\nSignificant Feature Impact Paths (Alpha = 0.10):")
        
        has_significant = False
        for param, coef in res.params.items():
            p_val = res.pvalues[param]
            if p_val < 0.10 and param != 'const':
                print(f"  -> {param:<20} | Coef: {coef:>8.4f} | p-value: {p_val:.5f} (*)")
                has_significant = True
        if not has_significant:
            print("  -> No individual feature lags cross the significance threshold.")
            
    print("\n" + "="*85 + "\n")

if __name__ == "__main__":
    master_path = "data/processed/master_methodology_trends.csv"
    run_distributed_lag_pipeline(master_path)