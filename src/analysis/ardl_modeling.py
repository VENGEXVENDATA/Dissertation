"""
src/analysis/ardl_modeling.py
=============================
Estimates dynamically balanced Autoregressive Distributed Lag (ARDL) models 
informed by peak empirical cross-correlations, resolving unit root bias 
and residual autocorrelation for thesis-grade output.
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

def run_ardl_pipeline(trends_path: str | Path) -> None:
    """
    Evaluates unit roots and runs dynamically balanced ARDL frameworks 
    where directions and lags are customized to empirical market signatures.
    """
    trends_path = Path(trends_path)
    if not trends_path.exists():
        print(f"[ERROR] Trends file missing at: {trends_path.absolute()}")
        return

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
    
    # Configure each market to align with your empirical CCF findings
    # For BOVESPA: Network leads Volatility (Y = Vol, X = ARI)
    # For NIFTY50 & S&P 500: Volatility leads Network (Y = ARI, X = Vol)
    market_configs = {
        'sp500': {
            'dep_var': 'ari_stability',
            'indep_var': vol_col,
            'max_lags': 12,  # CCF peak lag is 12 days
            'label': "SP500 (Volatility Impact on Network Stability)"
        },
        'nifty50': {
            'dep_var': 'ari_stability',
            'indep_var': vol_col,
            'max_lags': 4,   # CCF peak lag is 4 days
            'label': "NIFTY50 (Volatility Impact on Network Stability)"
        },
        'bovespa': {
            'dep_var': vol_col,
            'indep_var': 'ari_stability',
            'max_lags': 2,   # CCF peak lag is -1 day; check short lag structure
            'label': "BOVESPA (Network Stability Leading Volatility Feedback)"
        }
    }

    print("\n" + "="*95)
    print("      DISSERTATION ECONOMETRIC PIPELINE: DYNAMIC ARDL MODELING")
    print("=====================================================================================")
    
    for market, config in market_configs.items():
        m_data = pivot_df[pivot_df['market'] == market].dropna().copy()
        
        if len(m_data) < 50:
            continue
            
        print(f"\n📊 ESTIMATING MODEL FOR MARKET: {config['label'].upper()}")
        print("-" * 85)
        
        # 1. Stationarity Audit (ADF Tests)
        p_val_dep = adfuller(m_data[config['dep_var']])[1]
        p_val_ind = adfuller(m_data[config['indep_var']])[1]
        
        dep_is_stationary = p_val_dep <= 0.05
        ind_is_stationary = p_val_ind <= 0.05
        
        print(f"  -> Stationarity check (ADF p): {config['dep_var']} = {p_val_dep:.4f} | {config['indep_var']} = {p_val_ind:.4f}")
        
        # 2. Dynamic Variable Differencing to avoid spurious regressions
        y_series = m_data[config['dep_var']].copy()
        x_series = m_data[config['indep_var']].copy()
        
        dep_name = config['dep_var']
        ind_name = config['indep_var']
        
        if not dep_is_stationary:
            print(f"  -> [ADJUSTMENT] Dependent variable '{config['dep_var']}' is non-stationary. First-differenced.")
            y_series = y_series.diff()
            dep_name = f"d_{config['dep_var']}"
            
        if not ind_is_stationary:
            print(f"  -> [ADJUSTMENT] Independent variable '{config['indep_var']}' is non-stationary. First-differenced.")
            x_series = x_series.diff()
            ind_name = f"d_{config['indep_var']}"

        # Combine processed vectors into model frame
        reg_df = pd.DataFrame({dep_name: y_series, ind_name: x_series}).dropna()
        
        # 3. Construct ARDL Regressor Columns (Lagged Dependent and Independent variables)
        X_dict = {}
        
        # Lagged dependent features to clean residual serial correlation
        for lag in range(1, 3):  # Standard baseline AR(2) process
            X_dict[f'{dep_name}_lag_{lag}'] = reg_df[dep_name].shift(lag)
            
        # Lagged explanatory features matching the CCF lead-lag envelope
        for lag in range(1, config['max_lags'] + 1):
            X_dict[f'{ind_name}_lag_{lag}'] = reg_df[ind_name].shift(lag)
            
        X_df = pd.DataFrame(X_dict, index=reg_df.index)
        X_df[dep_name] = reg_df[dep_name]
        X_df[ind_name] = reg_df[ind_name]
        X_df = X_df.dropna()
        
        # Isolate targets
        y_model = X_df[dep_name]
        X_features = X_df[[c for c in X_df.columns if c != dep_name]]
        X_features = sm.add_constant(X_features)
        
        # Fit Ordinary Least Squares
        model = sm.OLS(y_model, X_features)
        res = model.fit()
        
        # 4. Diagnostic & Output Summary
        print(f"  -> Model Fit Summary: R-squared = {res.rsquared:.4f} | Adj. R-squared = {res.rsquared_adj:.4f}")
        
        # Durbin-Watson diagnostic (values close to 2.0 mean residual autocorrelation is resolved)
        dw_stat = sm.stats.stattools.durbin_watson(res.resid)
        print(f"  -> Residual Durbin-Watson statistic: {dw_stat:.4f} (Target ~ 2.0)")
        
        print("\nSignificant Impact Trajectories (Alpha = 0.10):")
        has_significant = False
        for param, coef in res.params.items():
            p_val = res.pvalues[param]
            if p_val < 0.10 and param != 'const':
                print(f"   -> {param:<25} | Coef: {coef:>8.4f} | p-value: {p_val:.5f} (*)")
                has_significant = True
        if not has_significant:
            print("   -> No individual lag coefficients meet significance thresholds.")
            
    print("\n" + "="*95 + "\n")

if __name__ == "__main__":
    master_path = "data/processed/master_methodology_trends.csv"
    run_ardl_pipeline(master_path)
    