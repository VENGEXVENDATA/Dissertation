"""
src/analysis/volatility_integration.py
======================================
Ingests separate market raw price parquet data files, calculates 
volatility proxies, and aligns them chronologically with master network metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def integrate_volatility_proxies(
    trends_path: str | Path,
    raw_data_dir: str | Path,
    output_path: str | Path
) -> None:
    """
    Computes daily price-based volatility from separate asset parquet files
    and appends it directly to corresponding chronological target dates.
    """
    trends_df = pd.read_csv(trends_path)
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    
    raw_dir = Path(raw_data_dir)
    volatility_records = []
    markets = trends_df['market'].unique()
    
    # Map market tokens to your actual raw file configurations
    market_files = {
        'sp500': 'prices_sp500.parquet',
        'nifty50': 'prices_nifty50.parquet',
        'bovespa': 'prices_bovespa.parquet'
    }
    
    for market in markets:
        file_path = raw_dir / market_files.get(market, f"prices_{market}.parquet")
        
        if not file_path.exists():
            print(f"[WARNING] Raw pricing file missing for {market}: {file_path.absolute()}. Skipping volatility calculation.")
            continue
            
        # 1. Load the specific raw pricing asset data frame
        prices_df = pd.read_parquet(file_path)
        prices_df.index = pd.to_datetime(prices_df.index)
        
        # 2. Extract a market proxy series using the mean of all liquid asset returns
        # This gives an excellent internal benchmark return matrix for the index
        prices_df = prices_df.ffill().bfill()
        daily_returns = np.log(prices_df / prices_df.shift(1)).dropna()
        market_proxy_returns = daily_returns.mean(axis=1)
        
        # 3. Calculate 21-day rolling historical volatility (annualized)
        hist_vol = market_proxy_returns.rolling(window=21).std() * np.sqrt(252)
        
        m_dates = trends_df[trends_df['market'] == market]['date'].unique()
        for dt in m_dates:
            dt_ts = pd.to_datetime(dt)
            # Find the closest trading date available in the index matrix
            closest_idx = hist_vol.index.get_indexer([dt_ts], method='pad')[0]
            if closest_idx == -1:
                continue
            closest_date = hist_vol.index[closest_idx]
            vol_val = hist_vol.loc[closest_date]
            
            volatility_records.append({
                "date": dt_ts,
                "market": market,
                "market_volatility": vol_val
            })
            
    if not volatility_records:
        print("[ERROR] Zero volatility matching indices computed. Check file paths.")
        return
        
    vol_df = pd.DataFrame(volatility_records)
    
    # 4. Merge cleanly back into your master methodology tracking file
    master_extended = pd.merge(trends_df, vol_df, on=['date', 'market'], how='left')
    
    master_extended.to_csv(output_path, index=False)
    print(f"[SUCCESS] Price volatility integration complete. Master sheet saved: {output_path}")

if __name__ == "__main__":
    integrate_volatility_proxies(
        trends_path="data/processed/master_methodology_trends.csv",
        raw_data_dir="data/raw",
        output_path="data/processed/master_methodology_trends.csv"
    )