"""
src/analysis/volatility_integration.py
======================================
Ingests market-specific macro implied volatility indices (VIX, India VIX, VXEWZ)
and aligns them chronologically with master network metrics to address review constraints.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def integrate_implied_volatility_indices(
    trends_path: str | Path,
    vix_data_dir: str | Path,
    output_path: str | Path
) -> None:
    """
    Loads market-specific implied volatility index csv/parquet files and appends
    them directly to corresponding chronological network stability entries.
    """
    trends_path = Path(trends_path)
    trends_df = pd.read_csv(trends_path)
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    
    vix_dir = Path(vix_data_dir)
    markets = trends_df['market'].unique()
    
    # Map market tokens directly to market-specific implied volatility files
    # Download these files from Bloomberg, Yahoo Finance, or NSE/B3 directly.
    vix_files = {
        'sp500': 'vix_us_historical.csv',       # Standard CBOE VIX
        'nifty50': 'india_vix_historical.csv',   # NSE India VIX
        'bovespa': 'vxewz_bovespa_historical.csv' # CBOE Brazil ETF Volatility Index
    }
    
    aligned_dfs = []
    
    for market in markets:
        m_df = trends_df[trends_df['market'] == market].copy()
        
        file_name = vix_files.get(market)
        if not file_name:
            print(f"[WARNING] Unmapped market token encountered: {market}. Skipping.")
            aligned_dfs.append(m_df)
            continue
            
        file_path = vix_dir / file_name
        
        if not file_path.exists():
            print(f"[WARNING] Implied VIX file missing for {market.upper()} at: {file_path.absolute()}")
            print("           Falling back to old data frame slice without local index assignment.")
            aligned_dfs.append(m_df)
            continue
            
        # 1. Load the specific implied volatility file 
        # (Assuming file has 'date' and a closing value column like 'close' or 'vix')
        vix_df = pd.read_csv(file_path)
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        
        # Clean column structure: ensure it standardizes on a simple target string
        vix_df = vix_df.rename(columns={c: 'market_volatility' for c in vix_df.columns if c.lower() in ['close', 'vix_close', 'vix', 'value']})
        vix_df = vix_df[['date', 'market_volatility']].dropna()
        
        # 2. Sort chronologically to safely allow lookbacks / backward padding
        vix_df = vix_df.sort_values('date').reset_index(drop=True)
        
        # 3. Drop any pre-existing market_volatility columns in trends to prevent duplicate _x/_y tags
        if 'market_volatility' in m_df.columns:
            m_df = m_df.drop(columns=['market_volatility'])
            
        # 4. Merge using asof or left join to match trading day calendars perfectly
        m_df = pd.merge_asof(
            m_df.sort_values('date'), 
            vix_df, 
            on='date', 
            direction='backward' # Pads weekend or non-synchronized settlement gaps
        )
        
        aligned_dfs.append(m_df)
        print(f"[SUCCESS] Integrated local implied index data for market: {market.upper()}")

    # 5. Concatenate everything back together and overwrite output path
    master_extended = pd.concat(aligned_dfs, ignore_index=True)
    master_extended.to_csv(output_path, index=False)
    print(f"\n[PIPELINE COMPLETE] Master file cleanly saved to: {output_path}")

if __name__ == "__main__":
    integrate_implied_volatility_indices(
        trends_path="data/processed/master_methodology_trends.csv",
        vix_data_dir="data/raw/vix_indices",
        output_path="data/processed/master_methodology_trends.csv"
    )