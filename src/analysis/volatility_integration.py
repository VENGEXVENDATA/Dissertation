"""
src/analysis/volatility_integration.py
======================================
Integrates macro volatility metrics with master temporal network trends.
For markets where implied VIX data is incomplete or unavailable (e.g., Bovespa),
it dynamically calculates a 21-day Annualized Rolling Realized Volatility proxy
from daily index return series.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.fetch_vix_data import fetch_and_save_vix


def calculate_realized_volatility(raw_price_path: Path, window: int = 21) -> pd.DataFrame:
    """
    Computes 21-day annualized rolling realized volatility from daily price data.
    Formula: Annualized Std Dev = Rolling Std Dev(log_returns) * sqrt(252) * 100
    """
    if raw_price_path.suffix == ".parquet":
        prices_df = pd.read_parquet(raw_price_path)
    else:
        prices_df = pd.read_csv(raw_price_path)

    # Convert date and set index with explicit nanosecond type
    if "date" in prices_df.columns:
        prices_df["date"] = pd.to_datetime(prices_df["date"]).astype("datetime64[ns]")
        prices_df = prices_df.set_index("date")
    else:
        prices_df.index = pd.to_datetime(prices_df.index).astype("datetime64[ns]")

    # Compute market-wide mean daily log return across available constituents
    log_returns = np.log(prices_df / prices_df.shift(1))
    market_return = log_returns.mean(axis=1)

    # 21-day rolling standard deviation annualized (252 trading days)
    realized_vol = market_return.rolling(window=window).std() * np.sqrt(252) * 100.0

    vol_df = pd.DataFrame({
        "date": pd.to_datetime(realized_vol.index).astype("datetime64[ns]"),
        "market_volatility": realized_vol.values
    }).dropna().sort_values("date").reset_index(drop=True)

    return vol_df


def integrate_implied_volatility_indices(
    trends_path: str | Path,
    vix_data_dir: str | Path,
    raw_data_dir: str | Path,
    output_path: str | Path
) -> None:
    trends_path = Path(trends_path)
    if not trends_path.exists():
        raise FileNotFoundError(f"[CRITICAL ERROR] Master trends file missing at: {trends_path.absolute()}")

    vix_dir = Path(vix_data_dir)
    raw_dir = Path(raw_data_dir)
    
    # Auto-fetch VIX if missing
    vix_files = {
        'sp500': 'vix_us_historical.csv',
        'nifty50': 'india_vix_historical.csv',
        'bovespa': 'vxewz_bovespa_historical.csv'
    }
    
    missing_files = [fn for fn in vix_files.values() if not (vix_dir / fn).exists()]
    if missing_files:
        print(f"[NOTICE] Implied VIX files missing ({missing_files}). Attempting auto-fetch...")
        try:
            fetch_and_save_vix(vix_dir)
        except Exception as e:
            print(f"[WARNING] VIX download issue: {e}")

    trends_df = pd.read_csv(trends_path)
    markets = trends_df['market'].unique()
    
    aligned_dfs = []
    
    for market in markets:
        m_df = trends_df[trends_df['market'] == market].copy()
        
        # Strip existing volatility columns to avoid duplicate suffixes (_x, _y)
        cols_to_drop = [c for c in m_df.columns if 'volatility' in c.lower()]
        if cols_to_drop:
            m_df = m_df.drop(columns=cols_to_drop)
            
        file_name = vix_files.get(str(market).lower())
        file_path = vix_dir / file_name if file_name else None
        
        use_realized_fallback = False
        vix_df = pd.DataFrame()

        # Check if implied VIX file exists and has sufficient data (>1000 rows)
        if file_path and file_path.exists():
            vix_df = pd.read_csv(file_path)
            if len(vix_df) < 1000:  # If sparse (e.g., VXEWZ has only ~380 rows)
                print(f"[NOTICE] {str(market).upper()} implied VIX file has insufficient history ({len(vix_df)} rows). Using 21-day Realized Volatility.")
                use_realized_fallback = True
            else:
                vix_df['date'] = pd.to_datetime(vix_df['date'])
                vix_df = vix_df.rename(columns={
                    c: 'market_volatility' for c in vix_df.columns if c.lower() in ['close', 'vix_close', 'vix', 'value', 'market_volatility']
                })
                vix_df = vix_df[['date', 'market_volatility']].dropna()
        else:
            use_realized_fallback = True

        # Fallback to 21-day Realized Volatility calculation from raw price Parquet/CSV
        if use_realized_fallback:
            raw_price_file = raw_dir / f"prices_{str(market).lower()}.parquet"
            if not raw_price_file.exists():
                raw_price_file = raw_dir / f"prices_{str(market).lower()}.csv"

            if raw_price_file.exists():
                print(f"[CALCULATING] Computing 21-day Realized Volatility proxy for {str(market).upper()} from {raw_price_file.name}...")
                vix_df = calculate_realized_volatility(raw_price_file, window=21)
            else:
                raise FileNotFoundError(f"[ERROR] Cannot find raw price file for {str(market).upper()} at {raw_price_file}")

        # FORCE EXACT MATCHING DTYPE (datetime64[ns]) BEFORE MERGE
        m_df['date'] = pd.to_datetime(m_df['date']).astype("datetime64[ns]")
        vix_df['date'] = pd.to_datetime(vix_df['date']).astype("datetime64[ns]")
        
        vix_df = vix_df.sort_values('date').reset_index(drop=True)
        m_df = m_df.sort_values('date').reset_index(drop=True)
        
        # Merge via backward fill matching calendar dates
        m_df = pd.merge_asof(
            m_df, 
            vix_df, 
            on='date', 
            direction='backward'
        )
        
        aligned_dfs.append(m_df)
        print(f"[SUCCESS] Integrated volatility proxy for market: {str(market).upper()} (Rows: {len(m_df)})")

    master_extended = pd.concat(aligned_dfs, ignore_index=True)
    master_extended.to_csv(output_path, index=False)
    print(f"\n[PIPELINE COMPLETE] Master trends saved with complete volatility data to: {output_path}")


if __name__ == "__main__":
    integrate_implied_volatility_indices(
        trends_path="data/processed/master_methodology_trends.csv",
        vix_data_dir="data/raw/vix_indices",
        raw_data_dir="data/raw",
        output_path="data/processed/master_methodology_trends.csv"
    )