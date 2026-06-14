"""
src/correlation/correlation_matrix.py
======================================
Preprocesses full raw price assets, drops tickers exceeding 5% missing thresholds,
and segments data frames into rolling window slices using an approved 21-day step size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def clean_and_stationarize(market_name: str, raw_dir: str = "data/raw", max_missing_pct: float = 0.05) -> pd.DataFrame:
    """
    Loads the full authentic asset pool for a market and filters out entries exceeding missing limits.
    Returns a clean log-return matrix.
    """
    raw_path = Path(raw_dir) / f"prices_{market_name}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file missing at {raw_path}")
        
    prices = pd.read_parquet(raw_path)
    total_days = len(prices)
    
    # Check data completeness metrics over full series length
    missing_pct = prices.isna().sum() / total_days
    valid_tickers = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    
    logger.info("%s Universe filtering: %d of %d tickers passed the < %2.0f%% gap filter.", 
                market_name.upper(), len(valid_tickers), prices.shape[1], max_missing_pct * 100)
                
    # Forward-fill / backward-fill localized missing points
    clean_prices = prices[valid_tickers].ffill().bfill()
    
    # Calculate daily stationary log-returns
    log_returns = np.log(clean_prices / clean_prices.shift(1)).dropna()
    return log_returns


def slice_rolling_windows(returns_df: pd.DataFrame, window_size: int = 126, step_size: int = 21) -> list[tuple[str, pd.DataFrame]]:
    """
    Segments log-returns into window chunks using a 126-day window and 21-day steps.
    """
    total_days = len(returns_df)
    windows = []
    
    start_idx = 0
    while start_idx + window_size <= total_days:
        end_idx = start_idx + window_size
        window_chunk = returns_df.iloc[start_idx:end_idx]
        end_date = returns_df.index[end_idx - 1].strftime("%Y-%m-%d")
        
        windows.append((end_date, window_chunk))
        start_idx += step_size
        
    return windows


def compute_mantegna_distances(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a Pearson correlation matrix into Euclidean distances:
    d_ij = sqrt(2 * (1 - rho_ij))
    """
    corr_filled = corr_matrix.fillna(0.0)  # Handle any flat-pricing variance anomalies
    rho = np.clip(corr_filled.values, -1.0, 1.0)
    distances = np.sqrt(2.0 * (1.0 - rho))
    return pd.DataFrame(distances, index=corr_matrix.index, columns=corr_matrix.columns)
