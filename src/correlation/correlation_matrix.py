"""
src/correlation/correlation_matrix.py
======================================
Preprocesses raw asset prices, filters out assets with excessive gaps or
extended zero-return flatline streaks, and segments returns into rolling windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def clean_and_stationarize(market_name: str, raw_dir: str | None = None) -> pd.DataFrame:
    """
    Loads raw market prices and filters assets using centralized quality constraints
    (missing gap thresholds and maximum zero-return streak limits).
    """
    config = get_config()
    
    target_raw_dir = raw_dir or config["data"]["raw_dir"]
    max_missing_pct = config["quality"]["max_missing_pct"]
    zero_streak_limit = config["quality"]["zero_return_streak_flag"]
    
    raw_path = Path(target_raw_dir) / f"prices_{market_name}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file missing at {raw_path}")
        
    prices = pd.read_parquet(raw_path)
    total_days = len(prices)
    
    # 1. Primary Filter: Missing data threshold check
    missing_pct = prices.isna().sum() / total_days
    initial_valid = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    
    # 2. Secondary Filter: Zero-return flatline streak check
    clean_prices = prices[initial_valid].ffill().bfill()
    log_returns_temp = np.log(clean_prices / clean_prices.shift(1))
    
    final_valid_tickers = []
    for ticker in initial_valid:
        series = log_returns_temp[ticker]
        # Identify consecutive zero returns using a cumulative sum grouping
        is_zero = (series == 0.0).astype(int)
        max_streak = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).sum().max()
        
        if max_streak < zero_streak_limit:
            final_valid_tickers.append(ticker)
        else:
            logger.warning(
                "%s: Dropped %s due to an extensive flatline streak of %d days.",
                market_name.upper(), ticker, max_streak
            )
            
    logger.info(
        "%s Universe filtering: %d of %d tickers passed all quality checks.", 
        market_name.upper(), len(final_valid_tickers), prices.shape[1]
    )
                
    if not final_valid_tickers:
        raise RuntimeError(f"Zero assets passed quality filters for market: {market_name}")

    # Establish finalized cleaned returns matrix
    final_returns = log_returns_temp[final_valid_tickers].dropna()
    return final_returns


def slice_rolling_windows(returns_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """
    Segments log-returns into chronological chunks using centralized window parameters.
    """
    config = get_config()
    window_size = config["windows"]["primary"]
    step_size = config["windows"]["step"]
    
    total_days = len(returns_df)
    windows = []
    
    start_idx = 0
    while start_idx + window_size <= total_days:
        end_idx = start_idx + window_size
        window_chunk = returns_df.iloc[start_idx:end_idx]
        end_date = returns_df.index[end_idx - 1].strftime("%Y-%m-%d")
        
        windows.append((end_date, window_chunk))
        start_idx += step_size
        
    logger.info(
        "Segmented returns matrix into %d historical chunks (w=%d, step=%d).",
        len(windows), window_size, step_size
    )
    return windows