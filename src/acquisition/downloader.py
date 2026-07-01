"""
src/acquisition/downloader.py
==============================
Downloads adjusted close price data from Yahoo Finance via yfinance.
Implements retry logic, rate limiting, and saves to Parquet format.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.logging_config import get_logger
from src.utils.timer import stage_timer

logger = get_logger(__name__)


def download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[pd.Series]:
    """
    Download adjusted close prices for a single ticker with retry logic.
    """
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if data.empty:
                logger.warning("No data returned for %s (attempt %d)", ticker, attempt)
                time.sleep(retry_delay)
                continue

            # Extract Close column
            if "Close" in data.columns:
                series = data["Close"]
            else:
                logger.warning("No Close column for %s", ticker)
                return None

            # Flatten MultiIndex if it occurs
            if isinstance(series, pd.DataFrame):
                series = series.squeeze()
                if isinstance(series, pd.DataFrame):  # Squeeze guard
                    series = series.iloc[:, 0]

            series.name = ticker
            logger.debug("Downloaded %s: %d rows", ticker, len(series))
            return series

        except Exception as e:
            logger.warning("Attempt %d failed for %s: %s", attempt, ticker, e)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    logger.error("All %d attempts failed for %s — skipping", max_retries, ticker)
    return None


@stage_timer("Batch Market Download")
def download_market_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    market_name: str,
    output_dir: str = "data/raw",
    batch_size: int = 25,  # Reduced batch size slightly for tighter API window limits
    batch_delay: float = 3.0,
) -> pd.DataFrame:
    """
    Download price data for all tickers in a market.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"prices_{market_name}.parquet"

    if output_path.exists():
        logger.info("Found existing data at %s — loading", output_path)
        return pd.read_parquet(output_path)

    logger.info("Downloading %d tickers for %s (%s to %s)",
                len(tickers), market_name, start_date, end_date)

    all_prices: dict[str, pd.Series] = {}
    failed_tickers: list[str] = []

    # Split into batches
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch_num, batch in enumerate(tqdm(batches, desc=f"Downloading {market_name}")):
        try:
            # Use single-thread for international exchanges to avoid dropped data frames
            is_intl = any(".NS" in t or ".SA" in t for t in batch)
            
            raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False if is_intl else True,
                group_by="ticker",
            )

            if raw.empty:
                logger.warning("Empty batch %d — falling back to individual processing", batch_num)
                for ticker in batch:
                    series = download_single_ticker(ticker, start_date, end_date)
                    if series is not None:
                        all_prices[ticker] = series
                    else:
                        failed_tickers.append(ticker)
                continue

            # Extract columns defensively
            for ticker in batch:
                try:
                    series = None
                    if len(batch) == 1:
                        if "Close" in raw.columns:
                            series = raw["Close"].squeeze()
                    else:
                        # Check if the level exists in the columns structure
                        available_tickers = raw.columns.levels[0] if isinstance(raw.columns, pd.MultiIndex) else raw.columns
                        if ticker in available_tickers:
                            if "Close" in raw[ticker].columns:
                                series = raw[ticker]["Close"].squeeze()

                    if series is None or isinstance(series, pd.DataFrame) or series.dropna().empty:
                        logger.warning("Data check failed for %s — processing individually", ticker)
                        fallback_series = download_single_ticker(ticker, start_date, end_date)
                        if fallback_series is not None and not fallback_series.dropna().empty:
                            all_prices[ticker] = fallback_series
                        else:
                            failed_tickers.append(ticker)
                        continue

                    series.name = ticker
                    all_prices[ticker] = series

                except Exception as ex:
                    logger.warning("Error parsing %s in batch: %s. Reverting to single lookup.", ticker, ex)
                    fallback_series = download_single_ticker(ticker, start_date, end_date)
                    if fallback_series is not None:
                        all_prices[ticker] = fallback_series
                    else:
                        failed_tickers.append(ticker)

        except Exception as e:
            logger.error("Batch %d failed completely: %s — falling back to loop processing", batch_num, e)
            for ticker in batch:
                series = download_single_ticker(ticker, start_date, end_date)
                if series is not None:
                    all_prices[ticker] = series
                else:
                    failed_tickers.append(ticker)

        if batch_num < len(batches) - 1:
            time.sleep(batch_delay)

    if not all_prices:
        raise RuntimeError(f"No data successfully extracted for market target: {market_name}")

    # Align indexes cleanly across all variables
    prices_df = pd.DataFrame(all_prices)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.index.name = "date"
    prices_df.sort_index(inplace=True)

    prices_df.to_parquet(output_path)
    logger.info(
        "Saved %s: shape=%s, failed=%d tickers, path=%s",
        market_name, prices_df.shape, len(failed_tickers), output_path
    )

    if failed_tickers:
        failed_path = Path(output_dir) / f"failed_tickers_{market_name}.txt"
        failed_path.write_text("\n".join(failed_tickers))
        logger.warning("Failed log written to %s", failed_path)

    return prices_df
