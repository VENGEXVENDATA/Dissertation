"""
src/utils/fetch_vix_data.py
===========================
Utility to automatically download historical implied volatility indices from Yahoo Finance:
  1. ^VIX       -> S&P 500 Volatility Index (vix_us_historical.csv)
  2. ^INDIAVIX  -> India VIX (india_vix_historical.csv)
  3. ^VXEWZ     -> CBOE Brazil ETF Volatility Index (vxewz_bovespa_historical.csv)

Saves clean CSV files directly to data/raw/vix_indices/
"""

from pathlib import Path
import yfinance as yf
import pandas as pd
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def fetch_and_save_vix(output_dir: str | Path = "data/raw/vix_indices") -> None:
    vix_dir = Path(output_dir)
    vix_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = "2010-01-01"
    end_date = "2024-12-31"

    vix_targets = {
        "^VIX": "vix_us_historical.csv",
        "^INDIAVIX": "india_vix_historical.csv",
        "^VXEWZ": "vxewz_bovespa_historical.csv"
    }

    logger.info("Starting automatic download of implied volatility indices (2010-2024)...")

    for ticker, file_name in vix_targets.items():
        output_path = vix_dir / file_name
        logger.info("Fetching %s from Yahoo Finance...", ticker)
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                logger.error("Empty dataframe returned for %s", ticker)
                continue
                
            # Handle MultiIndex columns if returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                if "Close" in df.columns.levels[0]:
                    vix_series = df["Close"].squeeze()
                else:
                    vix_series = df.iloc[:, 0]
            elif "Close" in df.columns:
                vix_series = df["Close"]
            else:
                vix_series = df.iloc[:, 0]

            # Standardize column structure
            vix_df = pd.DataFrame({"date": vix_series.index, "market_volatility": vix_series.values})
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.strftime("%Y-%m-%d")
            vix_df = vix_df.dropna().sort_values("date").reset_index(drop=True)

            vix_df.to_csv(output_path, index=False)
            logger.info("Successfully saved %d rows to %s", len(vix_df), output_path)

        except Exception as e:
            logger.error("Failed downloading %s: %s", ticker, e)


if __name__ == "__main__":
    fetch_and_save_vix()