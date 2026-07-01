"""
pipelines/run_acquisition.py
=============================
Entry point to download price data for all three markets.
Run from project root:  python pipelines/run_acquisition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition.constituent_loader import get_tickers_for_market
from src.acquisition.downloader import download_market_prices
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger
from src.utils.timer import stage_timer

logger = get_logger("pipelines.run_acquisition")


@stage_timer("Full Data Acquisition Pipeline")
def run_acquisition() -> None:
    """Download price data for all configured markets."""
    cfg = get_config()

    start_date = cfg["data"]["start_date"]
    end_date   = cfg["data"]["end_date"]
    output_dir = cfg["data"]["raw_dir"]
    markets    = list(cfg["markets"].keys())

    logger.info("Markets to download: %s", markets)
    logger.info("Date range: %s to %s", start_date, end_date)

    results: dict[str, tuple[int, int]] = {}

    for market in markets:
        logger.info("=" * 60)
        logger.info("Starting download: %s", market.upper())
        logger.info("=" * 60)

        try:
            tickers = get_tickers_for_market(market)
            logger.info("%s: %d tickers identified", market, len(tickers))

            prices = download_market_prices(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                market_name=market,
                output_dir=output_dir,
            )

            results[market] = prices.shape
            logger.info("%s complete: %d dates x %d tickers",
                        market, prices.shape[0], prices.shape[1])

        except Exception as e:
            logger.error("FAILED for %s: %s", market, e)
            results[market] = (0, 0)

    # Summary — ASCII only, no emoji (Windows CP1252 safe)
    logger.info("")
    logger.info("=" * 60)
    logger.info("ACQUISITION SUMMARY")
    logger.info("=" * 60)
    for market, shape in results.items():
        status = "OK" if shape[0] > 0 else "FAILED"
        logger.info("[%s] %s: %d dates x %d tickers",
                    status, market.upper(), shape[0], shape[1])


if __name__ == "__main__":
    run_acquisition()
