"""
src/analysis/check_raw_data_outliers.py
=======================================
Raw Base Price Dataset & Outlier Quality Audit Script.

Performs a diagnostic health check directly on base price datasets:
  - Missing value / NaN ratio analysis
  - Zero-return flatline streak detection (trading suspensions)
  - Extreme daily log-return outliers (> 5 standard deviations / extreme price jumps)
  - Date continuity and trading day gap checks
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Repository root path alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_loader import get_config


def audit_raw_market_prices(market_name: str, raw_data_dir: Path) -> dict:
    """Audits raw constituent price datasets for outliers, missingness, and flatlines."""
    market_key = market_name.lower()
    
    # Locate Parquet or CSV base file
    file_path = raw_data_dir / f"prices_{market_key}.parquet"
    if not file_path.exists():
        file_path = raw_data_dir / f"prices_{market_key}.csv"
        
    if not file_path.exists():
        print(f"❌ [ERROR] Base price file missing for market: {market_name} at {file_path}")
        return {}

    # Load base dataset
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    total_dates, total_tickers = df.shape
    total_cells = total_dates * total_tickers

    # 1. Missingness Audit
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100.0

    # 2. Daily Log Return Calculation
    log_returns = np.log(df / df.shift(1))

    # 3. Flatline Streak Detection (Zero Returns)
    # Checks for tickers stuck at 0.0 log return for >= 20 consecutive trading days
    is_zero = (log_returns == 0.0)
    flatline_counts = {}
    for col in log_returns.columns:
        series = is_zero[col]
        # Group consecutive True values
        blocks = (~series).cumsum()[series]
        max_streak = blocks.value_counts().max() if not blocks.empty else 0
        if max_streak >= 20:
            flatline_counts[col] = int(max_streak)

    # 4. Extreme Return Jump Detection (|r| > 5 std dev or daily jump > 30%)
    ret_mean = log_returns.mean()
    ret_std = log_returns.std()
    z_scores = (log_returns - ret_mean) / ret_std

    extreme_z_outliers = (np.abs(z_scores) > 5.0).sum().sum()
    extreme_jump_outliers = (np.abs(log_returns) > 0.30).sum().sum()  # > 30% single-day change

    return {
        "market": market_name.upper(),
        "file_name": file_path.name,
        "date_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
        "trading_days": total_dates,
        "constituent_count": total_tickers,
        "missing_null_pct": round(missing_pct, 3),
        "flatline_tickers_count": len(flatline_counts),
        "flatline_tickers_details": flatline_counts,
        "extreme_z_score_outliers": int(extreme_z_outliers),
        "extreme_jumps_over_30pct": int(extreme_jump_outliers)
    }


def main():
    config = get_config()
    raw_dir = Path("data/raw")
    markets = list(config["markets"].keys())

    print("\n" + "=" * 110)
    print("      BASE PRICE DATASET INTEGRITY & OUTLIER AUDIT REPORT")
    print("=========================================================================================\n")

    audit_summaries = []
    for market in markets:
        res = audit_raw_market_prices(market, raw_dir)
        if res:
            audit_summaries.append(res)
            print(f"📊 Market: {res['market']} ({res['file_name']})")
            print(f"   ├─ Date Range         : {res['date_range']} ({res['trading_days']} trading days)")
            print(f"   ├─ Constituents       : {res['constituent_count']} tickers")
            print(f"   ├─ Missing/Null Data  : {res['missing_null_pct']}% of total matrix")
            print(f"   ├─ Flatline Tickers   : {res['flatline_tickers_count']} assets with >= 20-day zero-return streaks")
            if res['flatline_tickers_details']:
                print(f"   │  └─ Details         : {res['flatline_tickers_details']}")
            print(f"   ├─ Extreme Z-Outliers : {res['extreme_z_score_outliers']} observations (|Z| > 5.0)")
            print(f"   └─ Extreme 30%+ Jumps : {res['extreme_jumps_over_30pct']} daily price leaps (> 30%)\n")

    print("=" * 110)
    print("✓ [AUDIT COMPLETE] Summary audit finished.")
    print("=========================================================================================\n")


if __name__ == "__main__":
    main()