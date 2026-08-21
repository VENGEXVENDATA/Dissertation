"""
src/analysis/compare_community_algorithms.py
=============================================
Master Algorithm-Wise Comparison & Evaluation Script.

Runs all 6 community detection algorithms across 5 theoretical families on cached PMFGs:
  1. Modularity Maximization: Louvain (Consensus 100x), Leiden (Consensus 100x)
  2. Information Theory / Flow: Infomap
  3. Local Consensus / Dynamics: Label Propagation Algorithm (LPA)
  4. Spectral / Linear Algebra: Spectral Clustering (Laplacian Eigenvectors)
  5. Hierarchical / Distance Metric: Hierarchical Agglomerative Clustering (HAC)

Evaluates and compiles:
  - Baseline Mean Stability (μ) and Volatility (σ)
  - Cross-correlation (r) with the primary Louvain benchmark trajectory
  - Econometric ARDL Fit (R² and Durbin-Watson statistic using master_methodology_trends.csv)
  - Execution time profiling

Outputs:
  - CSV report: data/processed/community_algorithm_comparison_matrix.csv
  - Formatted LaTeX code for Dissertation Chapter 4
"""

from __future__ import annotations

import sys
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.community.consensus_louvain import track_timeline_stability as track_louvain
from src.community.consensus_leiden import track_leiden_timeline_stability as track_leiden
from src.community.alternative_algorithms import track_alternative_timeline_stability as track_alternative
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# ECONOMETRIC HELPER FUNCTIONS
# =============================================================================

def evaluate_ardl_fit(ari_series: pd.Series, vix_series: pd.Series, max_lags: int = 6) -> tuple[float, float]:
    """
    Fits a dynamic ARDL(2, max_lags) regression on first-differenced/stationary series
    and returns (R_squared, Durbin_Watson_stat).
    """
    df_reg = pd.DataFrame({"y": ari_series, "x": vix_series}).dropna()
    if len(df_reg) < 30:
        return np.nan, np.nan

    # Perform ADF unit root test and first-difference if non-stationary (p > 0.05)
    y = df_reg["y"].diff() if adfuller(df_reg["y"])[1] > 0.05 else df_reg["y"]
    x = df_reg["x"].diff() if adfuller(df_reg["x"])[1] > 0.05 else df_reg["x"]

    reg_data = pd.DataFrame({"y": y, "x": x}).dropna()

    X_dict = {}
    for lag in range(1, 3):
        X_dict[f"y_lag_{lag}"] = reg_data["y"].shift(lag)
    for lag in range(1, max_lags + 1):
        X_dict[f"x_lag_{lag}"] = reg_data["x"].shift(lag)

    X_df = pd.DataFrame(X_dict, index=reg_data.index)
    X_df["y"] = reg_data["y"]
    X_df["x"] = reg_data["x"]
    X_df = X_df.dropna()

    if len(X_df) < 20:
        return np.nan, np.nan

    y_target = X_df["y"]
    X_features = sm.add_constant(X_df.drop(columns=["y"]))

    try:
        model = sm.OLS(y_target, X_features).fit()
        dw_stat = sm.stats.stattools.durbin_watson(model.resid)
        return float(model.rsquared), float(dw_stat)
    except Exception:
        return np.nan, np.nan


# =============================================================================
# MASTER COMPARISON RUNNER
# =============================================================================

def run_algorithm_comparison() -> pd.DataFrame:
    config = get_config()
    markets = list(config["markets"].keys())
    cache_base_dir = Path("data/interim/pmfg_cache")
    processed_dir = Path(config["data"]["processed_dir"])
    
    master_trends_path = processed_dir / "master_methodology_trends.csv"

    # Pre-load integrated volatility from master_methodology_trends.csv
    master_vol_df = pd.DataFrame()
    if master_trends_path.exists():
        master_vol_df = pd.read_csv(master_trends_path)
        master_vol_df["date"] = pd.to_datetime(master_vol_df["date"])
        # Standardize market string capitalization to prevent key mismatch
        master_vol_df["market_lower"] = master_vol_df["market"].str.lower().str.strip()

    # Algorithm registry mapping choices to functions
    ALGORITHM_SUITE = [
        ("Louvain", "Modularity", lambda g: track_louvain(g)),
        ("Leiden", "Modularity (Refined)", lambda g: track_leiden(g)),
        ("Infomap", "Information Flow", lambda g: track_alternative(g, "infomap")),
        ("Label Propagation", "Local Dynamics", lambda g: track_alternative(g, "lpa")),
        ("Spectral Clustering", "Spectral / Laplacian", lambda g: track_alternative(g, "spectral")),
        ("Hierarchical (HAC)", "Distance Metric Space", lambda g: track_alternative(g, "hac")),
    ]

    comparison_records = []

    print("\n" + "=" * 110)
    print("      DISSERTATION MASTER BENCHMARK: COMMUNITY DETECTION ALGORITHM COMPARISON")
    print("=========================================================================================")

    for market in markets:
        market_lower = market.lower().strip()
        market_cache = cache_base_dir / market
        if not market_cache.exists():
            print(f"⚠️ [WARNING] PMFG Cache missing for market: {market.upper()}. Skipping...")
            continue

        # Load graph sequence from disk checkpoints
        pickle_files = sorted(list(market_cache.glob("pmfg_*.pkl")))
        if not pickle_files:
            print(f"⚠️ [WARNING] No .pkl graph files found in {market_cache}. Skipping...")
            continue

        print(f"\n📂 Loading {len(pickle_files)} PMFG graphs for market: {market.upper()}...")
        graph_sequence = []
        for p_file in pickle_files:
            date_str = p_file.stem.replace("pmfg_", "")
            with open(p_file, "rb") as f:
                graph = pickle.load(f)
                graph.graph["market"] = market
                graph.graph["date"] = date_str
                graph_sequence.append((date_str, graph))

        # Isolate market-specific volatility series from master_methodology_trends.csv
        vix_df = pd.DataFrame()
        if not master_vol_df.empty and "market_volatility" in master_vol_df.columns:
            m_slice = master_vol_df[master_vol_df["market_lower"] == market_lower]
            vix_df = m_slice[["date", "market_volatility"]].rename(
                columns={"market_volatility": "vix"}
            ).dropna().sort_values("date").drop_duplicates()

        # Baseline Louvain ARI map for cross-correlation
        louvain_ari_map = {}

        print("-" * 110)
        print(f"{'Algorithm':<22} | {'Family':<20} | {'Mean ARI (μ)':<12} | {'Std (σ)':<8} | {'Corr w/ Louvain':<16} | {'ARDL R²':<8} | {'DW Stat':<8}")
        print("-" * 110)

        for algo_name, family_name, tracker_func in ALGORITHM_SUITE:
            t_start = time.perf_counter()
            
            # Execute community detection tracker across graph windows
            ledger_df = tracker_func(graph_sequence)
            t_elapsed = time.perf_counter() - t_start

            if ledger_df.empty or "ari_stability" not in ledger_df.columns:
                print(f"❌ [ERROR] {algo_name} produced empty results for {market}.")
                continue

            macro_df = ledger_df[["date", "ari_stability"]].drop_duplicates().copy()
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            macro_df = macro_df.sort_values("date").dropna()

            # Truncate S&P 500 flatline initialization artifacts (2010-2013)
            if market_lower == "sp500":
                macro_df = macro_df[macro_df["date"] >= pd.to_datetime("2014-01-01")]

            ari_vec = macro_df["ari_stability"].values
            mean_ari = float(np.mean(ari_vec))
            std_ari = float(np.std(ari_vec))

            # Store Louvain baseline ARI trajectory for trajectory cross-correlations
            if algo_name == "Louvain":
                louvain_ari_map = dict(zip(macro_df["date"], macro_df["ari_stability"]))
                corr_with_louvain = 1.0000
            else:
                common_dates = [d for d in macro_df["date"] if d in louvain_ari_map]
                if len(common_dates) > 10:
                    vec_louvain = [louvain_ari_map[d] for d in common_dates]
                    macro_indexed = macro_df.set_index("date")
                    vec_current = [macro_indexed.loc[d, "ari_stability"] for d in common_dates]
                    corr_with_louvain = float(np.corrcoef(vec_louvain, vec_current)[0, 1])
                else:
                    corr_with_louvain = np.nan

            # Calculate ARDL R² & DW Statistic against integrated market volatility
            r2_val, dw_val = np.nan, np.nan
            if not vix_df.empty:
                merged = pd.merge_asof(
                    macro_df.sort_values("date"),
                    vix_df.sort_values("date"),
                    on="date",
                    direction="backward"
                ).dropna()
                
                if len(merged) >= 20:
                    max_lags_val = 6 if market_lower == "sp500" else (4 if market_lower == "nifty50" else 2)
                    r2_val, dw_val = evaluate_ardl_fit(merged["ari_stability"], merged["vix"], max_lags=max_lags_val)

            r2_str = f"{r2_val:.4f}" if not np.isnan(r2_val) else "N/A"
            dw_str = f"{dw_val:.4f}" if not np.isnan(dw_val) else "N/A"

            print(f"{algo_name:<22} | {family_name:<20} | {mean_ari:>12.4f} | {std_ari:>8.4f} | {corr_with_louvain:>16.4f} | {r2_str:>8} | {dw_str:>8}")

            comparison_records.append({
                "market": market.upper(),
                "algorithm": algo_name,
                "family": family_name,
                "mean_ari": round(mean_ari, 4),
                "std_ari": round(std_ari, 4),
                "corr_louvain_baseline": round(corr_with_louvain, 4),
                "ardl_r2": round(r2_val, 4) if not np.isnan(r2_val) else "N/A",
                "durbin_watson": round(dw_val, 4) if not np.isnan(dw_val) else "N/A",
                "runtime_seconds": round(t_elapsed, 2)
            })

    # Compile Final DataFrame
    summary_df = pd.DataFrame(comparison_records)
    
    # Save CSV output
    csv_output_path = processed_dir / "community_algorithm_comparison_matrix.csv"
    summary_df.to_csv(csv_output_path, index=False)
    print("\n" + "=" * 110)
    print(f"✓ [SUCCESS] Compiled algorithm comparison matrix saved to: {csv_output_path}")

    # Output formatted LaTeX code for LaTeX chapter integration
    generate_latex_table(summary_df)

    return summary_df


def generate_latex_table(df: pd.DataFrame) -> None:
    """Generates LaTeX code for Chapter 4 formatted to University of Edinburgh LaTeX layout."""
    print("\n" + "=" * 110)
    print("      PUBLICATION-READY LATEX TABLE CODE (CHAPTER 4 RESULTS)")
    print("=========================================================================================\n")

    latex_code = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Cross-Family Community Detection Algorithm Benchmark and Robustness Assessment (2010--2024)}",
        "\\label{tab:community_algorithm_comparison}",
        "\\small",
        "\\begin{tabularx}{\\textwidth}{ll X r r r r}",
        "\\hline\\hline",
        "\\textbf{Market} & \\begin{tabular}[l]{@{}l@{}}\\textbf{Algorithm}\\\\ \\textbf{Family}\\end{tabular} & \\textbf{Algorithm} & \\begin{tabular}[r]{@{}r@{}}\\textbf{Mean ARI}\\\\ ($\\mu$)\\end{tabular} & \\begin{tabular}[r]{@{}r@{}}\\textbf{Std Dev}\\\\ ($\\sigma$)\\end{tabular} & \\begin{tabular}[r]{@{}r@{}}\\textbf{Corr w/}\\\\ \\textbf{Louvain}\\end{tabular} & \\begin{tabular}[r]{@{}r@{}}\\textbf{ARDL}\\\\ $R^2$\\end{tabular} \\\\",
        "\\hline"
    ]

    current_market = ""
    for _, row in df.iterrows():
        m_str = row['market'] if row['market'] != current_market else ""
        current_market = row['market']
        
        corr_str = f"{row['corr_louvain_baseline']:.4f}" if isinstance(row['corr_louvain_baseline'], float) else "1.0000"
        r2_str = str(row['ardl_r2'])
        
        latex_code.append(
            f"{m_str:<10} & {row['family']:<20} & {row['algorithm']:<22} & {row['mean_ari']:>8.4f} & {row['std_ari']:>8.4f} & {corr_str:>12} & {r2_str:>8} \\\\"
        )
        if row['algorithm'] == "Hierarchical (HAC)":
            latex_code.append("\\hline")

    latex_code.extend([
        "\\hline\\hline",
        "\\end{tabularx}",
        "\\begin{minipage}{\\textwidth}",
        "\\vspace{1ex}",
        "\\footnotesize \\textbf{Notes:} Mean ARI ($\\mu$) measures baseline topological stability over a 126-day rolling window ($step=21$). Corr w/ Louvain indicates Pearson correlation between the algorithm's stability time-series and the baseline Louvain consensus trajectory. ARDL $R^2$ reflects dynamic model fit incorporating market volatility proxies (VIX for S\\&P 500, India VIX for NIFTY 50, and 21-day Realized Volatility for Bovespa).",
        "\\end{minipage}",
        "\\end{table}"
    ])

    print("\n".join(latex_code))
    print("\n" + "=" * 110 + "\n")


if __name__ == "__main__":
    run_algorithm_comparison()