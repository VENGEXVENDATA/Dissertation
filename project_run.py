"""
project_run.py
==============
Master Orchestration Script for MSc Data Science Dissertation.
Executes the end-to-end quantitative pipeline sequentially from raw data 
acquisition and outlier auditing through to advanced econometric modeling 
and cross-family algorithmic benchmarking.

Usage:
------
    python project_run.py                   # Runs complete pipeline (Stages 1-6)
    python project_run.py --skip-acquisition # Skips Stage 1 if raw prices exist
    python project_run.py --skip-benchmark   # Skips Stage 6 cross-family comparison
"""

from __future__ import annotations

import argparse
import sys
import subprocess
import time
from pathlib import Path

# Enforce UTF-8 output encoding for standard terminal compatibility
sys.stdout.reconfigure(encoding="utf-8")

# Complete sequential execution pipeline stages
PIPELINE_STAGES = [
    {
        "stage": "STAGE 1: Data Acquisition",
        "script": "pipelines/run_acquisition.py",
        "flag_skip": "skip_acquisition",
        "description": "Fetching raw price data from Yahoo Finance / Metadata"
    },
    {
        "stage": "STAGE 1B: Base Price Dataset Integrity & Outlier Audit",
        "script": "src/analysis/check_raw_data_outliers.py",
        "flag_skip": None,
        "description": "Screening missingness, zero-return flatlines, and return outliers"
    },
    {
        "stage": "STAGE 2: PMFG Network Construction & Consensus Partitioning",
        "script": "pipelines/run_pipeline.py",
        "flag_skip": None,
        "description": "Constructing PMFGs with checkpointing & calculating association-matrix consensus ARI"
    },
    {
        "stage": "STAGE 3: Macro Implied & Realized Volatility Integration",
        "script": "src/analysis/volatility_integration.py",
        "flag_skip": None,
        "description": "Merging VIX, India VIX, and 21-day Realized Volatility into master trends"
    },
    {
        "stage": "STAGE 4A: Stationarity Pre-Tests & Timeline Plotting",
        "script": "pipelines/run_analysis.py",
        "flag_skip": None,
        "description": "ADF unit root testing & publication-quality timeline plot generation"
    },
    {
        "stage": "STAGE 4B: Autocorrelation Audit & Mann-Whitney U Testing",
        "script": "src/analysis/check_autocorrelation_severity.py",
        "flag_skip": None,
        "description": "Evaluating window overlap persistence bias and directional rank-sum tests"
    },
    {
        "stage": "STAGE 4C: Regime Significance Testing",
        "script": "src/analysis/statistical_significance.py",
        "flag_skip": None,
        "description": "Evaluating calendar crisis dissolution across baseline regimes"
    },
    {
        "stage": "STAGE 5D: Inter-Crisis Long-Memory (Hurst) Analysis",
        "script": "src/analysis/inter_crisis_analysis.py",
        "flag_skip": None,
        "description": "Tracking structural variance and Hurst exponents (H) in calm states"
    },
    {
        "stage": "STAGE 5E: Topology-Informed Portfolio Backtest",
        "script": "src/analysis/portfolio_optimization_1.py",
        "flag_skip": None,
        "description": "Backtesting central vs. peripheral asset allocations"
    },
    {
        "stage": "STAGE 6: Cross-Family Community Algorithm Benchmark",
        "script": "src/analysis/compare_community_algorithms.py",
        "flag_skip": "skip_benchmark",
        "description": "Evaluating 6 algorithms across 5 theoretical families and exporting LaTeX table"
    }
]


def print_banner(text: str, char: str = "=") -> None:
    width = 95
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def run_script(script_path: str) -> tuple[bool, float]:
    """
    Executes a sub-script as an isolated python process.
    Returns (success_boolean, execution_time_seconds).
    """
    start_time = time.perf_counter()
    python_cmd = sys.executable

    if not Path(script_path).exists():
        print(f"❌ [CRITICAL ERROR] Target script does not exist: {script_path}")
        return False, 0.0

    try:
        subprocess.run([python_cmd, script_path], check=True)
        elapsed = time.perf_counter() - start_time
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ [STAGE FAILED] Subprocess exit code {e.returncode} for script: {script_path}")
        return False, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Execution Pipeline Orchestrator")
    parser.add_argument(
        "--skip-acquisition",
        action="store_true",
        help="Skip Stage 1 data acquisition if raw price data is already downloaded."
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip Stage 6 cross-family community algorithm comparison benchmark."
    )
    args = parser.parse_args()

    print_banner("UNIVERSITY OF EDINBURGH - MSC DATA SCIENCE DISSERTATION", "=")
    print(" Master Pipeline Orchestrator: Temporal Stock Networks & Market Stress")
    print(" Scope: S&P 500, NIFTY 50, Bovespa (2010-2024)")
    print_banner("INITIALIZING SEQUENTIAL STAGE EXECUTION", "=")

    summary_log = []
    total_start_time = time.perf_counter()

    for idx, step in enumerate(PIPELINE_STAGES, 1):
        stage_name = step["stage"]
        script = step["script"]
        skip_flag = step["flag_skip"]
        description = step["description"]

        if skip_flag and getattr(args, skip_flag, False):
            print(f"\n[SKIP] {stage_name} (--{skip_flag.replace('_', '-')} flag detected)")
            summary_log.append((stage_name, "SKIPPED", 0.0))
            continue

        print_banner(f"[{idx}/{len(PIPELINE_STAGES)}] {stage_name}", "-")
        print(f"▶ Script: {script}")
        print(f"▶ Scope : {description}\n")

        success, elapsed = run_script(script)

        if success:
            print(f"\n✓ [{stage_name}] Finished successfully in {elapsed:.2f}s")
            summary_log.append((stage_name, "SUCCESS", elapsed))
        else:
            print(f"\n❌ [{stage_name}] Terminated with errors in {elapsed:.2f}s")
            summary_log.append((stage_name, "FAILED", elapsed))
            print_banner("PIPELINE HALTED DUE TO STAGE FAILURE", "!")
            print("Fix the error in the module above before re-running.\n")
            sys.exit(1)

    total_elapsed = time.perf_counter() - total_start_time

    print_banner("MASTER PIPELINE EXECUTION SUMMARY", "=")
    print(f"{'Stage Name':<65} | {'Status':<10} | {'Runtime':<10}")
    print("-" * 92)
    for stage_name, status, elapsed in summary_log:
        status_icon = "✓ OK" if status == "SUCCESS" else ("- SKIP" if status == "SKIPPED" else "❌ FAIL")
        print(f"{stage_name:<65} | {status_icon:<10} | {elapsed:>7.2f}s")
    print("-" * 92)
    print(f"Total Pipeline Runtime: {total_elapsed / 60:.2f} minutes")
    print_banner("ALL STAGES COMPLETED PRISTINELY & REPRODUCIBLY", "=")


if __name__ == "__main__":
    main()