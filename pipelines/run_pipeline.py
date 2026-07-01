"""
pipelines/run_pipeline.py
==========================
Parallelized master orchestration script executing the approved research design.
Distributes temporal windows across available cluster cores using multiprocessing pools.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from functools import partial

# Maintain repo mapping stability across nested environments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Aligned imports reflecting our refactored architecture
from src.correlation.correlation_matrix import clean_and_stationarize, slice_rolling_windows
from src.network.network_engine import compute_mantegna_distances
from src.network.pmfg_engine import build_pmfg_network_fast
from src.community.consensus_louvain import track_timeline_stability
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger
from src.utils.timer import stage_timer

logger = get_logger("pipelines.run_pipeline")


def process_single_window(window_tuple, market_name):
    """
    Worker function executed in parallel across CPU cores.
    Processes a single temporal window slice.
    """
    date_str, window_chunk = window_tuple
    try:
        # 1. Pearson Correlation
        corr_matrix = window_chunk.corr(method="pearson")
        
        # 2. Mantegna Distance Calculation
        dist_matrix = compute_mantegna_distances(corr_matrix)
        
        # 3. Fast Planarity Constrained Network Construction
        pmfg_net = build_pmfg_network_fast(dist_matrix)
        pmfg_net.graph["market"] = market_name
        
        # Unbuffered stdout tracking optimized for Slurm captures
        print(f"   [CORE PROGRESS] Finished PMFG calculation for {market_name.upper()} window: {date_str}", flush=True)
        return (date_str, pmfg_net)
        
    except Exception as e:
        print(f"   [CORE ERROR] Failed processing window {date_str} for {market_name}: {e}", flush=True)
        return None


@stage_timer("Approved Parallel Dissertation Research Pipeline")
def main() -> None:
    # Pull centralized project configuration
    config = get_config()
    
    markets = list(config["markets"].keys())
    output_dir = Path(config["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract core count allocated by Slurm dynamically
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    logger.info("Initializing multi-core process engine mapping to %d available CPUs.", num_cores)
    
    macro_summaries = []
    
    for market in markets:
        logger.info("=" * 70)
        logger.info("PROCESSING ACCORDING TO DISSERTATION METHODOLOGY: %s", market.upper())
        logger.info("=" * 70)
        
        try:
            # 1. Load asset universe using centralized data constraints
            returns_df = clean_and_stationarize(market_name=market)
            
            # 2. Slice time-series using parameters defined in settings.yaml
            sliced_windows = slice_rolling_windows(returns_df)
            logger.info("%s partitioned into %d temporal windows.", market.upper(), len(sliced_windows))
            
            # 3. Construct PMFG topological networks IN PARALLEL
            logger.info("Spawning parallel process workers for PMFG planarity constraints...")
            
            # Use partial to bind the specific market string to the workers
            worker_func = partial(process_single_window, market_name=market)
            
            with mp.Pool(processes=num_cores) as pool:
                parallel_results = pool.map(worker_func, sliced_windows)
            
            # Filter out failed loops
            graph_sequence = [res for res in parallel_results if res is not None]
            
            if not graph_sequence:
                logger.error("No valid graph sequences extracted for market: %s. Skipping step.", market)
                continue
                
            # 4. Compute consensus partitions and evaluate temporal stability (ARI)
            logger.info("Computing 100-run consensus partitions and ARI tracking for %s...", market.upper())
            stability_ledger = track_timeline_stability(graph_sequence)
            
            # Defensive validation check: Ensure the returned dataframe contains processed metrics
            if stability_ledger.empty or "ari_stability" not in stability_ledger.columns:
                logger.error("Stability tracking ledger generated empty data output for market: %s", market)
                continue
                
            # Export localized market tracking metrics
            stability_ledger.to_csv(output_dir / f"{market}_methodology_stability.csv", index=False)
            
            # Isolate macro view for historical plotting
            macro_trend = stability_ledger[["date", "market", "ari_stability"]].drop_duplicates()
            macro_summaries.append(macro_trend)
            logger.info("%s processing completed successfully.", market.upper())
            
        except Exception as e:
            logger.error("Pipeline crashed on market processing for %s: %s", market, e, exc_info=True)
            
    if macro_summaries:
        master_file = pd.concat(macro_summaries, axis=0)
        master_file.to_csv(output_dir / "master_methodology_trends.csv", index=False)
        logger.info("[SUCCESS] Pristine tracking records saved to: data/processed/master_methodology_trends.csv")
    else:
        logger.error("[CRITICAL FAILURE] Pipeline execution finished but zero master files were written.")


if __name__ == "__main__":
    # Crucial for safe execution of Python pools across OS layers
    mp.freeze_support()
    main()