"""
pipelines/run_pipeline.py
==========================
Parallelized master orchestration script executing the approved research design.
Distributes temporal windows across available cluster cores using multiprocessing pools,
with per-window disk checkpointing and dynamic community algorithm dispatching across 
all 5 theoretical families (Louvain, Leiden, Infomap, LPA, Spectral, HAC).
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from functools import partial

# Maintain repo mapping stability across nested environments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correlation.correlation_matrix import clean_and_stationarize, slice_rolling_windows
from src.network.network_engine import compute_mantegna_distances
from src.network.pmfg_engine import build_pmfg_network_fast
from src.utils.config_loader import get_config
from src.utils.logging_config import get_logger
from src.utils.timer import stage_timer

logger = get_logger("pipelines.run_pipeline")


def process_single_window(window_tuple: tuple[str, pd.DataFrame], market_name: str, cache_dir: Path):
    """
    Worker function executed in parallel across CPU cores.
    Checks for cached graph files on disk before executing PMFG construction.
    """
    date_str, window_chunk = window_tuple
    
    # Establish per-window checkpoint file path
    market_cache_dir = cache_dir / market_name
    market_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = market_cache_dir / f"pmfg_{date_str}.pkl"

    # Checkpoint Guard: Load graph if already computed in a previous SLURM run
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                pmfg_net = pickle.load(f)
            
            # Re-assert graph metadata defensively
            pmfg_net.graph["market"] = market_name
            pmfg_net.graph["date"] = date_str
            
            print(f"   [CHECKPOINT LOAD] Loaded cached PMFG for {market_name.upper()} window: {date_str}", flush=True)
            return (date_str, pmfg_net)
        except Exception as e:
            print(f"   [CHECKPOINT ERROR] Corrupted cache for {date_str}, recomputing... ({e})", flush=True)

    try:
        # 1. Pearson Correlation Matrix
        corr_matrix = window_chunk.corr(method="pearson")
        
        # 2. Mantegna Distance Calculation
        dist_matrix = compute_mantegna_distances(corr_matrix)
        
        # 3. Fast Planarity-Constrained PMFG Construction
        pmfg_net = build_pmfg_network_fast(dist_matrix)
        pmfg_net.graph["market"] = market_name
        pmfg_net.graph["date"] = date_str
        
        # Save graph object to disk checkpoint
        with open(cache_file, "wb") as f:
            pickle.dump(pmfg_net, f)
            
        print(f"   [CORE SUCCESS] Computed & cached PMFG for {market_name.upper()} window: {date_str}", flush=True)
        return (date_str, pmfg_net)
        
    except Exception as e:
        print(f"   [CORE ERROR] Failed processing window {date_str} for {market_name}: {e}", flush=True)
        return None


@stage_timer("Approved Parallel Dissertation Research Pipeline")
def main() -> None:
    config = get_config()
    
    markets = list(config["markets"].keys())
    output_dir = Path(config["data"]["processed_dir"])
    cache_dir = Path("data/interim/pmfg_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract CPU core allocation assigned by SLURM
    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    logger.info("Initializing process engine with %d available CPUs.", num_cores)
    
    # Extract community detection settings dynamically from YAML
    comm_config = config.get("community", {})
    algorithm = comm_config.get("algorithm_primary", "louvain").lower().strip()
    consensus_runs = comm_config.get("consensus_runs", 100)
    
    macro_summaries = []
    
    for market in markets:
        logger.info("=" * 70)
        logger.info("PROCESSING ACCORDING TO DISSERTATION METHODOLOGY: %s", market.upper())
        logger.info("=" * 70)
        
        try:
            # 1. Load asset universe using centralized quality constraints
            returns_df = clean_and_stationarize(market_name=market)
            
            # 2. Slice time-series using parameters defined in settings.yaml
            sliced_windows = slice_rolling_windows(returns_df)
            logger.info("%s partitioned into %d temporal windows.", market.upper(), len(sliced_windows))
            
            # 3. Construct PMFG topological networks IN PARALLEL with disk caching
            logger.info("Spawning process workers for window evaluation...")
            worker_func = partial(process_single_window, market_name=market, cache_dir=cache_dir)
            
            with mp.Pool(processes=num_cores) as pool:
                parallel_results = pool.map(worker_func, sliced_windows)
            
            graph_sequence = [res for res in parallel_results if res is not None]
            
            if not graph_sequence:
                logger.error("No valid graph sequences extracted for market: %s.", market)
                continue
                
            # 4. Dynamic Community Partitioning & ARI Temporal Tracking
            logger.info(
                "Computing consensus partitions and ARI tracking using [%s] for %s...",
                algorithm.upper(), market.upper()
            )
            
            if algorithm == "leiden":
                from src.community.consensus_leiden import track_leiden_timeline_stability
                stability_ledger = track_leiden_timeline_stability(graph_sequence)
            elif algorithm == "louvain":
                from src.community.consensus_louvain import track_timeline_stability
                stability_ledger = track_timeline_stability(graph_sequence)
            elif algorithm in ["infomap", "lpa", "spectral", "hac"]:
                from src.community.alternative_algorithms import track_alternative_timeline_stability
                stability_ledger = track_alternative_timeline_stability(graph_sequence, algorithm_name=algorithm)
            else:
                raise ValueError(f"Unrecognized algorithm setting: '{algorithm}'")
            
            if stability_ledger.empty:
                logger.error("Stability tracking ledger generated empty output for market: %s", market)
                continue
                
            # DEFENSIVE GUARD: Validate column existence before extracting macro trends
            required_cols = ["date", "market", "ari_stability"]
            missing_cols = [c for c in required_cols if c not in stability_ledger.columns]
            
            if missing_cols:
                logger.error(
                    "Missing expected columns %s in stability_ledger for market %s. Available: %s",
                    missing_cols, market.upper(), list(stability_ledger.columns)
                )
                continue
                
            # Save localized market stability ledger
            stability_ledger.to_csv(output_dir / f"{market}_methodology_stability.csv", index=False)
            
            # Extract clean macro trend slice
            macro_trend = stability_ledger[required_cols].drop_duplicates()
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
    mp.freeze_support()
    main()