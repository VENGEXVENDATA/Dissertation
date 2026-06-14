"""
pipelines/run_pipeline.py
==========================
Master orchestration script executing the approved research design.
Loads full indexes, slices with 21-day steps, builds PMFGs, runs 100-fold consensus,
and saves metrics for downstream econometric modeling.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correlation.correlation_matrix import clean_and_stationarize, slice_rolling_windows, compute_mantegna_distances
from src.network.pmfg_engine import build_pmfg_network
from src.community.consensus_louvain import track_timeline_stability
from src.utils.logging_config import get_logger
from src.utils.timer import stage_timer

logger = get_logger("pipelines.run_pipeline")


@stage_timer("Approved Dissertation Research Pipeline")
def main() -> None:
    markets = ["sp500", "nifty50", "bovespa"]
    window_size = 126  # Approved baseline config
    step_size = 21     # Approved 1-month window step
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    macro_summaries = []
    
    for market in markets:
        logger.info("=" * 70)
        logger.info("PROCESSING ACCORDING TO DISSERTATION METHODOLOGY: %s", market.upper())
        logger.info("=" * 70)
        
        try:
            # 1. Load full asset universe and convert to logs
            returns_df = clean_and_stationarize(market_name=market, max_missing_pct=0.05)
            
            # 2. Slice time-series using the 21-day step size constraint
            sliced_windows = slice_rolling_windows(returns_df, window_size=window_size, step_size=step_size)
            logger.info("%s partitioned into %d temporal windows.", market.upper(), len(sliced_windows))
            
            # 3. Construct PMFG topological networks sequentially
            logger.info("Constructing sequential PMFG networks... (This performs planarity tests)")
            graph_sequence = []
            for date_str, window_chunk in sliced_windows:
                corr_matrix = window_chunk.corr(method="pearson")
                dist_matrix = compute_mantegna_distances(corr_matrix)
                
                # Build Planar Maximally Filtered Graph
                pmfg_net = build_pmfg_network(dist_matrix)
                pmfg_net.graph["market"] = market
                
                graph_sequence.append((date_str, pmfg_net))
                
            # 4. Run 100 independent Louvain iterations per slice for consensus stability
            logger.info("Computing 100-run consensus partitions and ARI tracking...")
            stability_ledger = track_timeline_stability(graph_sequence)
            
            # Export market tracking metrics
            stability_ledger.to_csv(output_dir / f"{market}_methodology_stability.csv", index=False)
            
            # Isolate macro view for historical plotting
            macro_trend = stability_ledger[["date", "market", "ari_stability"]].drop_duplicates()
            macro_summaries.append(macro_trend)
            logger.info("%s processing completed successfully.", market.upper())
            
        except Exception as e:
            logger.error("Pipeline crashed on market processing for %s: %s", market, e)
            
    if macro_summaries:
        master_file = pd.concat(macro_summaries, axis=0)
        master_file.to_csv(output_dir / "master_methodology_trends.csv", index=False)
        logger.info("[SUCCESS] Pristine tracking records saved to: data/processed/master_methodology_trends.csv")


if __name__ == "__main__":
    main()
