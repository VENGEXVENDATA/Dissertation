"""
src/analysis/algorithm_benchmark.py
====================================
Cross-family benchmark runner comparing Louvain, Leiden, Infomap, LPA, and Spectral
clustering on the exact same PMFG temporal stock networks.
Outputs comparative stability mean, standard deviation, and cross-correlation matrix.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.community.consensus_louvain import track_timeline_stability
from src.community.consensus_leiden import track_leiden_timeline_stability
from src.community.alternative_algorithms import track_alternative_timeline_stability

def benchmark_all_families(market_name: str = "sp500") -> pd.DataFrame:
    cache_dir = Path("data/interim/pmfg_cache") / market_name
    if not cache_dir.exists():
        print(f"❌ Cache directory missing: {cache_dir}")
        return pd.DataFrame()
        
    # Load all cached PMFG graphs
    pickle_files = sorted(list(cache_dir.glob("pmfg_*.pkl")))
    print(f"Found {len(pickle_files)} cached window graphs for {market_name.upper()}.")
    
    graph_sequence = []
    for p_file in pickle_files:
        date_str = p_file.stem.replace("pmfg_", "")
        with open(p_file, "rb") as f:
            graph = pickle.load(f)
            graph.graph["market"] = market_name
            graph.graph["date"] = date_str
            graph_sequence.append((date_str, graph))
            
    algorithms = {
        "Louvain (Modularity)": lambda g: track_timeline_stability(g),
        "Leiden (Modularity Refined)": lambda g: track_leiden_timeline_stability(g),
        "Infomap (Information Flow)": lambda g: track_alternative_timeline_stability(g, "infomap"),
        "Label Propagation (Local)": lambda g: track_alternative_timeline_stability(g, "lpa"),
        "Spectral (Linear Algebra)": lambda g: track_alternative_timeline_stability(g, "spectral")
    }
    
    results = []
    
    print("\n" + "="*85)
    print(f"   CROSS-FAMILY COMMUNITY DETECTION BENCHMARK: {market_name.upper()}")
    print("="*85)
    
    for name, func in algorithms.items():
        print(f"▶ Running {name}...")
        df = func(graph_sequence)
        macro_df = df[["date", "ari_stability"]].drop_duplicates()
        
        ari_vals = macro_df["ari_stability"].dropna()
        mean_ari = ari_vals.mean()
        std_ari = ari_vals.std()
        
        results.append({
            "Algorithm Family": name,
            "Mean ARI (μ)": f"{mean_ari:.4f}",
            "Std Dev (σ)": f"{std_ari:.4f}",
            "Min ARI": f"{ari_vals.min():.4f}",
            "Max ARI": f"{ari_vals.max():.4f}"
        })
        
    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))
    print("="*85 + "\n")
    return summary_df

if __name__ == "__main__":
    benchmark_all_families("sp500")