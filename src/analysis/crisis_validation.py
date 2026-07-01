"""
src/analysis/crisis_validation.py
=================================
Automated econometric mapping validating network stability against YAML-defined crisis phases.
Explicitly targets config/settings.yaml and config/crisis_periods.yaml.
"""

import yaml
from pathlib import Path
import pandas as pd

def load_combined_config():
    """Locates and combines settings.yaml and crisis_periods.yaml dynamically."""
    script_dir = Path(__file__).resolve().parent
    
    # Traverse up to find the true root containing the 'config' folder
    root_dir = None
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / "config").is_dir():
            root_dir = parent
            break
            
    if root_dir is None:
        # Hardcoded manual absolute path fallback to bypass deep nested shells
        root_dir = Path("/home/s2843292/Dissertation/Dissertation")
        if not (root_dir / "config").is_dir():
            root_dir = Path("/home/s2843292/Dissertation")

    settings_path = root_dir / "config" / "settings.yaml"
    crisis_path = root_dir / "config" / "crisis_periods.yaml"

    if not settings_path.exists() or not crisis_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Could not find YAML files.\nChecked:\n- {settings_path}\n- {crisis_path}"
        )

    # Load and merge both configurations
    with open(settings_path, "r") as f:
        config = yaml.safe_load(f) or {}
    with open(crisis_path, "r") as f:
        crisis_data = yaml.safe_load(f) or {}
        
    config.update(crisis_data)
    return config, root_dir

def assign_crisis_phase(window_date, config):
    """Labels a specific network timeline date relative to crisis events."""
    dt = pd.to_datetime(window_date)
    
    # Check Crises
    for crisis in config.get('crises', []):
        c_start = pd.to_datetime(crisis['crisis_start'])
        c_end = pd.to_datetime(crisis['crisis_end'])
        # 63 trading days is roughly 3 calendar months
        pre_start = c_start - pd.DateOffset(months=3)
        
        if c_start <= dt <= c_end:
            return f"In-Crisis ({crisis['short_name']})"
        elif pre_start <= dt < c_start:
            return f"Pre-Crisis ({crisis['short_name']})"
            
    # Check Tranquil
    for tranq in config.get('tranquil_periods', []):
        t_start = pd.to_datetime(tranq['start'])
        t_end = pd.to_datetime(tranq['end'])
        if t_start <= dt <= t_end:
            return "Tranquil Recovery"
            
    return "Baseline"

def main():
    config, root_dir = load_combined_config()
    
    # Locate master_methodology_trends.csv inside the resolved project root
    data_path = root_dir / "data" / "processed" / "master_methodology_trends.csv"
    
    if not data_path.exists():
        # Fallback to current working directory checking
        data_path = Path("data/processed/master_methodology_trends.csv")

    if not data_path.exists():
        print(f"[ERROR] Target file not found: {data_path.absolute()}")
        return
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # ---------------------------------------------------------------------------
    # FIX INCONSISTENCY: Truncate S&P 500 initialization leakage (2010-2013)
    # ---------------------------------------------------------------------------
    corrupt_sp500_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
    df = df[~corrupt_sp500_mask].reset_index(drop=True)
    
    # Map the calendar constraints over the sanitized matrix slices
    df['regime'] = df['date'].apply(lambda x: assign_crisis_phase(x, config))
    
    print("\n" + "="*75)
    print("   SANED EMPIRICAL VERIFICATION: MEAN NETWORK STABILITY ($ARI$) BY MACRO REGIME")
    print("="*75)
    
    # Generate unique point-in-time entries to avoid coordinate weight bias
    pivot_df = df[['date', 'market', 'regime', 'ari_stability']].drop_duplicates()
    
    summary = pivot_df.groupby(['regime', 'market'])['ari_stability'].mean().unstack(level=1)
    print(summary.round(4).to_string())
    print("="*75 + "\n")

if __name__ == "__main__":
    main()