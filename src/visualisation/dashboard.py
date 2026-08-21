"""
src/visualization/dashboard.py
==============================
Interactive Streamlit dashboard visualizing rolling network ARI stability, 
local implied volatility index dynamics, lead-lag CCFs, and ARDL outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as str_lit
import plotly.graph_objects as go
import yaml

# Maintain repository architecture alignment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Set page configuration for wide layout and clean typography
str_lit.set_page_config(
    page_title="Global Equity Network Stability Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# DATA LOADING & CONFIGURATION CACHING
# ---------------------------------------------------------------------------
@str_lit.cache_data
def load_master_dataset(trends_path: Path) -> pd.DataFrame:
    if not trends_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(trends_path)
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()
    
    # Apply baseline truncation patch for S&P 500 historical artifacts
    corrupt_mask = (df['market'] == 'sp500') & (df['date'] < pd.to_datetime('2014-01-01'))
    df = df[~corrupt_mask].reset_index(drop=True)
    return df

@str_lit.cache_data
def load_crisis_config(root_dir: Path) -> dict:
    crisis_path = root_dir / "config" / "crisis_periods.yaml"
    if not crisis_path.exists():
        return {}
    with open(crisis_path, "r") as f:
        return yaml.safe_load(f) or {}

# Locate repository root paths
script_dir = Path(__file__).resolve().parent
root_dir = None
for parent in [script_dir] + list(script_dir.parents):
    if (parent / "config").is_dir():
        root_dir = parent
        break
if root_dir is None:
    root_dir = Path("/home/s2843292/Dissertation/Dissertation")

data_path = root_dir / "data" / "processed" / "master_methodology_trends.csv"

# Ingest data
df_master = load_master_dataset(data_path)
crisis_config = load_crisis_config(root_dir)

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
str_lit.sidebar.title("🔬 Thesis Controls")
str_lit.sidebar.markdown("---")

# Market selector
market_mapping = {
    "S&P 500 (United States)": "sp500",
    "NIFTY 50 (India)": "nifty50",
    "BOVESPA (Brazil)": "bovespa"
}
selected_market_label = str_lit.sidebar.selectbox(
    "Select Target Stock Market:",
    list(market_mapping.keys())
)
target_market = market_mapping[selected_market_label]

# Dynamic column fallback checks
vol_col = 'market_volatility_x' if 'market_volatility_x' in df_master.columns else 'market_volatility'

# Sub-slice dataset to selected market
df_filtered = df_master[df_master['market'] == target_market].sort_values('date').reset_index(drop=True)

str_lit.sidebar.markdown(f"**Data Span:**\n`{df_filtered['date'].min().strftime('%Y-%m-%d')}` to `{df_filtered['date'].max().strftime('%Y-%m-%d')}`")
str_lit.sidebar.markdown(f"**Total Records ($N_{{raw}}$):** `{len(df_filtered)}`")

# ---------------------------------------------------------------------------
# MAIN PAGE INTERFACE
# ---------------------------------------------------------------------------
str_lit.title("🌐 Global Financial Networks: Cointegration & Lead-Lag Dashboard")
str_lit.subheader("MSc Data Science Dissertation — Academic Verification Suite")
str_lit.markdown("""
This analytics suite provides empirical verification for **Network Topology Invariance** 
under systemic macroeconomic shocks across developed and emerging equity markets.
""")

tab1, tab2, tab3 = str_lit.tabs([
    "📊 Temporal Network Stability & Crisis Regimes", 
   "", 
    ""
])

# ---------------------------------------------------------------------------
# TAB 1: TEMPORAL NETWORK STABILITY
# ---------------------------------------------------------------------------
with tab1:
    str_lit.header("Temporal Evolution of Community Structures vs. Implied Volatility")
    str_lit.markdown("""
    This dynamic visualization aligns your rolling **Adjusted Rand Index (ARI)** timeline against your 
    localized market-specific implied volatility index (VIX / India VIX / VXEWZ). Shaded areas represent 
    historical macroeconomic shocks defined in your thesis configuration.
    """)
    
    # Construct interactive dual-axis Plotly Chart
    fig = go.Figure()
    
    # 1. Plot Network ARI Stability Line
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['ari_stability'],
        name="Network stability (ARI)",
        line=dict(color='#2ca02c', width=2),
        yaxis="y1"
    ))
    
    # 2. Plot Implied Volatility Index Line
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered[vol_col],
        name="Implied Volatility Index",
        line=dict(color='#d62728', width=1.5, dash='dot'),
        yaxis="y2"
    ))
    
    # 3. Add shaded crisis regions dynamically from yaml config
    for crisis in crisis_config.get('crises', []):
        c_start = pd.to_datetime(crisis['crisis_start'])
        c_end = pd.to_datetime(crisis['crisis_end'])
        
        # Check if the crisis overlaps with our current filtered dataset timeline range
        if df_filtered['date'].min() <= c_end and df_filtered['date'].max() >= c_start:
            fig.add_vrect(
                x0=c_start, x1=c_end,
                fillcolor="#ff7f0e", opacity=0.12,
                layer="below", line_width=0,
                annotation_text=crisis['short_name'],
                annotation_position="top left",
                annotation_font=dict(size=10, color="#7f7f7f")
            )

    fig.update_layout(
        title=dict(text=f"{selected_market_label.split(' ')[0]} Historical Trajectory Mapping", font=dict(size=16)),
        xaxis=dict(title="Timeline (Trading Windows)"),
        yaxis=dict(
            title=dict(
                text="Adjusted Rand Index ($ARI$)",
                font=dict(color="#2ca02c")
            ),
            tickfont=dict(color="#2ca02c")
        ),
        yaxis2=dict(
            title=dict(
                text="Implied Volatility Index (%)",
                font=dict(color="#d62728")
            ),
            tickfont=dict(color="#d62728"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=550
    )
    
    str_lit.plotly_chart(fig, use_container_width=True)
    
    # Display brief stats summaries
    col1, col2, col3 = str_lit.columns(3)
    with col1:
        str_lit.metric("Average ARI Stability Score", f"{df_filtered['ari_stability'].mean():.4f}")
    with col2:
        str_lit.metric("Average Implied Volatility Level", f"{df_filtered[vol_col].mean():.2f}%")
    with col3:
        correlation_raw = df_filtered['ari_stability'].corr(df_filtered[vol_col])
        str_lit.metric("Linear Pearson Correlation ($r$)", f"{correlation_raw:.4f}")

# ---------------------------------------------------------------------------
# FOOTER & SUBMISSION BRANDING
# ---------------------------------------------------------------------------
str_lit.markdown("---")
str_lit.caption("University of Edinburgh — Master of Science in Data Science Dissertation. August Submission Timeline.")