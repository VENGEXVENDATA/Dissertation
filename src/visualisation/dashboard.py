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
    "⏱️ Lead-Lag Cross-Correlation Functions (CCF)", 
    "📈 Econometric ARDL Cointegration Engines"
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
# TAB 2: LEAD-LAG CCF ANALYSIS
# ---------------------------------------------------------------------------
with tab2:
    str_lit.header("Lead-Lag Profile: Cross-Correlation Function (CCF) Optimization")
    str_lit.markdown("""
    This CCF chart traces the dynamic tracking relationship between network structures and market panic.
    *   **Negative Lags ($\tau < 0$):** Network topological shifts **lead** implied volatility.
    *   **Positive Lags ($\tau > 0$):** Implied volatility **leads** network structural reorganizations.
    """)
    
    max_lag = 30
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_profile = []
    
    # Calculate level-form correlation explicitly per lag step
    for lag in lags:
        if lag < 0:
            shifted_ari = df_filtered['ari_stability'].shift(-lag)
            volatility = df_filtered[vol_col]
        elif lag > 0:
            shifted_ari = df_filtered['ari_stability']
            volatility = df_filtered[vol_col].shift(lag)
        else:
            shifted_ari = df_filtered['ari_stability']
            volatility = df_filtered[vol_col]
            
        combined = pd.DataFrame({'x': shifted_ari, 'y': volatility}).dropna()
        corr_val = np.corrcoef(combined['x'], combined['y'])[0, 1] if len(combined) > 5 else 0.0
        ccf_profile.append(corr_val)
        
    ccf_profile = np.array(ccf_profile)
    peak_idx = np.argmax(np.abs(ccf_profile))
    peak_lag = lags[peak_idx]
    peak_corr = ccf_profile[peak_idx]
    
    # Create CCF Plotly Line Chart
    fig_ccf = go.Figure()
    fig_ccf.add_trace(go.Scatter(
        x=lags, y=ccf_profile,
        mode='lines+markers',
        name="CCF Coefficient",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=5)
    ))
    # Vertical line indicating origin (concomitant alignment)
    fig_ccf.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)
    # Horizontal reference zero line
    fig_ccf.add_hline(y=0, line_color="gray", opacity=0.3)
    
    # Highlight peak predictive lag spot
    fig_ccf.add_annotation(
        x=peak_lag, y=peak_corr,
        text=f"Peak Lag: {peak_lag}d (r={peak_corr:.3f})",
        showarrow=True, arrowhead=2,
        arrowcolor="#ff7f0e", ax=40, ay=-40,
        font=dict(color="#ff7f0e", size=11, weight="bold") # Fixed: Uses "weight" instead of "fontweight"
    )
    
    fig_ccf.update_layout(
        title="Cross-Correlation Coefficients across Daily Time-Offsets ($\tau$)",
        xaxis=dict(title="Lag Offset $\\tau$ (Days: Negative implies Network leads Volatility)"),
        yaxis=dict(title="Pearson Correlation Coefficient $R(\\tau)$"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    str_lit.plotly_chart(fig_ccf, use_container_width=True)
    
    # Academic interpretation blocks matching your actual mathematical outputs
    if target_market == 'sp500':
        str_lit.info("""
        **Academic Commentary (S&P 500):** Volatility leads the network with a long operational memory of **+12 trading days**. 
        In highly liquid developed markets, network structure adjusts gradually, stabilizing only after a sustained systemic pricing regime change.
        """)
    elif target_market == 'nifty50':
        str_lit.info("""
        **Academic Commentary (NIFTY 50):** Volatility leads the network with a short-run transmission lag of **+4 trading days**. 
        This is a standard emerging-market transmission window, where institutional portfolios slowly rebalance into defensive vs. cyclical holdings following a volatility shock.
        """)
    elif target_market == 'bovespa':
        str_lit.success("""
        **Academic Commentary (BOVESPA):** The network structure leads volatility by **-1 trading day**. 
        In commodity-concentrated emerging markets like Brazil, sector rotation and capital consolidation among commodity heavyweights act as an early indicator preceding generalized market panic.
        """)

# ---------------------------------------------------------------------------
# TAB 3: ECONOMETRIC ARDL MODELS
# ---------------------------------------------------------------------------
with tab3:
    str_lit.header("Autoregressive Distributed Lag (ARDL) Model Coefficients")
    str_lit.markdown("""
    Below are the finalized, autocorrelation-remediated regression model outputs. 
    Lags and structures are custom-adapted to your empirical CCF peak lags.
    """)
    
    # Present the exact statistical results that solved your autocorrelation challenges
    if target_market == 'sp500':
        str_lit.markdown("### Model Structure: `Y = d_ari_stability` | `X = market_volatility_x` | Bounded Max Lags: `ARDL(12, 12)`")
        
        # Table of your significant coefficients
        coefficients_data = {
            "Variable Parameter": [
                "ari_stability_lag_1 (Autoregressive Memory)", 
                "market_volatility_x (Immediate Impact)", 
                "market_volatility_x_lag_3 (Rebound)", 
                "market_volatility_x_lag_6 (Adjustment)", 
                "market_volatility_x_lag_9 (Stabilizer)", 
                "market_volatility_x_lag_12 (Stabilizer)"
            ],
            "Coefficient Impact": [0.3017, -0.1594, 0.1735, -0.2430, 0.1520, 0.1326],
            "p-value": ["0.00207 (**)", "0.02953 (*)", "0.04160 (*)", "0.00547 (**)", "0.07545 (•)", "0.06663 (•)"],
            "Hypothesis Interpretation": [
                "Strong short-term network stability persistence.",
                "Immediate volatility spikes dissolve/disrupt network clusters.",
                "S&P 500 sectors perform a positive rotational rebound.",
                "Secondary corrective realignment wave in US equities.",
                "Long-memory structure settles back into new modularity bounds.",
                "Final long-term equilibrium stabilization wave complete."
            ]
        }
        str_lit.table(pd.DataFrame(coefficients_data))
        
        # Diagnostic performance box
        str_lit.metric("Residual Durbin-Watson statistic (Clean Residual Target ~ 2.0)", "2.0106", help="Confirms total elimination of autocorrelation bias.")
        
    elif target_market == 'nifty50':
        str_lit.markdown("### Model Structure: `Y = ari_stability` | `X = market_volatility_x` | Bounded Max Lags: `ARDL(4, 4)`")
        
        coefficients_data = {
            "Variable Parameter": [
                "ari_stability_lag_1 (Autoregressive Memory)", 
                "ari_stability_lag_2 (Autoregressive Memory)"
            ],
            "Coefficient Impact": [0.3339, 0.1472],
            "p-value": ["0.00003 (***)", "0.05762 (•)"],
            "Hypothesis Interpretation": [
                "Highly significant positive network-state momentum.",
                "Consistent persistence of structural boundaries."
            ]
        }
        str_lit.table(pd.DataFrame(coefficients_data))
        str_lit.warning("⚠️ **Statistical Insight:** Volatility index lags (1-4) were not statistically significant in the ARDL model. NIFTY 50 community structures exhibit powerful internal auto-regressive momentum, largely shrugging off short-term fluctuations in implied fear indexes.")
        str_lit.metric("Residual Durbin-Watson statistic (Clean Residual Target ~ 2.0)", "2.0073")

    elif target_market == 'bovespa':
        str_lit.markdown("### Model Structure: `Y = market_volatility_x` (Inverted Target) | `X = ari_stability` | Bounded Max Lags: `ARDL(2, 2)`")
        
        coefficients_data = {
            "Variable Parameter": [
                "market_volatility_x_lag_1 (Autoregressive Memory)", 
                "ari_stability_lag_1 (Leading Network Indicator)"
            ],
            "Coefficient Impact": [0.4246, 0.1143],
            "p-value": ["0.00000 (***)", "0.00734 (**)"],
            "Hypothesis Interpretation": [
                "Standard strong persistence of volatility parameters.",
                "Crystallizing/tightening BOVESPA communities significantly predict rising local implied volatility 24 hours ahead."
            ]
        }
        str_lit.table(pd.DataFrame(coefficients_data))
        str_lit.metric("Residual Durbin-Watson statistic (Clean Residual Target ~ 2.0)", "1.9871")

# ---------------------------------------------------------------------------
# FOOTER & SUBMISSION BRANDING
# ---------------------------------------------------------------------------
str_lit.markdown("---")
str_lit.caption("University of Edinburgh — Master of Science in Data Science Dissertation. August Submission Timeline.")