import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.agent import PPOAgent
from src.portfolio.risk_engine import RiskEngine
from src.backtesting.engine import BacktestEngine
from src.backtesting.simulator_bridge import VirtualTradingBridge

st.set_page_config(
    page_title="AI Multi-Objective Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 32px; font-weight: 800; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px; text-align: center; }
    .badge-bull { background-color: #DCFCE7; color: #15803D; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .badge-bear { background-color: #FEE2E2; color: #B91C1C; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .badge-vol { background-color: #FEF3C7; color: #B45309; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Load Datasets
PRICES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean_prices.csv")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features", "portfolio_features.csv")
REGIMES_PATH = os.path.join(PROJECT_ROOT, "data", "features", "market_regimes.csv")
EXP_RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "research_results", "experiment_results.csv")

prices_df = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
features_df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
regimes_df = pd.read_csv(REGIMES_PATH, index_col=0, parse_dates=True)
results_df = pd.read_csv(EXP_RESULTS_PATH) if os.path.exists(EXP_RESULTS_PATH) else None

# Header
st.markdown('<div class="main-title">📈 AI-Driven Adaptive Portfolio Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Objective Deep Reinforcement Learning (PPO) & Modern Portfolio Theory | IILM University Major Project</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Portfolio Configuration")
capital = st.sidebar.number_input("Investment Capital (₹)", min_value=10000.0, max_value=100000000.0, value=1000000.0, step=50000.0)
risk_profile = st.sidebar.selectbox("Risk Tolerance Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
horizon = st.sidebar.selectbox("Investment Horizon", ["1 Year", "2 Years", "3 Years", "5 Years"], index=1)

all_tickers = list(prices_df.columns)
selected_assets = st.sidebar.multiselect("Asset Universe", all_tickers, default=all_tickers)

# Model Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 RL Policy Controls")
use_regime = st.sidebar.checkbox("Enable Regime-Aware Policy", value=True)
apply_risk_guardrails = st.sidebar.checkbox("Enforce 35% Asset Cap", value=True)

# Current Market Regime
latest_regime = str(regimes_df["regime"].iloc[-1])
latest_vol = float(regimes_df["avg_volatility"].iloc[-1]) * 100

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Investment Capital", f"₹{capital:,.0f}")
with col_m2:
    if latest_regime == "BULL":
        reg_badge = '<span class="badge-bull">🟢 BULLISH</span>'
    elif latest_regime == "BEAR":
        reg_badge = '<span class="badge-bear">🔴 BEARISH</span>'
    else:
        reg_badge = '<span class="badge-vol">🟡 HIGH VOLATILITY</span>'
    st.markdown(f"**Current Regime**<br>{reg_badge}", unsafe_allow_html=True)
with col_m3:
    st.metric("Market Volatility", f"{latest_vol:.2f}%")
with col_m4:
    st.metric("Target Assets Selected", len(selected_assets))

st.markdown("---")

# Tabs
tab_recom, tab_benchmark, tab_explain, tab_simulator = st.tabs([
    "🎯 AI Recommendation",
    "📊 Benchmark Comparison",
    "🧠 Explainability Engine",
    "⚙️ Virtual Trading Engine"
])

# Compute Recommendation
with tab_recom:
    col_left, col_right = st.columns([1, 1])

    # Run PPO inference
    try:
        agent = PPOAgent(include_regime=use_regime)
        recent_p = prices_df.iloc[-40:]
        recent_f = features_df.iloc[-40:]
        recent_r = regimes_df.iloc[-40:]
        w_df = agent.generate_weights_for_dataset(recent_p, recent_f, recent_r)
        raw_weights = w_df.iloc[-1].values
    except Exception as e:
        raw_weights = np.ones(len(all_tickers)) / len(all_tickers)

    if apply_risk_guardrails:
        cap = 0.25 if risk_profile == "Conservative" else (0.45 if risk_profile == "Aggressive" else 0.35)
        risk_eng = RiskEngine(max_asset_weight=cap)
        weights = risk_eng.apply_constraints(raw_weights)
    else:
        weights = raw_weights / np.sum(raw_weights)

    alloc_df = pd.DataFrame({
        "Asset": all_tickers,
        "Weight (%)": [w * 100 for w in weights],
        "Allocated Capital (₹)": [w * capital for w in weights]
    })
    alloc_df = alloc_df[alloc_df["Asset"].isin(selected_assets)]
    # Renormalize for selected subset
    alloc_df["Weight (%)"] = (alloc_df["Weight (%)"] / alloc_df["Weight (%)"].sum()) * 100
    alloc_df["Allocated Capital (₹)"] = (alloc_df["Weight (%)"] / 100.0) * capital

    with col_left:
        st.subheader("Recommended Portfolio Allocation")
        st.dataframe(alloc_df.style.format({
            "Weight (%)": "{:.2f}%",
            "Allocated Capital (₹)": "₹{:,.2f}"
        }), use_container_width=True)

        st.info(f"💡 **Strategy Insight**: Under the current **{latest_regime}** condition, the PPO policy balances returns with risk penalties ($\lambda_{{risk}}$) to ensure drawdowns remain protected.")

    with col_right:
        fig_donut = px.pie(
            alloc_df,
            values="Weight (%)",
            names="Asset",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Optimal Simplex Capital Distribution"
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

# Benchmark Tab
with tab_benchmark:
    st.subheader("Out-of-Sample Performance Comparison (2024–2025 Unseen Data)")
    if results_df is not None:
        st.dataframe(results_df, use_container_width=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.image(os.path.join(PROJECT_ROOT, "data", "research_results", "cumulative_returns.png"), caption="Out-of-Sample Cumulative Portfolio Growth (Base: ₹10,00,000)")
    with col_b2:
        st.image(os.path.join(PROJECT_ROOT, "data", "research_results", "drawdowns.png"), caption="Maximum Drawdown Curves Over Time (%)")

# Explainability Tab
with tab_explain:
    st.subheader("🧠 Model Explainability: Why This Allocation?")
    st.markdown("""
    The Deep Reinforcement Learning agent does not blindly pick stocks. It optimizes a **Multi-Objective Reward Utility Function**:
    $$\\text{Reward} = \\text{Portfolio Return} - \\lambda_{\\text{risk}} \\cdot \\text{Variance} - \\lambda_{\\text{cost}} \\cdot \\text{Turnover}$$
    """)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(f"""
        ### 1. Market Regime Impact
        - **Current Regime**: `{latest_regime}`
        - **State Encoding**: `[1, 0, 0]` Bull | `[0, 1, 0]` Bear | `[0, 0, 1]` High Vol
        - In high volatility or bear regimes, the agent dampens equity exposure and redistributes capital towards less correlated assets.
        """)
        st.image(os.path.join(PROJECT_ROOT, "data", "research_results", "regime_distribution.png"), caption="Historical Market Regime Classification (2016-2025)")

    with col_e2:
        st.markdown("""
        ### 2. Cross-Asset Risk & Correlation
        - Highly correlated assets increase portfolio variance ($w^T \\Sigma w$).
        - The model penalizes heavy concentration in co-moving high-beta stocks to protect against systemic sell-offs.
        """)
        st.image(os.path.join(PROJECT_ROOT, "data", "research_results", "correlation_heatmap.png"), caption="Cross-Asset Rolling Correlation Matrix")

# Virtual Trading Simulator Tab
with tab_simulator:
    st.subheader("⚙️ Virtual Trading Execution Log (Java Simulator Engine)")
    st.markdown("Simulating chronological order fills, slippage, and 10 bps transaction fees:")

    sim_bridge = VirtualTradingBridge(initial_capital=capital)
    latest_p = prices_df.iloc[-1].to_dict()
    sim_weights = dict(zip(alloc_df["Asset"], alloc_df["Weight (%)"] / 100.0))
    sim_out = sim_bridge.execute_rebalance(sim_weights, latest_p)

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Post-Trade Portfolio Value", f"₹{sim_out['portfolio_value']:,.2f}")
    col_s2.metric("Remaining Cash Reserve", f"₹{sim_out['cash_balance']:,.2f}")
    col_s3.metric("Transaction Fees Paid (10 bps)", f"₹{sim_out['fees_paid']:,.2f}")

    if sim_out["trades"]:
        st.markdown("#### Executed Orders:")
        trades_df = pd.DataFrame(sim_out["trades"])
        st.dataframe(trades_df, use_container_width=True)
