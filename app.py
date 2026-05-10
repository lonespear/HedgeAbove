"""
HedgeAbove - AI-Powered Finance Analytics Platform
Entry point for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from hedgeabove.db import init_db, load_positions, list_portfolios, create_portfolio
from hedgeabove.pages import (
    home, screener, portfolio, risk_analytics, optimization,
    predictions, options_pricing, backtest, alerts, strategy_lab,
)

# ── Page configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="HedgeAbove - Rise Above Market Uncertainty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Database init ───────────────────────────────────────────────
init_db()

# ── Session state bootstrap ────────────────────────────────────
if 'portfolio' not in st.session_state:
    # Load from DB (create default portfolio if none exists)
    portfolios = list_portfolios()
    if not portfolios:
        pid = create_portfolio("My Portfolio")
    else:
        pid = portfolios[0][0]
    st.session_state.active_portfolio_id = pid
    st.session_state.portfolio = load_positions(pid)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Widen the sidebar ~20% so longer page names don't wrap */
    section[data-testid="stSidebar"] {
        width: 290px !important;
        min-width: 290px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 290px !important;
        min-width: 290px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────
st.sidebar.image("hedge_above_logo.png", use_container_width=True)
st.sidebar.markdown(
    "<p style='text-align: center; font-family: monospace; color: #888; font-size: 0.9rem;'>"
    "Rise Above the Status Quo</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Stock Screener", "Alerts & Watchlists", "Strategy Lab",
     "Portfolio Builder", "Risk Analytics",
     "Portfolio Optimization", "AI Predictions", "Backtesting", "Options Pricing"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Account")
st.sidebar.info("**Free Tier**\n\nUnlimited Portfolios\nReal-time data via yfinance")
st.sidebar.button("Upgrade to Pro", type="primary", use_container_width=True)

# ── Page dispatch ───────────────────────────────────────────────
PAGE_MAP = {
    "Home": home.render,
    "Stock Screener": screener.render,
    "Alerts & Watchlists": alerts.render,
    "Strategy Lab": strategy_lab.render,
    "Portfolio Builder": portfolio.render,
    "Risk Analytics": risk_analytics.render,
    "Portfolio Optimization": optimization.render,
    "AI Predictions": predictions.render,
    "Backtesting": backtest.render,
    "Options Pricing": options_pricing.render,
}

PAGE_MAP[page]()

# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "HedgeAbove v0.3.0 | Rise Above Market Uncertainty | "
    "<a href='https://github.com/lonespear/HedgeAbove'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
