"""Home page."""

import streamlit as st


def render():
    st.header("Welcome to HedgeAbove")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Portfolio Analytics")
        st.write("Build and track portfolios with real-time market data from Yahoo Finance.")

    with col2:
        st.markdown("### Risk Management")
        st.write("VaR, Expected Shortfall, correlation analysis, and comprehensive risk metrics.")

    with col3:
        st.markdown("### Portfolio Optimization")
        st.write("Modern Portfolio Theory, Efficient Frontier, and optimal allocation strategies.")

    st.markdown("---")

    st.subheader("New Features")

    features = {
        "Portfolio Builder": [
            "Add/Edit/Delete positions",
            "Real-time price updates (yfinance)",
            "Automatic P&L tracking",
            "Persistent storage (SQLite)",
        ],
        "Risk Analytics": [
            "Value at Risk (Historical, Parametric, Monte Carlo)",
            "Expected Shortfall (CVaR)",
            "Correlation matrices with real data",
            "GARCH volatility & copula tail dependence",
        ],
        "Portfolio Optimization": [
            "Efficient Frontier visualization",
            "Max Sharpe / Min Volatility / Risk Parity",
            "Target Return optimization",
            "Rebalancing suggestions",
        ],
        "Stock Screener": [
            "2,535+ global stocks, 50+ metrics",
            "Filter by sector, valuation, profitability",
            "Preset strategies (Value, Growth, Quality, Dividend)",
            "Quick-add to portfolio",
        ],
    }

    col1, col2 = st.columns(2)

    with col1:
        for feature, items in list(features.items())[:2]:
            with st.expander(f"**{feature}**", expanded=True):
                for item in items:
                    st.write(f"- {item}")

    with col2:
        for feature, items in list(features.items())[2:]:
            with st.expander(f"**{feature}**", expanded=True):
                for item in items:
                    st.write(f"- {item}")

    st.markdown("---")
    st.success("Get Started: Select 'Stock Screener' to find stocks, then build your portfolio!")
