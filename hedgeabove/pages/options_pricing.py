"""Options Pricing page — Black-Scholes, Greeks, strategy builder, IV."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from hedgeabove.data.market import get_stock_info, get_historical_data
from hedgeabove.models.options import (
    black_scholes, calculate_greeks, implied_volatility, option_payoff,
)


def render():
    st.header("Options Pricing & Greeks Calculator")
    st.caption("Black-Scholes pricing model with full Greeks analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Option Pricing", "Options Strategy Builder",
        "Greeks Surface", "Implied Volatility",
    ])

    with tab1:
        _render_single_option()
    with tab2:
        _render_strategy_builder()
    with tab3:
        _render_greeks_surface()
    with tab4:
        _render_implied_vol()


def _render_single_option():
    st.subheader("Black-Scholes Option Pricing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Option Parameters")

        ticker_input = st.text_input("Stock Ticker (optional - for live price)", value="AAPL", key='bs_ticker')
        fetch_price = st.checkbox("Fetch current price", value=False, key='fetch_price')

        S_default = 100.0
        sigma_default = 0.30
        stock_info_fetched = None

        if fetch_price and ticker_input:
            with st.spinner(f"Fetching {ticker_input} data..."):
                stock_info_fetched = get_stock_info(ticker_input)
                if stock_info_fetched:
                    S_default = stock_info_fetched['current_price']
                    sigma_default = stock_info_fetched.get('beta', 0.3) * 0.20 if stock_info_fetched.get('beta') else 0.30
                    st.success(f"Current price: ${S_default:.2f}")
                else:
                    st.warning("Could not fetch price, using default")

        S = st.number_input("Current Stock Price ($)", value=float(S_default), min_value=0.01, step=1.0, key='bs_S')
        K = st.number_input("Strike Price ($)", value=float(S_default), min_value=0.01, step=1.0, key='bs_K')
        T = st.number_input("Time to Expiration (days)", value=30, min_value=1, step=1, key='bs_T') / 365.0
        r = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1, key='bs_r') / 100
        sigma = st.number_input("Volatility (% annual)", value=sigma_default * 100, min_value=1.0, max_value=200.0, step=1.0, key='bs_sigma') / 100
        option_type = st.selectbox("Option Type", ["call", "put"], key='bs_type')

    with col2:
        st.markdown("#### Pricing Results")

        option_price = black_scholes(S, K, T, r, sigma, option_type)
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)

        st.metric("Option Price", f"${option_price:.4f}")

        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        time_value = option_price - intrinsic

        ca, cb = st.columns(2)
        ca.metric("Intrinsic Value", f"${intrinsic:.4f}")
        cb.metric("Time Value", f"${time_value:.4f}")

        moneyness = S / K
        if moneyness > 1.02:
            money_status = "In-the-Money (ITM)" if option_type == 'call' else "Out-of-the-Money (OTM)"
        elif moneyness < 0.98:
            money_status = "Out-of-the-Money (OTM)" if option_type == 'call' else "In-the-Money (ITM)"
        else:
            money_status = "At-the-Money (ATM)"
        st.info(f"**Status:** {money_status} (S/K = {moneyness:.4f})")

    st.markdown("---")
    st.subheader("The Greeks")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Delta", f"{greeks['delta']:.4f}", help="$1 stock move sensitivity")
    col2.metric("Gamma", f"{greeks['gamma']:.4f}", help="Delta change rate")
    col3.metric("Theta", f"{greeks['theta']:.4f}", help="Daily time decay")
    col4.metric("Vega", f"{greeks['vega']:.4f}", help="1% vol sensitivity")
    col5.metric("Rho", f"{greeks['rho']:.4f}", help="1% rate sensitivity")

    # Payoff diagram
    st.markdown("---")
    st.subheader("Payoff Diagram at Expiration")

    S_range = np.linspace(S * 0.7, S * 1.3, 100)
    payoff_long = option_payoff(S_range, K, option_price, option_type, 'long')
    payoff_short = option_payoff(S_range, K, option_price, option_type, 'short')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=payoff_long, mode='lines',
                             name=f'Long {option_type.capitalize()}', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=S_range, y=payoff_short, mode='lines',
                             name=f'Short {option_type.capitalize()}', line=dict(color='red', width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=S, line_dash="dot", line_color="blue", opacity=0.5, annotation_text="Current Price")
    fig.add_vline(x=K, line_dash="dot", line_color="orange", opacity=0.5, annotation_text="Strike")
    fig.update_layout(title=f"{option_type.capitalize()} Option Payoff at Expiration",
                      xaxis_title="Stock Price at Expiration ($)", yaxis_title="Profit/Loss ($)",
                      hovermode='x unified', height=500)
    st.plotly_chart(fig, use_container_width=True)

    breakeven = K + option_price if option_type == 'call' else K - option_price
    max_loss = option_price
    max_gain = "Unlimited" if option_type == 'call' else f"${K - option_price:.2f}"

    col1, col2, col3 = st.columns(3)
    col1.metric("Break-Even Price", f"${breakeven:.2f}")
    col2.metric("Max Loss (Long)", f"${max_loss:.2f}")
    col3.metric("Max Gain (Long)", max_gain)


def _render_strategy_builder():
    st.subheader("Multi-Leg Options Strategy Builder")
    st.info("Coming Soon: Build complex strategies like spreads, straddles, iron condors, and more!")

    st.markdown("### Popular Strategies")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Bull Call Spread")
        st.write("- Buy Call (lower strike)")
        st.write("- Sell Call (higher strike)")
        st.write("- Limited risk, limited reward")
    with col2:
        st.markdown("#### Straddle")
        st.write("- Buy Call (ATM)")
        st.write("- Buy Put (ATM)")
        st.write("- Profit from volatility")
    with col3:
        st.markdown("#### Iron Condor")
        st.write("- Sell Call & Put (ATM)")
        st.write("- Buy Call & Put (OTM)")
        st.write("- Profit from low volatility")


def _render_greeks_surface():
    st.subheader("Greeks Surface Visualization")

    greek_to_plot = st.selectbox("Select Greek to Visualize", ["Delta", "Gamma", "Theta", "Vega"], key='greek_surface')
    S_surface = st.slider("Current Stock Price", 50.0, 200.0, 100.0, 5.0, key='surface_S')
    T_days = st.slider("Days to Expiration", 1, 180, 30, 1, key='surface_T')
    r_surface = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.5, key='surface_r') / 100
    option_type_surface = st.selectbox("Option Type", ["call", "put"], key='surface_type')

    if not st.button("Generate Greeks Surface", type="primary", key='gen_surface'):
        return

    with st.spinner("Generating 3D surface..."):
        strikes = np.linspace(S_surface * 0.7, S_surface * 1.3, 30)
        volatilities = np.linspace(0.1, 1.0, 30)
        K_grid, Sigma_grid = np.meshgrid(strikes, volatilities)
        Greek_grid = np.zeros_like(K_grid)
        T_years = T_days / 365.0

        for i in range(len(volatilities)):
            for j in range(len(strikes)):
                greeks = calculate_greeks(S_surface, strikes[j], T_years, r_surface,
                                          volatilities[i], option_type_surface)
                Greek_grid[i, j] = greeks[greek_to_plot.lower()]

        fig = go.Figure(data=[go.Surface(x=K_grid, y=Sigma_grid * 100, z=Greek_grid, colorscale='Viridis')])
        fig.update_layout(
            title=f"{greek_to_plot} Surface for {option_type_surface.capitalize()} Option",
            scene=dict(xaxis_title="Strike Price ($)", yaxis_title="Volatility (%)", zaxis_title=greek_to_plot),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Spot Price:** ${S_surface} | **Days to Expiration:** {T_days} | **Risk-Free Rate:** {r_surface*100:.1f}%")


def _render_implied_vol():
    st.subheader("Implied Volatility Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Input Parameters")
        S_iv = st.number_input("Current Stock Price ($)", value=100.0, min_value=0.01, step=1.0, key='iv_S')
        K_iv = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=1.0, key='iv_K')
        T_iv = st.number_input("Time to Expiration (days)", value=30, min_value=1, step=1, key='iv_T') / 365.0
        r_iv = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1, key='iv_r') / 100
        option_price_iv = st.number_input("Market Option Price ($)", value=5.0, min_value=0.01, step=0.1, key='iv_price')
        option_type_iv = st.selectbox("Option Type", ["call", "put"], key='iv_type')

    with col2:
        st.markdown("#### Results")

        if st.button("Calculate Implied Volatility", type="primary", key='calc_iv'):
            with st.spinner("Calculating implied volatility..."):
                iv = implied_volatility(option_price_iv, S_iv, K_iv, T_iv, r_iv, option_type_iv)

                if iv is not None:
                    st.success("Convergence achieved!")
                    st.metric("Implied Volatility", f"{iv * 100:.2f}%")

                    calculated_price = black_scholes(S_iv, K_iv, T_iv, r_iv, iv, option_type_iv)
                    st.metric("Recalculated Price", f"${calculated_price:.4f}")
                    st.caption(f"Market Price: ${option_price_iv:.4f} | Difference: ${abs(calculated_price - option_price_iv):.4f}")
                else:
                    st.error("Failed to converge. Check input parameters.")

    st.markdown("---")
    st.markdown("### Understanding Implied Volatility")
    st.write(
        "**Implied Volatility (IV)** is the market's forecast of a likely movement in a security's price. "
        "It's derived from the option's market price using the Black-Scholes model.\n\n"
        "- **High IV**: Market expects large price swings (options are expensive)\n"
        "- **Low IV**: Market expects small price movements (options are cheap)\n"
        "- **IV > Historical Vol**: Options may be overpriced\n"
        "- **IV < Historical Vol**: Options may be underpriced"
    )
