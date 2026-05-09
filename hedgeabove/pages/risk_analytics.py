"""Risk Analytics page."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from hedgeabove.data.market import get_current_price, get_historical_data
from hedgeabove.models.risk import (
    calculate_var, calculate_es, calculate_portfolio_metrics,
    calculate_tail_dependence,
)
from hedgeabove.models.timeseries import (
    fit_garch_model, forecast_volatility, garch_var,
    extract_volatility_regimes,
)


def render():
    st.header("Risk Analytics")

    if len(st.session_state.portfolio) == 0:
        st.warning("Please add positions to your portfolio first!")
        return

    symbols = st.session_state.portfolio['Symbol'].tolist()
    weights = st.session_state.portfolio['Shares'].values
    weights = weights / weights.sum()

    with st.spinner("Fetching historical data..."):
        hist_data = get_historical_data(symbols, period='1y')

    if hist_data.empty:
        st.error("Could not fetch historical data. Please try again.")
        return

    returns = hist_data.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)

    # Portfolio Metrics
    st.subheader("Portfolio Risk Metrics")
    metrics = calculate_portfolio_metrics(portfolio_returns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ann. Return", f"{metrics['Annualized Return']*100:.2f}%")
    col2.metric("Ann. Volatility", f"{metrics['Annualized Volatility']*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    col4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")

    col1, col2, _, _ = st.columns(4)
    col1.metric("Total Return", f"{metrics['Total Return']*100:.2f}%")
    col2.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")

    # VaR and ES
    st.subheader("Value at Risk & Expected Shortfall")

    col1, col2 = st.columns(2)
    with col1:
        confidence_level = st.slider("Confidence Level", 90, 99, 95) / 100
    with col2:
        time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"])

    var_hist = calculate_var(portfolio_returns, confidence_level, 'historical')
    var_param = calculate_var(portfolio_returns, confidence_level, 'parametric')
    var_mc = calculate_var(portfolio_returns, confidence_level, 'monte_carlo')
    es = calculate_es(portfolio_returns, confidence_level)

    portfolio_value = (st.session_state.portfolio['Shares'] *
                       st.session_state.portfolio['Symbol'].apply(get_current_price)).sum()

    horizon_days = {'1 Day': 1, '1 Week': 5, '1 Month': 21}[time_horizon]
    scale = np.sqrt(horizon_days)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR (Historical)", f"${abs(var_hist * portfolio_value * scale):,.2f}")
    col2.metric("VaR (Parametric)", f"${abs(var_param * portfolio_value * scale):,.2f}")
    col3.metric("VaR (Monte Carlo)", f"${abs(var_mc * portfolio_value * scale):,.2f}")
    col4.metric("Expected Shortfall", f"${abs(es * portfolio_value * scale):,.2f}")

    st.caption(f"Maximum expected loss at {confidence_level*100}% confidence over {time_horizon}")

    # Returns Distribution
    st.subheader("Returns Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=portfolio_returns * 100, nbinsx=50, name='Returns', marker_color='lightblue'))
    fig.add_vline(x=var_hist*100, line_dash="dash", line_color="red", annotation_text=f"VaR ({confidence_level*100}%)")
    fig.add_vline(x=portfolio_returns.mean()*100, line_dash="dash", line_color="green", annotation_text="Mean")
    fig.update_layout(title="Portfolio Daily Returns Distribution", xaxis_title="Daily Return (%)",
                      yaxis_title="Frequency", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation Matrix
    st.subheader("Asset Correlation Matrix")
    corr_matrix = returns.corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', color_continuous_scale='RdYlGn',
                    title='Correlation Matrix of Portfolio Assets')
    st.plotly_chart(fig, use_container_width=True)

    # Copula Analysis
    _render_copula_section(symbols, returns)

    st.markdown("---")

    # GARCH
    _render_garch_section(portfolio_returns)

    st.markdown("---")

    # Rolling Volatility
    st.subheader("Rolling Volatility (30-day)")
    rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode='lines',
                             name='Rolling Volatility', line=dict(color='orange', width=2)))
    fig.update_layout(title="30-Day Rolling Volatility (Annualized)",
                      xaxis_title="Date", yaxis_title="Volatility (%)", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def _render_copula_section(symbols, returns):
    st.subheader("Tail Dependence & Copula Analysis")

    if len(symbols) < 2:
        st.info("Add at least 2 assets to analyze tail dependence")
        return

    col1, col2 = st.columns(2)
    with col1:
        asset1 = st.selectbox("Asset 1", symbols, key='copula_asset1')
    with col2:
        asset2 = st.selectbox("Asset 2", [s for s in symbols if s != asset1], key='copula_asset2')

    copula_type = st.selectbox("Copula Type", ["Gaussian", "t-Copula", "Clayton", "Gumbel"])

    if st.button("Analyze Tail Dependence"):
        with st.spinner("Fitting copula model..."):
            returns1 = returns[asset1].dropna()
            returns2 = returns[asset2].dropna()
            common_index = returns1.index.intersection(returns2.index)
            returns1 = returns1.loc[common_index]
            returns2 = returns2.loc[common_index]

            tail_dep = calculate_tail_dependence(returns1, returns2,
                                                  copula_type.lower().replace('-', ''))

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Spearman's rho", f"{tail_dep['spearman_rho']:.3f}")
            col2.metric("Kendall's tau", f"{tail_dep['kendall_tau']:.3f}")
            col3.metric("Upper Tail", f"{tail_dep['upper_tail']:.3f}")
            col4.metric("Lower Tail", f"{tail_dep['lower_tail']:.3f}")

            tail_label = {
                'Gaussian': 'No tail dependence',
                't-Copula': 'Symmetric tail dependence',
                'Clayton': 'Lower tail dependence (crash risk)',
                'Gumbel': 'Upper tail dependence (boom risk)',
            }.get(copula_type, '')

            st.info(f"**{copula_type}**: {tail_label}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=returns1, y=returns2, mode='markers',
                                     marker=dict(size=5, opacity=0.6, color='blue'), name='Returns'))
            fig.update_layout(title=f"{asset1} vs {asset2} Returns with {copula_type}",
                              xaxis_title=f"{asset1} Returns", yaxis_title=f"{asset2} Returns")
            st.plotly_chart(fig, use_container_width=True)


def _render_garch_section(portfolio_returns):
    st.subheader("GARCH Volatility Forecasting & Regimes")

    if len(portfolio_returns) <= 100:
        st.info("Add more portfolio history (>100 days) for GARCH analysis")
        return

    col1, col2 = st.columns(2)
    with col1:
        forecast_horizon = st.slider("Forecast Horizon (days)", 5, 60, 30)
    with col2:
        garch_p = st.selectbox("GARCH p (lag)", [1, 2], index=0)
        garch_q = st.selectbox("GARCH q (lag)", [1, 2], index=0)

    if not st.button("Run GARCH Analysis"):
        return

    with st.spinner("Fitting GARCH model..."):
        garch_result = fit_garch_model(portfolio_returns, p=garch_p, q=garch_q)

    if not garch_result:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("AIC", f"{garch_result['aic']:.2f}")
    col2.metric("BIC", f"{garch_result['bic']:.2f}")
    current_vol = garch_result['conditional_volatility'].iloc[-1] / 100
    col3.metric("Current Volatility", f"{current_vol*np.sqrt(252)*100:.2f}%")

    vol_forecast = forecast_volatility(garch_result['model'], horizon=forecast_horizon)

    if vol_forecast:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_returns.index,
            y=garch_result['conditional_volatility'] / 100 * np.sqrt(252) * 100,
            mode='lines', name='Conditional Volatility', line=dict(color='orange', width=2),
        ))
        forecast_dates = pd.date_range(start=portfolio_returns.index[-1],
                                        periods=forecast_horizon + 1, freq='D')[1:]
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=vol_forecast['volatility'] * np.sqrt(252),
            mode='lines', name='Volatility Forecast', line=dict(color='red', width=2, dash='dash'),
        ))
        fig.update_layout(title=f"GARCH({garch_p},{garch_q}) Volatility: Historical & Forecast",
                          xaxis_title="Date", yaxis_title="Annualized Volatility (%)", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        regimes = extract_volatility_regimes(garch_result['conditional_volatility'])
        if regimes:
            col1, col2 = st.columns(2)
            col1.metric("High Vol Regime Avg", f"{regimes['high_vol_mean']/100*np.sqrt(252)*100:.2f}%")
            col2.metric("Low Vol Regime Avg", f"{regimes['low_vol_mean']/100*np.sqrt(252)*100:.2f}%")

        garch_var_95 = garch_var(portfolio_returns, garch_result['model'], confidence=0.95)
        if garch_var_95:
            st.info(f"**GARCH-Based 1-Day VaR (95%):** {garch_var_95:.4f} ({garch_var_95*100:.2f}%)")
