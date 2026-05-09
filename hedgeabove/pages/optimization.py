"""Portfolio Optimization page."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from hedgeabove.data.market import get_historical_data, get_current_price
from hedgeabove.models.risk import calculate_portfolio_metrics
from hedgeabove.models.optimization import optimize_portfolio, generate_efficient_frontier


def render():
    st.header("Modern Portfolio Theory & Optimization")

    if len(st.session_state.portfolio) < 2:
        st.warning("Please add at least 2 positions to your portfolio for optimization!")
        return

    symbols = st.session_state.portfolio['Symbol'].tolist()

    with st.spinner("Fetching historical data and calculating optimal portfolios..."):
        hist_data = get_historical_data(symbols, period='1y')

    if hist_data.empty:
        st.error("Could not fetch historical data. Please try again.")
        return

    returns = hist_data.pct_change().dropna()

    current_weights = st.session_state.portfolio['Shares'].values
    current_weights = current_weights / current_weights.sum()
    current_returns = (returns * current_weights).sum(axis=1)
    current_metrics = calculate_portfolio_metrics(current_returns)

    # Optimization Methods
    st.subheader("Optimal Portfolio Allocations")

    col1, col2 = st.columns(2)
    with col1:
        opt_methods = st.multiselect(
            "Select optimization methods to compare:",
            ["Max Sharpe Ratio", "Min Volatility", "Target Return", "Risk Parity"],
            default=["Max Sharpe Ratio", "Min Volatility"],
        )
    with col2:
        target_return = None
        if "Target Return" in opt_methods:
            target_return = st.slider("Target Annual Return (%)", 0, 50, 15) / 100

    optimal_portfolios = {}
    method_map = {
        "Max Sharpe Ratio": ('max_sharpe', 'Max Sharpe'),
        "Min Volatility": ('min_vol', 'Min Volatility'),
        "Target Return": ('target_return', 'Target Return'),
        "Risk Parity": ('risk_parity', 'Risk Parity'),
    }

    for method_name in opt_methods:
        method_key, label = method_map[method_name]
        kwargs = {}
        if method_key == 'target_return' and target_return:
            kwargs['target_return'] = target_return
        elif method_key == 'target_return' and not target_return:
            continue
        w = optimize_portfolio(returns, method_key, **kwargs)
        opt_ret = (returns * w).sum(axis=1)
        optimal_portfolios[label] = {
            'weights': w,
            'metrics': calculate_portfolio_metrics(opt_ret),
        }

    # Allocation comparison
    st.subheader("Optimal Allocations Comparison")
    allocation_df = pd.DataFrame(index=symbols)
    allocation_df['Current'] = current_weights * 100
    for name, portfolio in optimal_portfolios.items():
        allocation_df[name] = portfolio['weights'] * 100

    st.dataframe(
        allocation_df.style.format("{:.2f}%").background_gradient(cmap='Blues', axis=1),
        use_container_width=True,
    )

    # Metrics Comparison
    st.subheader("Performance Metrics Comparison")
    metrics_df = pd.DataFrame()
    metrics_df['Current Portfolio'] = pd.Series(current_metrics)
    for name, portfolio in optimal_portfolios.items():
        metrics_df[name] = pd.Series(portfolio['metrics'])

    display_metrics = metrics_df.T[['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown']]
    st.dataframe(
        display_metrics.style.format({
            'Annualized Return': '{:.2%}', 'Annualized Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}', 'Max Drawdown': '{:.2%}',
        }).background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn', vmin=-1, vmax=3),
        use_container_width=True,
    )

    # Efficient Frontier
    st.subheader("Efficient Frontier")
    with st.spinner("Generating efficient frontier..."):
        frontier_rets, frontier_vols, _ = generate_efficient_frontier(returns, num_portfolios=50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.array(frontier_vols) * 100, y=np.array(frontier_rets) * 100,
        mode='lines', name='Efficient Frontier', line=dict(color='blue', width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[current_metrics['Annualized Volatility'] * 100],
        y=[current_metrics['Annualized Return'] * 100],
        mode='markers', name='Current Portfolio', marker=dict(color='red', size=15, symbol='star'),
    ))
    colors = ['green', 'purple', 'orange', 'brown']
    for idx, (name, portfolio) in enumerate(optimal_portfolios.items()):
        fig.add_trace(go.Scatter(
            x=[portfolio['metrics']['Annualized Volatility'] * 100],
            y=[portfolio['metrics']['Annualized Return'] * 100],
            mode='markers', name=name, marker=dict(color=colors[idx % len(colors)], size=12),
        ))
    fig.update_layout(title="Efficient Frontier & Portfolio Comparison",
                      xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
                      hovermode='closest', showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Rebalancing Suggestions
    st.subheader("Rebalancing Suggestions")
    if optimal_portfolios:
        selected_portfolio = st.selectbox("Select optimal portfolio to view rebalancing actions:",
                                           list(optimal_portfolios.keys()))
        if selected_portfolio:
            optimal_weights = optimal_portfolios[selected_portfolio]['weights']
            portfolio_value = (st.session_state.portfolio['Shares'] *
                               st.session_state.portfolio['Symbol'].apply(get_current_price)).sum()

            rebalance_df = pd.DataFrame({
                'Symbol': symbols,
                'Current %': current_weights * 100,
                'Target %': optimal_weights * 100,
                'Difference %': (optimal_weights - current_weights) * 100,
                'Current Value': current_weights * portfolio_value,
                'Target Value': optimal_weights * portfolio_value,
                'Action $': (optimal_weights - current_weights) * portfolio_value,
            })
            rebalance_df['Action'] = rebalance_df['Action $'].apply(
                lambda x: f"Buy ${abs(x):,.2f}" if x > 0 else f"Sell ${abs(x):,.2f}" if x < 0 else "Hold"
            )
            st.dataframe(
                rebalance_df.style.format({
                    'Current %': '{:.2f}%', 'Target %': '{:.2f}%', 'Difference %': '{:+.2f}%',
                    'Current Value': '${:,.2f}', 'Target Value': '${:,.2f}', 'Action $': '${:+,.2f}',
                }).background_gradient(subset=['Difference %'], cmap='RdYlGn', vmin=-10, vmax=10),
                use_container_width=True,
            )
