"""
Backtesting & Model Comparison page.

Walk-forward validation of forecasting models with side-by-side metrics,
predicted vs actual charts, and error analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from hedgeabove.data.market import get_historical_data
from hedgeabove.models.backtest import (
    run_backtest, MODEL_REGISTRY,
    test_stationarity, test_arch_effects, test_normality, test_autocorrelation,
)


def render():
    st.header("Backtesting & Model Comparison")
    st.caption(
        "Walk-forward validation: train on past data, forecast ahead, "
        "compare against what actually happened — across multiple models simultaneously."
    )

    # ── Configuration ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL", key="bt_ticker")
        data_period = st.selectbox("Historical Data", ["1y", "2y", "3y", "5y"],
                                   index=1, key="bt_period")
    with col2:
        horizon = st.slider("Forecast Horizon (days)", 1, 30, 5, key="bt_horizon",
                            help="How many days ahead each forecast looks")
        step = st.slider("Step Size (days)", 1, 20, 5, key="bt_step",
                         help="How many days to advance between forecasts")

    train_pct = st.slider("Training Window (%)", 50, 90, 70, step=5, key="bt_train") / 100

    # Model selection
    st.subheader("Select Models to Compare")
    available = list(MODEL_REGISTRY.keys())
    labels = {k: MODEL_REGISTRY[k][0] for k in available}

    col1, col2, col3 = st.columns(3)
    selected_models = []
    for i, key in enumerate(available):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.checkbox(labels[key], value=key in ('naive', 'arima', 'mean_reversion'),
                           key=f"bt_model_{key}"):
                selected_models.append(key)

    if not selected_models:
        st.warning("Select at least one model.")
        return

    # ── Run backtest ────────────────────────────────────────────
    if not st.button("Run Backtest", type="primary", key="bt_run", use_container_width=True):
        return

    with st.spinner(f"Fetching {data_period} of data for {ticker}..."):
        hist = get_historical_data([ticker], period=data_period)

    if hist.empty:
        st.error(f"Could not fetch data for {ticker}")
        return

    prices = hist[ticker].dropna()
    st.info(f"**{ticker}**: {len(prices)} trading days, "
            f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

    # Assumption diagnostics
    _render_assumption_panel(prices)

    st.markdown("---")

    with st.spinner("Running walk-forward backtest... (this may take a minute for ARIMA/GARCH)"):
        results = run_backtest(prices, models=selected_models,
                               train_pct=train_pct, horizon=horizon, step=step)

    # ── Metrics table ───────────────────────────────────────────
    st.subheader("Model Performance Summary")

    metrics_df = results['metrics']

    # Highlight best values
    st.dataframe(
        metrics_df.style.format({
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'MAPE (%)': '{:.2f}',
            'Directional Accuracy (%)': '{:.1f}',
            'N Predictions': '{:.0f}',
        }).highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='#d4edda')
          .highlight_max(subset=['Directional Accuracy (%)'], color='#d4edda'),
        use_container_width=True,
    )

    # Interpretation
    if len(metrics_df) > 1:
        best_rmse = metrics_df['RMSE'].idxmin()
        best_dir = metrics_df['Directional Accuracy (%)'].idxmax()
        st.success(
            f"**Best price accuracy (RMSE):** {best_rmse}  |  "
            f"**Best directional accuracy:** {best_dir}"
        )

    # ── Predicted vs Actual chart ───────────────────────────────
    st.markdown("---")
    st.subheader("Predicted vs Actual Prices")

    forecasts = results['forecasts']
    if not forecasts:
        st.warning("No forecast data to display.")
        return

    fig = go.Figure()

    # Actual prices (full series)
    train_end = results['train_end_idx']
    fig.add_trace(go.Scatter(
        x=prices.index[:train_end], y=prices.values[:train_end],
        mode='lines', name='Training Data',
        line=dict(color='lightblue', width=1),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=prices.index[train_end:], y=prices.values[train_end:],
        mode='lines', name='Actual (Test Period)',
        line=dict(color='blue', width=2),
    ))

    # Each model's predictions
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'magenta']
    for i, (model_name, fdata) in enumerate(forecasts.items()):
        fig.add_trace(go.Scatter(
            x=fdata['dates'], y=fdata['predicted'],
            mode='markers', name=f'{model_name} (Predicted)',
            marker=dict(color=colors[i % len(colors)], size=3, opacity=0.5),
        ))

    fig.add_vline(x=prices.index[train_end], line_dash="dash", line_color="gray",
                  annotation_text="Train/Test Split")
    fig.update_layout(
        title=f"{ticker} — Walk-Forward Forecast Comparison",
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode='x unified', height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Error over time ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Forecast Error Over Time")

    error_tab1, error_tab2 = st.tabs(["Absolute Error", "Cumulative Error"])

    with error_tab1:
        fig_err = go.Figure()
        for i, (model_name, fdata) in enumerate(forecasts.items()):
            abs_error = np.abs(fdata['actual'] - fdata['predicted'])
            fig_err.add_trace(go.Scatter(
                x=fdata['dates'], y=abs_error,
                mode='lines', name=model_name,
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
        fig_err.update_layout(
            title="Absolute Prediction Error (|Actual - Predicted|)",
            xaxis_title="Date", yaxis_title="Absolute Error ($)",
            hovermode='x unified', height=400,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    with error_tab2:
        fig_cum = go.Figure()
        for i, (model_name, fdata) in enumerate(forecasts.items()):
            abs_error = np.abs(fdata['actual'] - fdata['predicted'])
            cum_error = np.cumsum(abs_error)
            fig_cum.add_trace(go.Scatter(
                x=fdata['dates'], y=cum_error,
                mode='lines', name=model_name,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_cum.update_layout(
            title="Cumulative Absolute Error",
            xaxis_title="Date", yaxis_title="Cumulative Error ($)",
            hovermode='x unified', height=400,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    # ── Error distribution ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Error Distribution by Model")

    error_data = []
    for model_name, fdata in forecasts.items():
        errors = fdata['actual'] - fdata['predicted']
        for e in errors:
            error_data.append({'Model': model_name, 'Error ($)': e})

    if error_data:
        error_df = pd.DataFrame(error_data)
        fig_dist = px.box(error_df, x='Model', y='Error ($)',
                          title="Prediction Error Distribution (Actual - Predicted)",
                          color='Model')
        fig_dist.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.caption(
            "**Positive errors** = model underestimated (actual was higher).  "
            "**Negative errors** = model overestimated.  "
            "A tight box around zero is ideal; bias shows systematic over/under-prediction."
        )

    # ── Regime analysis ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Performance by Volatility Regime")

    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    vol_median = rolling_vol.median()

    regime_rows = []
    for model_name, fdata in forecasts.items():
        for d, a, p in zip(fdata['dates'], fdata['actual'], fdata['predicted']):
            if d in rolling_vol.index:
                regime = "High Vol" if rolling_vol.loc[d] > vol_median else "Low Vol"
                regime_rows.append({
                    'Model': model_name,
                    'Regime': regime,
                    'Abs Error': abs(a - p),
                })

    if regime_rows:
        regime_df = pd.DataFrame(regime_rows)
        regime_summary = regime_df.groupby(['Model', 'Regime'])['Abs Error'].mean().unstack(fill_value=0)

        st.dataframe(
            regime_summary.style.format("{:.4f}")
            .background_gradient(cmap='RdYlGn_r', axis=None),
            use_container_width=True,
        )
        st.caption(
            "Average absolute error by volatility regime. "
            "Models that perform similarly in both regimes are more robust."
        )


def _render_assumption_panel(prices):
    """Show model assumption diagnostics with clear pass/fail."""
    with st.expander("Model Assumption Diagnostics", expanded=False):
        returns = prices.pct_change().dropna()

        col1, col2, col3, col4 = st.columns(4)

        # ADF stationarity test on returns
        with col1:
            st.markdown("**Stationarity (ADF)**")
            st.caption("Required by ARIMA")
            adf = test_stationarity(returns)
            if 'error' not in adf:
                if adf['is_stationary']:
                    st.success(f"PASS (p={adf['p_value']:.4f})")
                else:
                    st.error(f"FAIL (p={adf['p_value']:.4f})")
                st.caption(f"Returns are {'stationary' if adf['is_stationary'] else 'non-stationary'}. "
                           f"{'ARIMA is appropriate.' if adf['is_stationary'] else 'Consider differencing.'}")

        # ARCH effects test
        with col2:
            st.markdown("**ARCH Effects**")
            st.caption("Required by GARCH")
            arch = test_arch_effects(returns)
            if 'error' not in arch:
                if arch['has_arch_effects']:
                    st.success(f"PASS (p={arch['lm_p_value']:.4f})")
                    st.caption("Volatility clustering detected. GARCH is appropriate.")
                else:
                    st.warning(f"WEAK (p={arch['lm_p_value']:.4f})")
                    st.caption("Little volatility clustering. GARCH may not add value.")

        # Normality test
        with col3:
            st.markdown("**Normality (JB)**")
            st.caption("Informational")
            norm = test_normality(returns)
            if 'error' not in norm:
                if norm['is_normal']:
                    st.success(f"PASS (p={norm['p_value']:.4f})")
                else:
                    st.info(f"FAIL (p={norm['p_value']:.4f})")
                st.caption("Most returns are non-normal (fat tails). "
                           "This is typical and expected for financial data.")

        # Autocorrelation test
        with col4:
            st.markdown("**Autocorrelation (LB)**")
            st.caption("Required by ARIMA")
            ac = test_autocorrelation(returns)
            if 'error' not in ac:
                if ac['has_autocorrelation']:
                    st.success("PASS — autocorrelation detected")
                    st.caption("Returns have structure that ARIMA can model.")
                else:
                    st.warning("WEAK — little autocorrelation")
                    st.caption("Returns may be close to random walk. "
                               "ARIMA may not outperform naive.")
