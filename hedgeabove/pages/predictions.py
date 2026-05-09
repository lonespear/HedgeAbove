"""AI Predictions page — ARIMA, GARCH, and combined forecasting."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from hedgeabove.data.market import get_historical_data
from hedgeabove.models.timeseries import (
    fit_arima_model, forecast_arima, arima_diagnostics,
    fit_garch_model, forecast_volatility,
)
from hedgeabove.models.backtest import (
    test_stationarity, test_arch_effects, test_normality,
)


def render():
    st.header("Time Series Forecasting & Volatility Prediction")

    tab1, tab2, tab3 = st.tabs(["ARIMA Price Forecasts", "GARCH Volatility", "Combined Models"])

    with tab1:
        _render_arima_tab()
    with tab2:
        _render_garch_tab()
    with tab3:
        _render_combined_tab()


def _render_arima_tab():
    st.subheader("ARIMA Price & Returns Forecasting")

    col1, col2, col3 = st.columns(3)
    with col1:
        arima_ticker = st.text_input("Ticker Symbol", value="AAPL", key='arima_ticker')
    with col2:
        forecast_days = st.slider("Forecast Horizon (days)", 5, 90, 30, key='arima_days')
    with col3:
        forecast_type = st.selectbox("Forecast Type", ["Price", "Returns"], key='arima_type')

    auto_arima_flag = st.checkbox("Auto ARIMA (find best p,d,q)", value=True, key='auto_arima')

    if not auto_arima_flag:
        col1, col2, col3 = st.columns(3)
        with col1:
            p_order = st.number_input("p (AR order)", 0, 5, 1, key='arima_p')
        with col2:
            d_order = st.number_input("d (differencing)", 0, 2, 1, key='arima_d')
        with col3:
            q_order = st.number_input("q (MA order)", 0, 5, 1, key='arima_q')
    else:
        p_order, d_order, q_order = 1, 1, 1

    if not st.button("Generate ARIMA Forecast", type="primary", key='run_arima'):
        return

    with st.spinner(f"Fetching data and fitting ARIMA model for {arima_ticker}..."):
        hist_data = get_historical_data([arima_ticker], period='2y')

    if hist_data.empty:
        st.error(f"Could not fetch data for {arima_ticker}")
        return

    prices = hist_data[arima_ticker].dropna()
    returns = prices.pct_change().dropna()
    data_to_model = prices if forecast_type == "Price" else returns

    # Assumption diagnostics
    with st.expander("ARIMA Assumption Check", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Stationarity (ADF test on returns)**")
            adf = test_stationarity(returns)
            if 'error' not in adf:
                if adf['is_stationary']:
                    st.success(f"PASS — p-value: {adf['p_value']:.4f}")
                    st.caption("Returns are stationary. ARIMA assumptions are met.")
                else:
                    st.error(f"FAIL — p-value: {adf['p_value']:.4f}")
                    st.caption("Returns are non-stationary. Consider higher differencing (d).")
        with col_b:
            st.markdown("**Normality (Jarque-Bera on returns)**")
            norm = test_normality(returns)
            if 'error' not in norm:
                if norm['is_normal']:
                    st.success(f"PASS — p-value: {norm['p_value']:.4f}")
                else:
                    st.info(f"Non-normal — p-value: {norm['p_value']:.4f}")
                    st.caption("Fat tails detected. Confidence intervals may be too narrow.")

    arima_result = fit_arima_model(data_to_model, auto=auto_arima_flag,
                                    order=(p_order, d_order, q_order))
    if not arima_result:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Order (p,d,q)", f"{arima_result['order']}")
    col2.metric("AIC", f"{arima_result['aic']:.2f}")
    col3.metric("BIC", f"{arima_result['bic']:.2f}")

    forecast_result = forecast_arima(arima_result['model'], steps=forecast_days, alpha=0.05)
    if not forecast_result:
        return

    last_date = data_to_model.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='D')[1:]

    if forecast_type == "Returns":
        forecast_values = forecast_result['forecast'] * 100
        lower_bound = forecast_result['lower'] * 100 if forecast_result['lower'] is not None else None
        upper_bound = forecast_result['upper'] * 100 if forecast_result['upper'] is not None else None
        y_label = "Returns (%)"
    else:
        forecast_values = forecast_result['forecast']
        lower_bound = forecast_result['lower']
        upper_bound = forecast_result['upper']
        y_label = "Price ($)"

    fig = go.Figure()
    historical_plot = data_to_model.tail(90)
    if forecast_type == "Returns":
        historical_plot = historical_plot * 100

    fig.add_trace(go.Scatter(x=historical_plot.index, y=historical_plot.values,
                             mode='lines', name='Historical', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values,
                             mode='lines', name='Forecast', line=dict(color='red', width=2, dash='dash')))

    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,0,0,0)'),
            name='95% Confidence Interval',
        ))

    fig.update_layout(title=f"{arima_ticker} {forecast_type} Forecast (ARIMA{arima_result['order']})",
                      xaxis_title="Date", yaxis_title=y_label, hovermode='x unified', height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model Diagnostics"):
        diagnostics = arima_diagnostics(arima_result['residuals'])
        if 'error' not in diagnostics:
            col1, col2 = st.columns(2)
            col1.metric("Residual Mean", f"{diagnostics['residual_mean']:.6f}")
            col1.metric("Residual Std", f"{diagnostics['residual_std']:.6f}")
            col2.metric("Jarque-Bera Statistic", f"{diagnostics['jb_statistic']:.2f}")
            col2.metric("JB p-value", f"{diagnostics['jb_pvalue']:.4f}")

            if diagnostics['is_normal']:
                st.success("Residuals appear normally distributed (JB test)")
            else:
                st.warning("Residuals may not be normally distributed")

            st.write("**Ljung-Box Test for Autocorrelation:**")
            st.dataframe(diagnostics['ljung_box'], use_container_width=True)
        else:
            st.error(f"Diagnostic error: {diagnostics['error']}")


def _render_garch_tab():
    st.subheader("GARCH Volatility Modeling & Forecasting")

    col1, col2, col3 = st.columns(3)
    with col1:
        garch_ticker = st.text_input("Ticker Symbol", value="AAPL", key='garch_ticker')
    with col2:
        garch_horizon = st.slider("Forecast Horizon (days)", 5, 90, 30, key='garch_horizon')
    with col3:
        model_type = st.selectbox("Model Type", ["GARCH", "EGARCH", "GJR-GARCH"], key='garch_model_type')

    col1, col2 = st.columns(2)
    with col1:
        garch_p_param = st.selectbox("GARCH p (lag)", [1, 2, 3], index=0, key='garch_p_param')
    with col2:
        garch_q_param = st.selectbox("GARCH q (lag)", [1, 2, 3], index=0, key='garch_q_param')

    if not st.button("Run GARCH Forecast", type="primary", key='run_garch'):
        return

    with st.spinner(f"Fitting GARCH model for {garch_ticker}..."):
        hist_data = get_historical_data([garch_ticker], period='2y')

    if hist_data.empty:
        st.error(f"Could not fetch data for {garch_ticker}")
        return

    prices = hist_data[garch_ticker].dropna()
    returns = prices.pct_change().dropna()

    # Assumption diagnostics
    with st.expander("GARCH Assumption Check", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**ARCH Effects (Engle's LM test)**")
            from hedgeabove.models.backtest import test_arch_effects
            arch = test_arch_effects(returns)
            if 'error' not in arch:
                if arch['has_arch_effects']:
                    st.success(f"PASS — p-value: {arch['lm_p_value']:.4f}")
                    st.caption("Volatility clustering confirmed. GARCH will capture meaningful dynamics.")
                else:
                    st.warning(f"WEAK — p-value: {arch['lm_p_value']:.4f}")
                    st.caption("Little volatility clustering. GARCH may not improve over constant volatility.")
        with col_b:
            st.markdown("**Normality of Returns**")
            norm = test_normality(returns)
            if 'error' not in norm:
                if norm['is_normal']:
                    st.success(f"Normal — p-value: {norm['p_value']:.4f}")
                else:
                    st.info(f"Non-normal — p-value: {norm['p_value']:.4f}")
                    st.caption("Fat tails present. Consider Student-t error distribution (EGARCH/GJR).")

    garch_result = fit_garch_model(returns, p=garch_p_param, q=garch_q_param, model_type=model_type)
    if not garch_result:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", f"{model_type}({garch_p_param},{garch_q_param})")
    col2.metric("AIC", f"{garch_result['aic']:.2f}")
    col3.metric("BIC", f"{garch_result['bic']:.2f}")

    vol_forecast = forecast_volatility(garch_result['model'], horizon=garch_horizon)
    if not vol_forecast:
        return

    fig = go.Figure()
    hist_vol = garch_result['conditional_volatility'].tail(180) / 100 * np.sqrt(252) * 100
    fig.add_trace(go.Scatter(
        x=returns.index[-len(hist_vol):], y=hist_vol.values,
        mode='lines', name='Historical Volatility', line=dict(color='orange', width=2),
    ))
    forecast_dates = pd.date_range(start=returns.index[-1], periods=garch_horizon + 1, freq='D')[1:]
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=vol_forecast['volatility'] * np.sqrt(252) * 100,
        mode='lines', name='Volatility Forecast', line=dict(color='red', width=2, dash='dash'),
    ))
    fig.update_layout(title=f"{garch_ticker} Volatility Forecast ({model_type}({garch_p_param},{garch_q_param}))",
                      xaxis_title="Date", yaxis_title="Annualized Volatility (%)",
                      hovermode='x unified', height=500)
    st.plotly_chart(fig, use_container_width=True)

    current_vol = garch_result['conditional_volatility'].iloc[-1] / 100 * np.sqrt(252) * 100
    forecast_vol_avg = vol_forecast['volatility'].mean() * np.sqrt(252) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Volatility", f"{current_vol:.2f}%")
    col2.metric("Avg Forecast Volatility", f"{forecast_vol_avg:.2f}%")
    col3.metric("Expected Change", f"{forecast_vol_avg - current_vol:+.2f}%")


def _render_combined_tab():
    st.subheader("Combined ARIMA-GARCH Forecasting")

    st.info(
        "**Combined Model Approach:**\n"
        "1. **ARIMA** models the conditional mean (price trend)\n"
        "2. **GARCH** models the conditional variance (volatility clustering)\n"
        "3. Together they provide complete price distribution forecasts"
    )

    col1, col2 = st.columns(2)
    with col1:
        combined_ticker = st.text_input("Ticker Symbol", value="AAPL", key='combined_ticker')
    with col2:
        combined_horizon = st.slider("Forecast Horizon (days)", 5, 60, 30, key='combined_horizon')

    if not st.button("Run Combined Forecast", type="primary", key='run_combined'):
        return

    with st.spinner(f"Fitting combined ARIMA-GARCH model for {combined_ticker}..."):
        hist_data = get_historical_data([combined_ticker], period='2y')

    if hist_data.empty:
        st.error(f"Could not fetch data for {combined_ticker}")
        return

    prices = hist_data[combined_ticker].dropna()
    returns = prices.pct_change().dropna()

    arima_result = fit_arima_model(returns, auto=True)
    garch_result = fit_garch_model(returns, p=1, q=1)

    if not arima_result or not garch_result:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.write("**ARIMA Component:**")
        st.metric("Order", f"{arima_result['order']}")
        st.metric("AIC", f"{arima_result['aic']:.2f}")
    with col2:
        st.write("**GARCH Component:**")
        st.metric("Order", "(1,1)")
        st.metric("AIC", f"{garch_result['aic']:.2f}")

    arima_forecast = forecast_arima(arima_result['model'], steps=combined_horizon)
    garch_forecast = forecast_volatility(garch_result['model'], horizon=combined_horizon)

    if not arima_forecast or not garch_forecast:
        return

    n_simulations = 1000
    forecast_dates = pd.date_range(start=returns.index[-1],
                                    periods=combined_horizon + 1, freq='D')[1:]

    simulated_prices = np.zeros((n_simulations, combined_horizon))
    last_price = prices.iloc[-1]

    for sim in range(n_simulations):
        price_path = [last_price]
        for i in range(combined_horizon):
            mean_return = arima_forecast['forecast'][i]
            vol = garch_forecast['volatility'][i] / 100
            simulated_return = np.random.normal(mean_return, vol)
            next_price = price_path[-1] * (1 + simulated_return)
            price_path.append(next_price)
            simulated_prices[sim, i] = next_price

    median_forecast = np.median(simulated_prices, axis=0)
    lower_5 = np.percentile(simulated_prices, 5, axis=0)
    upper_95 = np.percentile(simulated_prices, 95, axis=0)
    lower_25 = np.percentile(simulated_prices, 25, axis=0)
    upper_75 = np.percentile(simulated_prices, 75, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.tail(90).index, y=prices.tail(90).values,
                             mode='lines', name='Historical', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=forecast_dates, y=median_forecast,
                             mode='lines', name='Median Forecast', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(
        x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
        y=upper_95.tolist() + lower_5.tolist()[::-1],
        fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,0,0,0)'),
        name='90% Confidence',
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
        y=upper_75.tolist() + lower_25.tolist()[::-1],
        fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,0,0,0)'),
        name='50% Confidence',
    ))
    fig.update_layout(title=f"{combined_ticker} Combined ARIMA-GARCH Price Forecast",
                      xaxis_title="Date", yaxis_title="Price ($)", hovermode='x unified', height=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${last_price:.2f}")
    col2.metric(f"{combined_horizon}-Day Median", f"${median_forecast[-1]:.2f}")
    expected_return = (median_forecast[-1] - last_price) / last_price * 100
    col3.metric("Expected Return", f"{expected_return:+.2f}%")
    col4.metric("90% Range", f"${lower_5[-1]:.2f} - ${upper_95[-1]:.2f}")
