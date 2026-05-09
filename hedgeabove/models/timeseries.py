"""
Time series models: ARIMA, GARCH, and combined forecasting.
"""

import numpy as np
import streamlit as st


@st.cache_data(ttl=3600)
def fit_arima_model(data, auto=True, order=(1, 1, 1)):
    """Fit ARIMA model with auto parameter selection."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        if auto:
            try:
                from pmdarima import auto_arima
                model = auto_arima(
                    data, seasonal=False, stepwise=True,
                    suppress_warnings=True, error_action='ignore',
                    max_p=5, max_d=2, max_q=5, trace=False,
                )
                fitted_model = model
                best_order = model.order
                aic = model.aic()
                bic = model.bic()
            except ImportError:
                best_aic = np.inf
                best_order = (1, 1, 1)
                best_model = None

                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                temp_model = ARIMA(data, order=(p, d, q))
                                temp_fit = temp_model.fit()
                                if temp_fit.aic < best_aic:
                                    best_aic = temp_fit.aic
                                    best_order = (p, d, q)
                                    best_model = temp_fit
                            except Exception:
                                continue

                fitted_model = best_model
                aic = best_model.aic
                bic = best_model.bic
        else:
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            best_order = order
            aic = fitted_model.aic
            bic = fitted_model.bic

        return {
            'model': fitted_model,
            'order': best_order,
            'aic': aic,
            'bic': bic,
            'residuals': fitted_model.resid,
        }
    except Exception as e:
        st.error(f"ARIMA fitting error: {str(e)}")
        return None


def forecast_arima(fitted_model, steps=30, alpha=0.05):
    """Generate ARIMA forecasts with confidence intervals."""
    try:
        if hasattr(fitted_model, 'predict') and hasattr(fitted_model, 'n_periods'):
            forecast = fitted_model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
            if isinstance(forecast, tuple):
                predictions = forecast[0]
                conf_int = forecast[1]
            else:
                predictions = forecast
                conf_int = None
        else:
            forecast_result = fitted_model.get_forecast(steps=steps)
            predictions = forecast_result.predicted_mean
            conf_int_df = forecast_result.conf_int(alpha=alpha)
            conf_int = conf_int_df.values if conf_int_df is not None else None

        return {
            'forecast': predictions,
            'conf_int': conf_int,
            'lower': conf_int[:, 0] if conf_int is not None else None,
            'upper': conf_int[:, 1] if conf_int is not None else None,
        }
    except Exception as e:
        st.error(f"ARIMA forecast error: {str(e)}")
        return None


def arima_diagnostics(residuals):
    """Generate ARIMA diagnostic statistics."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy.stats import jarque_bera

        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        jb_stat, jb_pvalue = jarque_bera(residuals)
        residual_mean = residuals.mean()
        residual_std = residuals.std()

        return {
            'ljung_box': lb_test,
            'jb_statistic': jb_stat,
            'jb_pvalue': jb_pvalue,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'is_normal': jb_pvalue > 0.05,
        }
    except Exception as e:
        return {'error': str(e)}


@st.cache_data(ttl=3600)
def fit_garch_model(returns, p=1, q=1, model_type='GARCH'):
    """Fit GARCH model to returns data."""
    try:
        from arch import arch_model

        returns_pct = returns * 100
        model = arch_model(returns_pct, vol=model_type, p=p, q=q)
        fitted_model = model.fit(disp='off', show_warning=False)

        return {
            'model': fitted_model,
            'params': fitted_model.params,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'conditional_volatility': fitted_model.conditional_volatility,
            'residuals': fitted_model.resid,
        }
    except Exception as e:
        st.error(f"GARCH fitting error: {str(e)}")
        return None


def forecast_volatility(fitted_garch, horizon=30):
    """Forecast volatility using fitted GARCH model."""
    try:
        forecast = fitted_garch.forecast(horizon=horizon)
        variance_forecast = forecast.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast)

        return {
            'volatility': volatility_forecast,
            'variance': variance_forecast,
            'horizon': horizon,
        }
    except Exception as e:
        st.error(f"Volatility forecast error: {str(e)}")
        return None


def garch_var(returns, fitted_garch, confidence=0.95):
    """Calculate VaR using GARCH volatility forecast."""
    try:
        from scipy.stats import norm

        forecast = fitted_garch.forecast(horizon=1)
        volatility = np.sqrt(forecast.variance.values[-1, 0]) / 100

        z_score = norm.ppf(1 - confidence)
        return z_score * volatility
    except Exception:
        return None


def extract_volatility_regimes(conditional_vol, threshold_percentile=75):
    """Identify high and low volatility regimes."""
    try:
        threshold = np.percentile(conditional_vol, threshold_percentile)

        high_vol_periods = conditional_vol > threshold
        low_vol_periods = conditional_vol <= np.percentile(conditional_vol, 25)

        return {
            'threshold': threshold,
            'high_vol_periods': high_vol_periods,
            'low_vol_periods': low_vol_periods,
            'high_vol_mean': conditional_vol[high_vol_periods].mean(),
            'low_vol_mean': conditional_vol[low_vol_periods].mean(),
        }
    except Exception:
        return None
