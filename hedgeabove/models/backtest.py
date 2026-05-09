"""
Walk-forward backtesting engine for time series models.

Supports: ARIMA, GARCH (volatility-scaled), Naive (last value), Mean Reversion,
           and combined ARIMA-GARCH.

Usage:
    results = run_backtest(prices, models=['arima', 'naive', 'mean_reversion'],
                           train_pct=0.7, horizon=5, step=5)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ── Metric helpers ──────────────────────────────────────────────

def _rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def _mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def _mape(actual, predicted):
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def _directional_accuracy(actual, predicted):
    """Percentage of times the model correctly predicts up/down direction."""
    if len(actual) < 2:
        return np.nan
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    n = len(actual_dir)
    return np.sum(actual_dir == pred_dir) / n * 100 if n > 0 else np.nan


def _compute_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return {
        'RMSE': _rmse(actual, predicted),
        'MAE': _mae(actual, predicted),
        'MAPE (%)': _mape(actual, predicted),
        'Directional Accuracy (%)': _directional_accuracy(actual, predicted),
    }


# ── Individual model forecasters ────────────────────────────────

def _forecast_naive(train_prices, horizon):
    """Naive: predict last known price for all future steps."""
    return np.full(horizon, train_prices.iloc[-1])


def _forecast_drift(train_prices, horizon):
    """Random walk with drift: last price + average daily change * steps."""
    n = len(train_prices)
    drift = (train_prices.iloc[-1] - train_prices.iloc[0]) / (n - 1)
    last = train_prices.iloc[-1]
    return np.array([last + drift * (i + 1) for i in range(horizon)])


def _forecast_mean_reversion(train_prices, horizon, halflife=21):
    """
    Mean-reversion: price reverts toward rolling mean with exponential decay.
    halflife controls speed of reversion (default 21 trading days ~ 1 month).
    """
    mean_price = train_prices.rolling(window=min(63, len(train_prices))).mean().iloc[-1]
    last = train_prices.iloc[-1]
    decay = np.log(2) / halflife
    forecasts = []
    for i in range(1, horizon + 1):
        forecasts.append(mean_price + (last - mean_price) * np.exp(-decay * i))
    return np.array(forecasts)


def _forecast_arima(train_prices, horizon):
    """ARIMA with auto order selection on price levels."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        best_aic = np.inf
        best_model = None
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(train_prices, order=(p, d, q))
                        fit = model.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_model = fit
                    except Exception:
                        continue

        if best_model is None:
            return None

        forecast = best_model.get_forecast(steps=horizon)
        return forecast.predicted_mean.values
    except Exception:
        return None


def _forecast_garch_scaled(train_prices, horizon):
    """
    GARCH-scaled forecast: drift from returns mean, scaled by GARCH volatility.
    This gives a volatility-aware point forecast rather than pure price prediction.
    """
    try:
        from arch import arch_model

        returns = train_prices.pct_change().dropna()
        returns_pct = returns * 100

        model = arch_model(returns_pct, vol='GARCH', p=1, q=1)
        fit = model.fit(disp='off', show_warning=False)

        # Forecast variance
        fcast = fit.forecast(horizon=horizon)
        vol_forecast = np.sqrt(fcast.variance.values[-1, :]) / 100  # back to decimal

        # Use mean return as drift
        mean_ret = returns.mean()
        last_price = train_prices.iloc[-1]

        prices = []
        p = last_price
        for i in range(horizon):
            p = p * (1 + mean_ret)  # drift
            prices.append(p)
        return np.array(prices)
    except Exception:
        return None


def _forecast_combined(train_prices, horizon):
    """Combined ARIMA (mean) + GARCH (variance) via Monte Carlo median."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from arch import arch_model

        returns = train_prices.pct_change().dropna()

        # ARIMA on returns
        best_aic = np.inf
        best_arima = None
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        m = ARIMA(returns, order=(p, d, q))
                        f = m.fit()
                        if f.aic < best_aic:
                            best_aic = f.aic
                            best_arima = f
                    except Exception:
                        continue

        if best_arima is None:
            return None

        arima_fcast = best_arima.get_forecast(steps=horizon).predicted_mean.values

        # GARCH on returns
        returns_pct = returns * 100
        garch_model = arch_model(returns_pct, vol='GARCH', p=1, q=1)
        garch_fit = garch_model.fit(disp='off', show_warning=False)
        garch_fcast = garch_fit.forecast(horizon=horizon)
        vol_fcast = np.sqrt(garch_fcast.variance.values[-1, :]) / 100

        # Monte Carlo simulation (200 paths for speed in backtest)
        n_sims = 200
        last_price = train_prices.iloc[-1]
        final_paths = np.zeros((n_sims, horizon))

        for s in range(n_sims):
            price = last_price
            for i in range(horizon):
                ret = np.random.normal(arima_fcast[i], vol_fcast[i])
                price = price * (1 + ret)
                final_paths[s, i] = price

        return np.median(final_paths, axis=0)
    except Exception:
        return None


# ── Assumption tests ────────────────────────────────────────────

def test_stationarity(series):
    """
    Augmented Dickey-Fuller test for stationarity.
    Returns dict with test statistic, p-value, and pass/fail.
    Used to validate ARIMA assumptions.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05,
            'n_lags': result[2],
            'n_obs': result[3],
        }
    except Exception as e:
        return {'error': str(e)}


def test_arch_effects(returns):
    """
    Engle's ARCH test — checks if GARCH modeling is warranted.
    Null hypothesis: no ARCH effects (homoscedastic).
    Returns dict with LM statistic, p-value, and whether GARCH is appropriate.
    """
    try:
        from statsmodels.stats.diagnostic import het_arch
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns.dropna(), nlags=5)
        return {
            'lm_statistic': lm_stat,
            'lm_p_value': lm_pvalue,
            'f_statistic': f_stat,
            'f_p_value': f_pvalue,
            'has_arch_effects': lm_pvalue < 0.05,
        }
    except Exception as e:
        return {'error': str(e)}


def test_normality(series):
    """Jarque-Bera normality test."""
    try:
        from scipy.stats import jarque_bera
        stat, p = jarque_bera(series.dropna())
        return {
            'jb_statistic': stat,
            'p_value': p,
            'is_normal': p > 0.05,
        }
    except Exception as e:
        return {'error': str(e)}


def test_autocorrelation(returns, lags=10):
    """Ljung-Box test for autocorrelation in returns."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(returns.dropna(), lags=lags, return_df=True)
        # Check if any lag is significant
        any_significant = (lb['lb_pvalue'] < 0.05).any()
        return {
            'ljung_box_df': lb,
            'has_autocorrelation': any_significant,
        }
    except Exception as e:
        return {'error': str(e)}


# ── Main backtest runner ────────────────────────────────────────

MODEL_REGISTRY = {
    'naive':           ('Naive (Last Price)',     _forecast_naive),
    'drift':           ('Random Walk + Drift',    _forecast_drift),
    'mean_reversion':  ('Mean Reversion',         _forecast_mean_reversion),
    'arima':           ('ARIMA (Auto)',            _forecast_arima),
    'garch':           ('GARCH-Scaled Drift',     _forecast_garch_scaled),
    'combined':        ('ARIMA + GARCH Combined', _forecast_combined),
}


def run_backtest(prices, models=None, train_pct=0.70, horizon=5, step=5):
    """
    Walk-forward backtest.

    Parameters
    ----------
    prices : pd.Series
        Historical price series (daily close).
    models : list[str] or None
        Model keys from MODEL_REGISTRY. None = all models.
    train_pct : float
        Fraction of data for initial training window (0.5–0.9).
    horizon : int
        Number of days to forecast each step.
    step : int
        How many days to advance the window each iteration.

    Returns
    -------
    dict with keys:
        'metrics'     : pd.DataFrame — model × metric summary table
        'forecasts'   : dict[model_name] → list of (dates, actual, predicted) tuples
        'prices'      : the original price series
        'train_end'   : index where initial training window ends
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())

    n = len(prices)
    train_end = int(n * train_pct)

    # Collect all forecasts per model
    all_results = {name: {'actual': [], 'predicted': [], 'dates': []} for name in models}

    # Walk-forward loop
    current_start = 0
    current_end = train_end

    while current_end + horizon <= n:
        train = prices.iloc[current_start:current_end]
        test = prices.iloc[current_end:current_end + horizon]
        test_dates = test.index

        for model_key in models:
            label, forecast_fn = MODEL_REGISTRY[model_key]
            try:
                pred = forecast_fn(train, horizon)
                if pred is not None and len(pred) == horizon:
                    all_results[model_key]['actual'].extend(test.values)
                    all_results[model_key]['predicted'].extend(pred)
                    all_results[model_key]['dates'].extend(test_dates)
            except Exception:
                pass

        current_end += step

    # Compute summary metrics
    metrics_rows = []
    for model_key in models:
        label = MODEL_REGISTRY[model_key][0]
        res = all_results[model_key]
        if len(res['actual']) == 0:
            metrics_rows.append({
                'Model': label,
                'RMSE': np.nan, 'MAE': np.nan,
                'MAPE (%)': np.nan, 'Directional Accuracy (%)': np.nan,
                'N Predictions': 0,
            })
            continue

        m = _compute_metrics(res['actual'], res['predicted'])
        m['Model'] = label
        m['N Predictions'] = len(res['actual'])
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).set_index('Model')

    # Build forecast dict with arrays for plotting
    forecasts = {}
    for model_key in models:
        label = MODEL_REGISTRY[model_key][0]
        res = all_results[model_key]
        if res['dates']:
            forecasts[label] = {
                'dates': pd.DatetimeIndex(res['dates']),
                'actual': np.array(res['actual']),
                'predicted': np.array(res['predicted']),
            }

    return {
        'metrics': metrics_df,
        'forecasts': forecasts,
        'prices': prices,
        'train_end_idx': train_end,
    }
