"""
Tearsheet metrics derived from a daily equity curve.

A tearsheet is the standard deliverable a quant produces alongside any
backtest: aggregate stats are useful, but stakeholders also want to see
how the strategy behaves *over time* — which years it lost money, how
long the drawdowns took to recover, whether the Sharpe is stable or
collapsing, what its market beta is.

Everything here consumes a daily-indexed pd.Series of NAV (the
``equity_curve`` returned by simulate_basket) and a matching
benchmark NAV when comparing.
"""
import numpy as np
import pandas as pd


def daily_returns(equity_curve):
    """Daily simple returns. Drops NaN."""
    if equity_curve is None or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    return equity_curve.pct_change().dropna()


def monthly_returns(equity_curve):
    """Compound to monthly returns. Series indexed by month-end."""
    if equity_curve is None or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    monthly_nav = equity_curve.resample("ME").last()
    return monthly_nav.pct_change().dropna()


def calendar_year_returns(equity_curve):
    """Compound to calendar-year returns. Series indexed by year-end.
    Includes the partial first/last years if applicable."""
    if equity_curve is None or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    yearly_nav = equity_curve.resample("YE").last()
    # Prepend the starting NAV so the first computed return spans
    # start-of-period to first year-end.
    start_val = float(equity_curve.iloc[0])
    yearly_nav = pd.concat([
        pd.Series([start_val], index=[equity_curve.index[0]]),
        yearly_nav,
    ])
    return yearly_nav.pct_change().dropna()


def drawdown_series(equity_curve):
    """Daily underwater percentage: 0 at peaks, negative below them."""
    if equity_curve is None or equity_curve.empty:
        return pd.Series(dtype=float)
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def drawdown_summary(equity_curve):
    """Max DD, the date it occurred, and the longest consecutive underwater
    stretch (in trading days, the granularity of the equity curve).
    """
    dd = drawdown_series(equity_curve)
    if dd.empty:
        return {}
    max_dd = float(dd.min())
    max_dd_date = dd.idxmin()
    underwater = dd < 0
    if not underwater.any():
        longest_run = 0
    else:
        # Run-length encode the underwater Boolean series.
        groups = (underwater != underwater.shift()).cumsum()
        run_lengths = underwater.groupby(groups).sum()
        longest_run = int(run_lengths.max())
    return {
        "max_drawdown": max_dd,
        "max_drawdown_date": max_dd_date,
        "longest_drawdown_days": longest_run,
    }


def rolling_sharpe(equity_curve, window=252, ann_factor=252):
    """Rolling annualized Sharpe of daily returns over `window` trading days.
    Returns Series of the same length as `equity_curve` (with NaN for the
    warm-up period)."""
    rets = daily_returns(equity_curve)
    if len(rets) < window:
        return pd.Series(dtype=float)
    rolling_mean = rets.rolling(window).mean()
    rolling_std = rets.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(ann_factor)


def beta_to_benchmark(strategy_curve, benchmark_curve):
    """OLS slope of strategy daily returns regressed on benchmark daily
    returns. Returns float or None if the data is too thin."""
    s_rets = daily_returns(strategy_curve)
    b_rets = daily_returns(benchmark_curve)
    aligned = pd.DataFrame({"s": s_rets, "b": b_rets}).dropna()
    if len(aligned) < 30:
        return None
    var_b = float(aligned["b"].var())
    if var_b <= 0:
        return None
    cov = float(aligned["s"].cov(aligned["b"]))
    return cov / var_b


def tearsheet(equity_curve, benchmark_curve=None):
    """Bundle of all tearsheet primitives — monthly + yearly returns,
    drawdown analysis, rolling Sharpe, optional beta. Used by the CLI
    `--tearsheet` flag and the Streamlit Strategy Lab tearsheet view."""
    out = {
        "monthly_returns": monthly_returns(equity_curve),
        "calendar_year_returns": calendar_year_returns(equity_curve),
        "drawdown_series": drawdown_series(equity_curve),
        **drawdown_summary(equity_curve),
        "rolling_sharpe_252d": rolling_sharpe(equity_curve, 252),
    }
    if benchmark_curve is not None and not benchmark_curve.empty:
        out["beta"] = beta_to_benchmark(equity_curve, benchmark_curve)
    return out
