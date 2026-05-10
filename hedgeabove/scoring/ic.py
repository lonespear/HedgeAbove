"""
Information coefficient (IC) — the canonical factor-predictiveness test.

For each rebalance date in the period:
  1. Compute the factor value for each ticker, point-in-time
     (fundamentals via EDGAR, technicals via bars up to rebalance_d)
  2. Compute the forward N-day return for each ticker from rebalance_d
  3. Take Spearman rank correlation across the cross-section

The resulting IC time series tells you whether the factor is predictive
*at all*, and how stably. Standard quant heuristics:
  - Mean IC > 0.03 = signal of predictiveness
  - Mean IC > 0.05 = real factor
  - IR (mean / std) annualized >= 0.5 = actionable
  - % positive periods > 55% = stable signal direction

This is the rigorous quant complement to single-rule backtesting:
backtests measure how a *threshold* on a factor performs; IC measures
whether the *factor itself* has any cross-sectional information.
"""
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from hedgeabove.indicators.technical import flatten_columns
from hedgeabove.scoring.composite import FACTORS


_FUNDAMENTAL_FACTORS = {
    "pe", "pb", "roe", "profit_margin", "debt_to_equity",
    "revenue_growth", "earnings_growth",
}


def _bars_for_universe(symbols, period):
    """Pull daily bars for each symbol. Returns {sym: DataFrame}; missing
    or empty downloads are silently dropped."""
    bars_map = {}
    for s in symbols:
        try:
            df = yf.download(s, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            df = flatten_columns(df)
            if not df.empty and "Close" in df.columns:
                bars_map[s] = df
        except Exception:
            continue
    return bars_map


def factor_ic(factor_name, symbols, period="5y", horizon=21,
              rebalance_freq="ME"):
    """Compute IC time series for a factor across `symbols`.

    Args:
      factor_name: registered factor name (see scoring.composite.FACTORS).
      symbols: list of tickers to use as the cross-section.
      period: yfinance lookback (e.g. "5y", "10y", "max").
      horizon: forward-return horizon in trading days (default 21 ≈ 1 month).
      rebalance_freq: pandas freq alias — "ME" month-end (default),
        "W" weekly, "QE" quarter-end.

    Returns DataFrame with columns: rebalance_date, n_tickers, ic, p_value.
    Periods with fewer than 5 valid tickers are skipped.
    """
    from scipy.stats import spearmanr

    if factor_name not in FACTORS:
        raise ValueError(f"Unknown factor: {factor_name!r}. "
                         f"Available: {list(FACTORS)}")
    factor_fn = FACTORS[factor_name]
    is_fundamental = factor_name in _FUNDAMENTAL_FACTORS

    if is_fundamental:
        from hedgeabove.data.edgar import get_fundamentals_as_of  # noqa

    bars_map = _bars_for_universe(symbols, period)
    if not bars_map:
        return pd.DataFrame()

    # Build the union daily index, then resample to the rebalance frequency.
    all_dates = pd.DatetimeIndex(sorted(
        set().union(*(set(df.index) for df in bars_map.values()))
    ))
    rebal_series = pd.Series(all_dates, index=all_dates).resample(rebalance_freq).last().dropna()
    rebal_dates = list(rebal_series.values)

    rows = []
    for rebal_d in rebal_dates:
        d_ts = pd.Timestamp(rebal_d)
        factor_vals = {}
        fwd_returns = {}

        for sym, df in bars_map.items():
            idx_arr = df.index.get_indexer([d_ts], method="ffill")
            idx = int(idx_arr[0]) if len(idx_arr) else -1
            if idx < 0 or idx >= len(df):
                continue
            entry_price = float(df.iloc[idx]["Close"])

            # Factor value at rebalance date
            if is_fundamental:
                from hedgeabove.data.edgar import get_fundamentals_as_of
                bar_date = df.index[idx].date() if hasattr(df.index[idx], "date") else df.index[idx]
                info = get_fundamentals_as_of(sym, bar_date, current_price=entry_price)
                fv = factor_fn(sym, info, df.iloc[:idx + 1])
            else:
                fv = factor_fn(sym, None, df.iloc[:idx + 1])
            if fv is None or (isinstance(fv, float) and np.isnan(fv)):
                continue

            # Forward return
            fwd_idx = idx + horizon
            if fwd_idx >= len(df):
                continue
            fwd_ret = float(df.iloc[fwd_idx]["Close"]) / entry_price - 1.0

            factor_vals[sym] = fv
            fwd_returns[sym] = fwd_ret

        if len(factor_vals) < 5:
            continue
        common = list(factor_vals.keys())
        f_arr = np.array([factor_vals[s] for s in common], dtype=float)
        r_arr = np.array([fwd_returns[s] for s in common], dtype=float)

        try:
            ic, p = spearmanr(f_arr, r_arr)
        except Exception:
            continue
        if np.isnan(ic):
            continue

        rows.append({
            "rebalance_date": d_ts.date(),
            "n_tickers": len(common),
            "ic": float(ic),
            "p_value": float(p) if not np.isnan(p) else None,
        })

    return pd.DataFrame(rows)


def factor_ic_summary(ic_df, periods_per_year=12):
    """Aggregate IC stats. `periods_per_year` annualizes the IR
    (12 = monthly rebalance, 4 = quarterly, 52 = weekly)."""
    if ic_df is None or ic_df.empty:
        return {}
    ics = ic_df["ic"].dropna()
    if ics.empty:
        return {}
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    n = len(ics)
    return {
        "n_periods": n,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir_annualized": (mean_ic / std_ic * np.sqrt(periods_per_year))
                          if std_ic > 0 else None,
        "t_stat": (mean_ic / (std_ic / np.sqrt(n))) if std_ic > 0 else None,
        "pct_positive": float((ics > 0).mean()),
        "min_ic": float(ics.min()),
        "max_ic": float(ics.max()),
    }
