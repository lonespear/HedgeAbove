"""
Signal backtesting: replay technical rules over historical bars and measure
forward returns. Used by ``cli analyze`` and the Streamlit Rule Analytics UI.

Limitation: fundamental rules use today's snapshot from ``ticker.info``,
so backtesting them with current ratios against historical dates would
inject look-ahead bias. Only technical rules (which depend on point-in-time
bars + indicators) are supported here. Fundamental backtesting requires
point-in-time fundamentals — a future enhancement requiring a paid data
source like Sharadar or alpha-vantage premium.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from hedgeabove import config
from hedgeabove.indicators.technical import add_indicators, flatten_columns
from hedgeabove.rules import technical as tech_rules


DEFAULT_HORIZONS = (5, 10, 20)


@dataclass
class FireEvent:
    fire_date: pd.Timestamp
    message: str
    price_at_fire: float
    fwd_returns: dict = field(default_factory=dict)  # {horizon_days: pct_return | None}


def replay_rule(symbol, rule_type, params=None, period="5y",
                horizons: Sequence[int] = DEFAULT_HORIZONS):
    """Replay one technical rule over historical bars for one symbol.

    Returns a list of FireEvent. Each event's `fwd_returns[h]` is the
    percentage change in close price between the fire bar and the bar `h`
    trading days later (None if there aren't enough bars yet, e.g. recent fires).
    """
    if rule_type not in tech_rules.REGISTRY:
        raise ValueError(
            f"replay_rule only supports technical rules; '{rule_type}' is not technical."
        )
    params = params or {}
    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    df = flatten_columns(df)
    if df.empty or len(df) < config.MA_SLOW + 5:
        return []
    df = add_indicators(df,
                        rsi_period=config.RSI_PERIOD,
                        ma_fast=config.MA_FAST,
                        ma_slow=config.MA_SLOW)

    fires = []
    n = len(df)
    for i in range(1, n):
        latest = df.iloc[i]
        prev = df.iloc[i - 1]
        msg = tech_rules.evaluate(rule_type, latest, prev, params)
        if msg is None:
            continue
        price = float(latest["Close"])
        fwd = {}
        for h in horizons:
            j = i + h
            fwd[h] = (float(df.iloc[j]["Close"]) / price - 1) if j < n else None
        fires.append(FireEvent(
            fire_date=df.index[i],
            message=msg,
            price_at_fire=price,
            fwd_returns=fwd,
        ))
    return fires


def summarize_fires(fires, horizons: Sequence[int] = DEFAULT_HORIZONS):
    """Aggregate per-horizon stats over a list of FireEvents."""
    out = {"n_fires": len(fires)}
    for h in horizons:
        rets = [f.fwd_returns.get(h) for f in fires]
        rets = [r for r in rets if r is not None]
        if not rets:
            out[f"n_with_fwd_{h}d"] = 0
            out[f"hit_rate_{h}d"] = None
            out[f"avg_return_{h}d"] = None
            out[f"median_return_{h}d"] = None
            out[f"std_return_{h}d"] = None
            out[f"sharpe_{h}d"] = None
            continue
        arr = np.array(rets, dtype=float)
        n_w = len(arr)
        avg = float(arr.mean())
        std = float(arr.std(ddof=1)) if n_w > 1 else 0.0
        out[f"n_with_fwd_{h}d"] = n_w
        out[f"hit_rate_{h}d"] = float((arr > 0).mean())
        out[f"avg_return_{h}d"] = avg
        out[f"median_return_{h}d"] = float(np.median(arr))
        out[f"std_return_{h}d"] = std
        out[f"sharpe_{h}d"] = (avg / std) if std > 0 else None
    return out


def summarize_rule(symbol, rule_type, params=None, period="5y",
                   horizons: Sequence[int] = DEFAULT_HORIZONS):
    """Convenience: replay + summarize. Returns (summary_dict, fire_events)."""
    fires = replay_rule(symbol, rule_type, params, period, horizons)
    summary = {
        "symbol": symbol,
        "rule_type": rule_type,
        "params": params or {},
        "period": period,
        **summarize_fires(fires, horizons),
    }
    return summary, fires


def fires_to_dataframe(fires, horizons: Sequence[int] = DEFAULT_HORIZONS):
    """Convert a list of FireEvents to a wide DataFrame for display/export."""
    rows = []
    for f in fires:
        row = {
            "date": f.fire_date.date(),
            "price": f.price_at_fire,
            "message": f.message,
        }
        for h in horizons:
            r = f.fwd_returns.get(h)
            row[f"fwd_{h}d"] = r if r is not None else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def cross_section_summary(symbols, rule_type, params=None, period="5y",
                          horizons: Sequence[int] = DEFAULT_HORIZONS,
                          progress=None):
    """Run summarize_rule across many symbols. Returns a ranked DataFrame.

    Useful for asking "which tickers does this rule work best on?". Sorted by
    20-day hit rate descending.
    """
    rows = []
    for i, sym in enumerate(symbols):
        if progress is not None:
            progress(i, len(symbols), sym)
        try:
            summary, _ = summarize_rule(sym, rule_type, params, period, horizons)
        except Exception:
            continue
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_col = f"hit_rate_{horizons[-1]}d"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    return df.reset_index(drop=True)
