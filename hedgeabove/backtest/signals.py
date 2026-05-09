"""
Signal backtesting: replay rules over historical bars and measure forward
returns. Used by ``cli analyze`` and the Streamlit Rule Analytics UI.

**Technical rules** are evaluated against indicator-augmented daily bars; the
indicator pipeline is point-in-time by construction.

**Fundamental rules** are evaluated against a point-in-time snapshot built
from SEC EDGAR XBRL via ``hedgeabove.data.edgar.get_fundamentals_as_of``.
The fundamentals dict has the same shape as the live ``get_stock_info`` so
the rule code is unchanged. Look-ahead is defeated because EDGAR facts are
filtered to ``filing_date <= bar_date`` with amendments deduped to the
earliest filing per period.

Caveats for fundamental rules:
- ``analyst_upside_above`` can't be backtested (analyst targets aren't filed
  with the SEC; require Finnhub historical snapshots, future enhancement).
- ``dividend_yield_above`` not yet wired (TTM dividend extraction is a TODO).
- Fundamental conditions persist for ~90 days between filings, so the same
  rule fires every day the condition holds — interpret hit rate as the
  win rate of being long *while* the condition is true, not as the win rate
  of statistically-independent trade events.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from hedgeabove import config
from hedgeabove.indicators.technical import add_indicators, flatten_columns
from hedgeabove.rules import technical as tech_rules
from hedgeabove.rules import fundamental as fund_rules


DEFAULT_HORIZONS = (5, 10, 20)


@dataclass
class FireEvent:
    fire_date: pd.Timestamp
    message: str
    price_at_fire: float
    fwd_returns: dict = field(default_factory=dict)  # {horizon_days: pct_return | None}


def replay_rule(symbol, rule_type, params=None, period="5y",
                horizons: Sequence[int] = DEFAULT_HORIZONS):
    """Replay one rule (technical or fundamental) over historical bars for
    one symbol. Returns a list of FireEvent.

    For fundamental rules, point-in-time fundamentals are pulled from SEC
    EDGAR via ``hedgeabove.data.edgar.get_fundamentals_as_of`` at each bar's
    date — no look-ahead.
    """
    is_technical = rule_type in tech_rules.REGISTRY
    is_fundamental = rule_type in fund_rules.REGISTRY
    if not (is_technical or is_fundamental):
        raise ValueError(f"Unknown rule type: {rule_type!r}")
    params = params or {}

    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    df = flatten_columns(df)
    if df.empty or len(df) < config.MA_SLOW + 5:
        return []
    if is_technical:
        df = add_indicators(df,
                            rsi_period=config.RSI_PERIOD,
                            ma_fast=config.MA_FAST,
                            ma_slow=config.MA_SLOW)
    else:
        from hedgeabove.data.edgar import get_fundamentals_as_of  # lazy

    fires = []
    n = len(df)
    for i in range(1, n):
        latest = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(latest["Close"])

        if is_technical:
            msg = tech_rules.evaluate(rule_type, latest, prev, params)
        else:
            bar_dt = df.index[i]
            bar_date = bar_dt.date() if hasattr(bar_dt, "date") else bar_dt
            stock_info = get_fundamentals_as_of(symbol, bar_date, current_price=price)
            msg = fund_rules.evaluate(rule_type, stock_info, params)

        if msg is None:
            continue
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


def replay_composite(symbol, rules_list, combiner="all", period="5y",
                     horizons: Sequence[int] = DEFAULT_HORIZONS):
    """Replay multiple rules on a symbol with an AND/OR/MAJORITY combiner.

    Args:
      rules_list: list of (rule_type, params_dict). Tech + fundamental can mix.
      combiner: 'all' (AND), 'any' (OR), 'majority' (>50% of rules fire)

    Each bar is evaluated against every rule. If the combiner threshold
    is met, the composite fires once for that bar with a message listing
    which sub-rules contributed. Forward returns are computed as in
    replay_rule.
    """
    if combiner not in ("all", "any", "majority"):
        raise ValueError(f"Unknown combiner: {combiner!r}. Use 'all', 'any', or 'majority'.")
    rules_list = [(rt, p or {}) for rt, p in rules_list]
    if not rules_list:
        return []
    unknown = [rt for rt, _ in rules_list
               if rt not in tech_rules.REGISTRY and rt not in fund_rules.REGISTRY]
    if unknown:
        raise ValueError(f"Unknown rule type(s): {unknown}")

    has_tech = any(rt in tech_rules.REGISTRY for rt, _ in rules_list)
    has_fund = any(rt in fund_rules.REGISTRY for rt, _ in rules_list)

    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    df = flatten_columns(df)
    if df.empty or len(df) < config.MA_SLOW + 5:
        return []
    if has_tech:
        df = add_indicators(df,
                            rsi_period=config.RSI_PERIOD,
                            ma_fast=config.MA_FAST,
                            ma_slow=config.MA_SLOW)
    if has_fund:
        from hedgeabove.data.edgar import get_fundamentals_as_of

    fires = []
    n = len(df)
    n_rules = len(rules_list)
    if combiner == "all":
        min_to_fire = n_rules
    elif combiner == "majority":
        min_to_fire = n_rules // 2 + 1
    else:  # any
        min_to_fire = 1

    for i in range(1, n):
        latest = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(latest["Close"])
        bar_dt = df.index[i]
        bar_date = bar_dt.date() if hasattr(bar_dt, "date") else bar_dt

        stock_info = None  # lazy fetch only if a fundamental rule is in the mix
        fired_rule_names = []
        for rule_type, params in rules_list:
            if rule_type in tech_rules.REGISTRY:
                msg = tech_rules.evaluate(rule_type, latest, prev, params)
            else:
                if stock_info is None:
                    stock_info = get_fundamentals_as_of(symbol, bar_date, current_price=price)
                msg = fund_rules.evaluate(rule_type, stock_info, params)
            if msg is not None:
                fired_rule_names.append(rule_type)

        if len(fired_rule_names) < min_to_fire:
            continue

        composite_msg = f"[{combiner.upper()}: {' + '.join(fired_rule_names)}]"
        fwd = {}
        for h in horizons:
            j = i + h
            fwd[h] = (float(df.iloc[j]["Close"]) / price - 1) if j < n else None
        fires.append(FireEvent(
            fire_date=df.index[i],
            message=composite_msg,
            price_at_fire=price,
            fwd_returns=fwd,
        ))
    return fires


def summarize_composite(symbol, rules_list, combiner="all", period="5y",
                        horizons: Sequence[int] = DEFAULT_HORIZONS):
    """Replay + aggregate stats for a composite of rules. Returns
    (summary_dict, fire_events). The summary's ``rule_type`` is a
    descriptive label like ``[ALL: rsi_oversold,golden_cross]`` for
    display, not a registered rule name."""
    fires = replay_composite(symbol, rules_list, combiner, period, horizons)
    label = f"[{combiner.upper()}: " + ",".join(rt for rt, _ in rules_list) + "]"
    summary = {
        "symbol": symbol,
        "rule_type": label,
        "params": {},
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
