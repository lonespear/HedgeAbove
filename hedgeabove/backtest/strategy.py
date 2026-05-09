"""
Strategy backtesting: walk fires across a basket and produce an equity curve.

The model is intentionally simple — single-position long-only:
  1. Replay the rule on each symbol in the basket.
  2. Sort all fires chronologically.
  3. Walk through them. If flat, take the next fire as entry. Hold for
     ``hold_days`` trading days. Exit at the close of bar ``entry_idx +
     hold_days``. While holding, ignore any other fires.
  4. After exit, look for the next fire and repeat.

This is the "naive sequential trader" baseline. It maps each fire to a
real trade with realized P&L on actual historical bars, so total return,
Sharpe, and max drawdown are all computable. Multi-position / overlap
strategies are a v2 (max_concurrent > 1 will need pro-rata sizing and
running-position bookkeeping).

Caveats:
- Costs and slippage = 0. Real world is harder.
- Cash earns 0%. Easy to extend to a risk-free rate.
- Fundamental rules persist for ~90 days, so single-position will catch
  the first crossing into "in-state" and exit hold_days later — the
  remaining ~70 days of "in-state" are skipped (correct under the
  single-position model; not the same as the per-day hit-rate stats).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from hedgeabove.indicators.technical import flatten_columns


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    return_pct: float
    holding_days: int
    rule_message: str


def simulate_basket(symbols, rule_type, params=None, period="5y", hold_days=20,
                    starting_nav=1.0, benchmark="SPY"):
    """Run the single-position basket simulator.

    Args:
      symbols: tickers to consider for entries
      rule_type: name of a registered rule (technical or fundamental)
      params: per-rule param overrides (dict)
      period: yfinance period string ("1y", "5y", "max", ...)
      hold_days: number of trading days to hold each position
      starting_nav: initial portfolio value (default 1.0)
      benchmark: ticker for buy-and-hold comparison (default SPY).
        Pass ``None`` to skip benchmark calculation.

    Returns dict with:
      summary: dict of n_trades, win_rate, avg/median trade return,
               total_return, ann_return, sharpe_ann, max_drawdown,
               trades_per_year, span_years, plus (when benchmark set)
               benchmark, benchmark_total_return, benchmark_ann_return,
               benchmark_max_drawdown, alpha_total, alpha_ann.
      trades: list[Trade]
      equity_curve: pd.Series of daily NAV (mark-to-market in trade,
                    flat in cash) over the strategy span.
      benchmark_curve: pd.Series of daily benchmark NAV over the same
                       span (same starting_nav). Empty if benchmark=None.
    """
    from hedgeabove.backtest.signals import replay_rule

    params = params or {}

    # Bars per symbol — needed to find exit prices `hold_days` after entry.
    bars_map = {}
    for s in symbols:
        df = yf.download(s, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        df = flatten_columns(df)
        if not df.empty:
            bars_map[s] = df

    # Replay rule on each symbol → chronological fire list.
    all_fires = []
    for s, df in bars_map.items():
        try:
            fires = replay_rule(s, rule_type, params, period)
        except Exception:
            fires = []
        for f in fires:
            all_fires.append((f.fire_date, s, f.price_at_fire, f.message))
    all_fires.sort(key=lambda x: x[0])

    empty = {"summary": {"n_trades": 0}, "trades": [],
             "equity_curve": pd.Series(dtype=float),
             "benchmark_curve": pd.Series(dtype=float)}
    if not all_fires:
        return empty

    # Sequential single-position walk.
    trades: list = []
    in_position_until = None
    for fire_date, sym, entry_price, msg in all_fires:
        if in_position_until is not None and fire_date < in_position_until:
            continue
        df = bars_map[sym]
        try:
            idx = df.index.get_loc(fire_date)
        except (KeyError, TypeError):
            idx_arr = df.index.get_indexer([fire_date], method="nearest")
            idx = int(idx_arr[0]) if len(idx_arr) else -1
        if idx < 0:
            continue
        exit_idx = idx + hold_days
        if exit_idx >= len(df):
            continue  # not enough future bars to close
        exit_date = df.index[exit_idx]
        exit_price = float(df.iloc[exit_idx]["Close"])
        trades.append(Trade(
            symbol=sym,
            entry_date=fire_date,
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            return_pct=exit_price / entry_price - 1.0,
            holding_days=hold_days,
            rule_message=msg,
        ))
        in_position_until = exit_date

    if not trades:
        return empty

    # Build a daily-aligned strategy NAV by walking the trades in order.
    # In a position: mark-to-market on close. In cash: flat at last realized NAV.
    span_start = trades[0].entry_date
    span_end = trades[-1].exit_date
    # Use the union of all bar dates within the span as our trading-day index.
    all_dates = pd.DatetimeIndex(sorted(
        set().union(*(set(df.index) for df in bars_map.values()))
    ))
    daily_idx = all_dates[(all_dates >= span_start) & (all_dates <= span_end)]

    nav_arr = []
    realized_nav = starting_nav
    ti = 0
    for d in daily_idx:
        # Advance past trades that have already closed by date d
        while ti < len(trades) and d > trades[ti].exit_date:
            realized_nav = realized_nav * (1 + trades[ti].return_pct)
            ti += 1
        if ti < len(trades):
            t = trades[ti]
            if t.entry_date <= d <= t.exit_date:
                df = bars_map[t.symbol]
                if d in df.index:
                    nav_arr.append(realized_nav * float(df.loc[d, "Close"]) / t.entry_price)
                else:
                    nav_arr.append(nav_arr[-1] if nav_arr else realized_nav)
                continue
        nav_arr.append(realized_nav)

    equity = pd.Series(nav_arr, index=daily_idx)

    returns = np.array([t.return_pct for t in trades], dtype=float)
    win_rate = float((returns > 0).mean())
    avg_return = float(returns.mean())
    median_return = float(np.median(returns))
    std_return = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0

    span_days = (trades[-1].exit_date - trades[0].entry_date).days
    span_years = max(span_days / 365.25, 1e-9)
    final_nav = float(equity.iloc[-1]) if len(equity) else starting_nav
    total_return = float(final_nav / starting_nav - 1)
    ann_return = float((final_nav / starting_nav) ** (1.0 / span_years) - 1) if span_years > 0 else 0.0
    trades_per_year = len(trades) / span_years
    sharpe_per_trade = (avg_return / std_return) if std_return > 0 else None
    sharpe_ann = (sharpe_per_trade * np.sqrt(trades_per_year)
                  if sharpe_per_trade is not None else None)

    eq_arr = equity.values.astype(float)
    running_max = np.maximum.accumulate(eq_arr)
    max_dd = float((eq_arr / running_max - 1).min())

    summary = {
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_return,
        "median_trade_return": median_return,
        "trade_return_std": std_return,
        "total_return": total_return,
        "ann_return": ann_return,
        "trades_per_year": trades_per_year,
        "span_years": span_years,
        "sharpe_per_trade": sharpe_per_trade,
        "sharpe_ann": sharpe_ann,
        "max_drawdown": max_dd,
    }

    # Benchmark: buy-and-hold over the same span, normalized to starting_nav.
    benchmark_curve = pd.Series(dtype=float)
    if benchmark and len(daily_idx):
        try:
            bdf = yf.download(benchmark, period=period, interval="1d",
                              progress=False, auto_adjust=True)
            bdf = flatten_columns(bdf)
        except Exception:
            bdf = pd.DataFrame()
        if not bdf.empty:
            close = bdf["Close"].reindex(daily_idx, method="ffill")
            close = close.dropna()
            if len(close) > 1:
                bench_norm = close / float(close.iloc[0]) * starting_nav
                benchmark_curve = bench_norm
                bench_total = float(bench_norm.iloc[-1] / starting_nav - 1)
                bench_ann = float((bench_norm.iloc[-1] / starting_nav) ** (1.0 / span_years) - 1) \
                    if span_years > 0 else 0.0
                bench_dd = float((bench_norm / bench_norm.cummax() - 1).min())
                summary["benchmark"] = benchmark
                summary["benchmark_total_return"] = bench_total
                summary["benchmark_ann_return"] = bench_ann
                summary["benchmark_max_drawdown"] = bench_dd
                summary["alpha_total"] = total_return - bench_total
                summary["alpha_ann"] = ann_return - bench_ann

    return {"summary": summary, "trades": trades, "equity_curve": equity,
            "benchmark_curve": benchmark_curve}


def trades_to_dataframe(trades):
    return pd.DataFrame([{
        "symbol": t.symbol,
        "entry": t.entry_date.date() if hasattr(t.entry_date, "date") else t.entry_date,
        "exit": t.exit_date.date() if hasattr(t.exit_date, "date") else t.exit_date,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "return_pct": t.return_pct,
        "hold_days": t.holding_days,
    } for t in trades])
