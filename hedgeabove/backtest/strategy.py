"""
Strategy backtesting: walk fires across a basket and produce an equity curve.

The model: long-only basket with up to ``max_concurrent`` open positions.
At each fire date:
  - Close any positions whose hold period ended on or before today.
  - For each fresh fire, if we have capacity (open count < max_concurrent),
    open a position sized at ``1/max_concurrent`` of current NAV. Capital
    comes from cash; if cash is short the position is sized to available
    cash (concentrated in cash-tight regimes — caveat below).
  - Mark to market: NAV = cash + Σ(open positions valued at today's close).

When ``max_concurrent=1`` the model degenerates to the original sequential
single-position backtest: fires arriving while in a position are skipped,
NAV compounds trade-to-trade, equity curve is daily mark-to-market.

Caveats:
- Costs and slippage = 0. Real world is harder.
- Cash earns 0%. Easy to extend to a risk-free rate.
- When new fires arrive faster than positions close, the basket can grow
  cash-poor and undersize new entries. Real PMs would defer or rebalance;
  v1 just clips at available cash.
- Fundamental rules persist for ~90 days; single-position catches the
  first in-state day and skips the rest. Multi-position will absorb more
  of those crossings, possibly into many concurrent positions on the
  same name across different fire days — interpret accordingly.
"""
from collections import defaultdict
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


def simulate_basket(symbols, rule_type=None, params=None, period="5y", hold_days=20,
                    starting_nav=1.0, benchmark="SPY",
                    rules=None, combiner="all",
                    max_concurrent=1):
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
    from hedgeabove.backtest.signals import replay_rule, replay_composite

    if rules is not None and rule_type is not None:
        raise ValueError("Pass either rule_type (single) or rules (list), not both.")
    if rules is None and rule_type is None:
        raise ValueError("Need rule_type (single) or rules (list of (rt, params)).")
    params = params or {}

    # Bars per symbol — needed to find exit prices `hold_days` after entry.
    bars_map = {}
    for s in symbols:
        df = yf.download(s, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        df = flatten_columns(df)
        if not df.empty:
            bars_map[s] = df

    # Replay rule(s) on each symbol → chronological fire list.
    all_fires = []
    for s, df in bars_map.items():
        try:
            if rules is not None:
                fires = replay_composite(s, rules, combiner=combiner, period=period)
            else:
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

    # (Trade-walk and daily NAV are now built together by _simulate_daily below.)

    # Strategy now uses a unified daily-loop simulator with cash + open
    # positions. max_concurrent=1 falls back to the original sequential
    # single-position behavior (only one trade open at a time, NAV compounds).
    # max_concurrent>1 enables real concurrent-position bookkeeping.
    trades = []
    equity = pd.Series(dtype=float)
    if all_fires:
        trades, equity = _simulate_daily(all_fires, bars_map, hold_days,
                                         starting_nav, max_concurrent)
    if not trades:
        return empty

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


def _close_at(df, exit_date):
    """Resolve the close price on `exit_date` (exact or nearest)."""
    try:
        return float(df.loc[exit_date, "Close"])
    except KeyError:
        idx_arr = df.index.get_indexer([exit_date], method="nearest")
        idx = int(idx_arr[0])
        return float(df.iloc[idx]["Close"])


def _simulate_daily(all_fires, bars_map, hold_days, starting_nav, max_concurrent):
    """Daily portfolio simulator. Returns (trades_list, daily_nav_series).

    Walks every trading day in the span:
      1. Close any positions whose exit_date is on or before today.
      2. Process any fires arriving today, opening positions until either
         we hit max_concurrent or run out of cash.
      3. Mark all open positions to today's close; NAV = cash + Σ position values.
    """
    # Group fires by their date (pandas Timestamp), preserving fire order
    # so deterministic basket builds match across runs.
    fires_by_date = defaultdict(list)
    for fire_date, sym, price, msg in all_fires:
        fires_by_date[fire_date].append((sym, price, msg))

    # Build the union daily index spanning from first fire to last possible exit.
    all_dates = pd.DatetimeIndex(sorted(
        set().union(*(set(df.index) for df in bars_map.values()))
    ))
    span_start = all_fires[0][0]
    # Last possible exit = last fire's date + hold_days bars on its symbol's calendar
    last_fire_date, last_fire_sym, _, _ = all_fires[-1]
    last_df = bars_map[last_fire_sym]
    try:
        last_idx = last_df.index.get_loc(last_fire_date)
    except (KeyError, TypeError):
        last_idx = int(last_df.index.get_indexer([last_fire_date], method="nearest")[0])
    last_exit_idx = min(last_idx + hold_days, len(last_df) - 1)
    span_end = last_df.index[last_exit_idx]

    daily_idx = all_dates[(all_dates >= span_start) & (all_dates <= span_end)]

    cash = float(starting_nav)
    target_weight = 1.0 / max_concurrent
    open_positions: list = []  # dicts: {symbol, entry_date, entry_price, exit_date, capital_used, rule_message}
    closed_trades: list = []
    nav_vals: list = []

    def _mtm_open(d):
        """Mark-to-market value of all currently open positions on date d."""
        total = 0.0
        for p in open_positions:
            sym_df = bars_map[p["symbol"]]
            if d in sym_df.index:
                px = float(sym_df.loc[d, "Close"])
            else:
                idx_arr = sym_df.index.get_indexer([d], method="ffill")
                if int(idx_arr[0]) < 0:
                    continue
                px = float(sym_df.iloc[int(idx_arr[0])]["Close"])
            total += p["capital_used"] * px / p["entry_price"]
        return total

    for d in daily_idx:
        # 1. Close positions whose hold has elapsed.
        still_open = []
        for p in open_positions:
            if d >= p["exit_date"]:
                exit_price = _close_at(bars_map[p["symbol"]], p["exit_date"])
                cash += p["capital_used"] * exit_price / p["entry_price"]
                closed_trades.append(Trade(
                    symbol=p["symbol"],
                    entry_date=p["entry_date"],
                    entry_price=p["entry_price"],
                    exit_date=p["exit_date"],
                    exit_price=exit_price,
                    return_pct=exit_price / p["entry_price"] - 1.0,
                    holding_days=hold_days,
                    rule_message=p["rule_message"],
                ))
            else:
                still_open.append(p)
        open_positions = still_open

        # 2. Process today's fires.
        for sym, entry_price, msg in fires_by_date.get(d, []):
            if len(open_positions) >= max_concurrent:
                break  # at capacity for the rest of the day
            sym_df = bars_map[sym]
            try:
                entry_idx = sym_df.index.get_loc(d)
            except (KeyError, TypeError):
                continue
            exit_idx = entry_idx + hold_days
            if exit_idx >= len(sym_df):
                continue  # not enough future bars to close cleanly
            exit_date = sym_df.index[exit_idx]

            current_nav = cash + _mtm_open(d)
            capital = min(current_nav * target_weight, cash)
            if capital <= 0:
                continue
            cash -= capital
            open_positions.append({
                "symbol": sym,
                "entry_date": d,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "capital_used": capital,
                "rule_message": msg,
            })

        # 3. Daily NAV.
        nav_vals.append(cash + _mtm_open(d))

    # Anything still open at end of span: close at last available bar.
    for p in open_positions:
        sym_df = bars_map[p["symbol"]]
        last_close = float(sym_df.iloc[-1]["Close"])
        cash += p["capital_used"] * last_close / p["entry_price"]
        closed_trades.append(Trade(
            symbol=p["symbol"],
            entry_date=p["entry_date"],
            entry_price=p["entry_price"],
            exit_date=sym_df.index[-1],
            exit_price=last_close,
            return_pct=last_close / p["entry_price"] - 1.0,
            holding_days=hold_days,
            rule_message=p["rule_message"],
        ))

    closed_trades.sort(key=lambda t: t.entry_date)
    equity = pd.Series(nav_vals, index=daily_idx)
    return closed_trades, equity


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
