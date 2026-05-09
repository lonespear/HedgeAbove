"""
Scanner: fetches market data, evaluates rules, dedups, fires alerts.
Headless-safe — no Streamlit imports anywhere in this call graph.

Each rule attached to a watchlist group is either *technical* (evaluated
against the most recent two daily bars + indicators) or *fundamental*
(evaluated against the get_stock_info dict). Per ticker we only fetch the
data sources actually needed by the rules attached to that group, so a
fundamentals-only watchlist doesn't pay for a 1y bar download, and a
technical-only watchlist doesn't pay for the slower ticker.info call.
"""
import json
from datetime import datetime
import yfinance as yf

from hedgeabove import config, db
from hedgeabove.indicators.technical import add_indicators, flatten_columns
from hedgeabove.rules import technical as tech_rules
from hedgeabove.rules import fundamental as fund_rules
from hedgeabove.alerts.telegram import send as send_telegram
from hedgeabove.data.market import get_stock_info


def _scan_ticker(symbol, rules, verbose=False):
    """Scan one ticker against a list of rule rows (id, rule_type, params_json).

    Returns the list of (rule_type, full_message) pairs that fired and were
    not already in the dedup table for today (UTC).
    """
    fired = []
    has_tech = any(rt in tech_rules.REGISTRY for _, rt, _ in rules)
    has_fund = any(rt in fund_rules.REGISTRY for _, rt, _ in rules)
    if not has_tech and not has_fund:
        return fired

    latest = prev = stock_info = None
    price = None

    try:
        if has_tech:
            df = yf.download(symbol, period=config.LOOKBACK, interval="1d",
                             progress=False, auto_adjust=True)
            df = flatten_columns(df)
            if df.empty or len(df) < config.MA_SLOW + 5:
                if verbose:
                    print(f"  {symbol}: insufficient bars ({len(df)} rows) — skipping technical rules")
            else:
                df = add_indicators(df, rsi_period=config.RSI_PERIOD,
                                    ma_fast=config.MA_FAST, ma_slow=config.MA_SLOW)
                latest, prev = df.iloc[-1], df.iloc[-2]
                price = float(latest["Close"])

        if has_fund:
            stock_info = get_stock_info(symbol)
            if stock_info and price is None:
                price = stock_info.get("current_price")

        for rule_id, rule_type, params_json in rules:
            params = json.loads(params_json) if params_json else {}

            if rule_type in tech_rules.REGISTRY:
                if latest is None:
                    continue
                msg = tech_rules.evaluate(rule_type, latest, prev, params)
            elif rule_type in fund_rules.REGISTRY:
                if stock_info is None:
                    continue
                msg = fund_rules.evaluate(rule_type, stock_info, params)
            else:
                if verbose:
                    print(f"  {symbol}: unknown rule type '{rule_type}' (skipping)")
                continue

            if msg is None:
                continue
            if db.alert_fired_today(symbol, rule_type):
                continue

            prefix = f"[{symbol}] ${price:.2f}" if price is not None else f"[{symbol}]"
            full = f"{prefix} — {msg}"
            db.log_alert(symbol, rule_type, full)
            fired.append((rule_type, full))
    except Exception as e:
        if verbose:
            print(f"  {symbol}: ERROR — {e}")
    return fired


def run(verbose=True):
    """Run a full scan over all watchlist groups. Returns count of alerts fired."""
    db.init_db()
    if verbose:
        print(f"\n=== Scan started {datetime.now():%Y-%m-%d %H:%M:%S} ===")

    total = 0
    groups = db.list_watchlist_groups()
    if not groups:
        if verbose:
            print("No watchlist groups configured. "
                  "Run: python -m hedgeabove.cli init")
        return 0

    for group_id, group_name in groups:
        tickers = db.get_watchlist_group_tickers(group_id)
        rules = db.list_alert_rules(group_id, enabled_only=True)
        if not tickers or not rules:
            if verbose:
                print(f"-- '{group_name}': skipped (tickers={len(tickers)}, rules={len(rules)})")
            continue
        if verbose:
            print(f"-- '{group_name}': {len(tickers)} tickers x {len(rules)} rules")
        for symbol in tickers:
            fired = _scan_ticker(symbol, rules, verbose=verbose)
            for _, msg in fired:
                if verbose:
                    print(f"  {msg}")
                send_telegram(msg)
                total += 1

    if verbose:
        print(f"=== Done. {total} alert(s) fired. ===\n")
    return total


if __name__ == "__main__":
    run()
