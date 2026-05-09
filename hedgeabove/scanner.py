"""
Scanner: fetches yfinance data, evaluates rules, dedups, fires alerts.
Headless-safe — no Streamlit imports anywhere in this call graph.

Iterates every enabled watchlist group's tickers x rules. For each ticker it
pulls a `LOOKBACK` window of daily bars, layers RSI/MACD/SMA via pandas-ta,
and asks each rule whether the latest two bars trigger. Fires Telegram for
every (symbol, rule) pair that hasn't already fired today (UTC).
"""
import json
from datetime import datetime
import yfinance as yf

from hedgeabove import config, db
from hedgeabove.indicators.technical import add_indicators, flatten_columns
from hedgeabove.rules import technical as tech_rules
from hedgeabove.alerts.telegram import send as send_telegram


def _scan_ticker(symbol, rules, verbose=False):
    """Scan one ticker against a list of rule rows (id, rule_type, params_json).

    Returns the list of (rule_type, message) pairs that fired.
    """
    fired = []
    try:
        df = yf.download(symbol, period=config.LOOKBACK, interval="1d",
                         progress=False, auto_adjust=True)
        df = flatten_columns(df)
        if df.empty or len(df) < config.MA_SLOW + 5:
            if verbose:
                print(f"  {symbol}: insufficient data ({len(df)} rows)")
            return fired
        df = add_indicators(df, rsi_period=config.RSI_PERIOD,
                            ma_fast=config.MA_FAST, ma_slow=config.MA_SLOW)
        latest, prev = df.iloc[-1], df.iloc[-2]
        price = float(latest["Close"])

        for rule_id, rule_type, params_json in rules:
            params = json.loads(params_json) if params_json else {}
            msg = tech_rules.evaluate(rule_type, latest, prev, params)
            if msg is None:
                continue
            if db.alert_fired_today(symbol, rule_type):
                continue
            full = f"[{symbol}] ${price:.2f} — {msg}"
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
