"""
Earnings calendar via yfinance — free, no API key needed.

yfinance.ticker.earnings_dates returns the most recent ~25 quarterly
earnings dates per company (≈6 years of history) plus any scheduled
upcoming dates. That's enough for typical 1-10y backtests where you
want to skip trades inside an earnings window to avoid gap risk.

Helpers:
  get_earnings_dates(ticker) -> list[date]    (sorted, deduped, cached)
  is_within_earnings_window(ticker, target, days_before, days_after) -> bool

A "window" of (5, 5) means: skip if target date is between
earnings_date - 5 days and earnings_date + 5 days inclusive.
"""
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=512)
def get_earnings_dates(ticker):
    """Return sorted list of date objects for `ticker`'s known earnings.
    Empty list on failure (network error, delisted, etc.) — never raises."""
    try:
        import yfinance as yf
        ed = yf.Ticker(ticker).earnings_dates
    except Exception:
        return []
    if ed is None or ed.empty:
        return []
    dates = []
    for ts in ed.index:
        try:
            if hasattr(ts, "date"):
                dates.append(ts.date())
            else:
                dates.append(pd.Timestamp(ts).date())
        except Exception:
            continue
    return sorted(set(dates))


def _to_date(d):
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.fromisoformat(d.replace("Z", "")).date()
    if hasattr(d, "date"):
        try:
            return d.date()
        except Exception:
            pass
    return None


def is_within_earnings_window(ticker, target_date, days_before=5, days_after=5):
    """True if `target_date` falls inside the inclusive
    [earnings - days_before, earnings + days_after] window for any known
    earnings date of `ticker`."""
    td = _to_date(target_date)
    if td is None:
        return False
    edates = get_earnings_dates(ticker)
    for ed in edates:
        if (ed - timedelta(days=days_before)) <= td <= (ed + timedelta(days=days_after)):
            return True
    return False
