"""
Sector / industry classification via yfinance — free, no API key.

yfinance.ticker.info exposes 'sector' (e.g. 'Technology', 'Healthcare')
and 'industry' (more granular). Used by the cross-sectional scorer for
sector-neutral Z-scoring (Z within sector, so a tech P/E is compared to
other tech P/Es, not to utilities) and by `cli score --by-sector` for
grouped output.

Cached per-ticker in-process — sector classification rarely changes
intra-session, and the lookup is the slowest part of `info` calls.
"""
from functools import lru_cache


@lru_cache(maxsize=2048)
def get_sector(ticker):
    """Return the sector string for `ticker`, or 'Unknown' on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        sec = info.get("sector")
        if sec:
            return str(sec)
    except Exception:
        pass
    return "Unknown"


@lru_cache(maxsize=2048)
def get_industry(ticker):
    """Return the more-granular industry string for `ticker`, or 'Unknown'."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        ind = info.get("industry")
        if ind:
            return str(ind)
    except Exception:
        pass
    return "Unknown"
