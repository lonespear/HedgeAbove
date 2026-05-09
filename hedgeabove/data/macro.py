"""
Macro data via FRED — free, no API key needed (pandas-datareader handles it).

Series cheat sheet (the ones we use for regime classification):
  VIXCLS    Cboe VIX index (daily)
  T10Y2Y    10-year minus 2-year Treasury yield (daily) — recession signal when negative
  UNRATE    Unemployment rate (monthly)
  SAHMREALTIME   Sahm rule recession indicator (monthly)

Returns pandas Series indexed by datetime. Dropna on fetch so callers
don't have to handle FRED's missing-day pattern.
"""
from datetime import date

import pandas as pd

_DEFAULT_START = "2000-01-01"


def get_series(series_id, start=None, end=None):
    """Fetch a FRED series. Returns pd.Series indexed by date."""
    import pandas_datareader.data as pdr
    df = pdr.DataReader(series_id, "fred",
                        start=start or _DEFAULT_START,
                        end=end or date.today())
    s = df[series_id].dropna()
    s.name = series_id
    return s


def get_vix(start=None, end=None):
    return get_series("VIXCLS", start, end)


def get_yield_curve_slope(start=None, end=None):
    """10y-2y slope. Positive = normal; negative = inverted (recession signal)."""
    return get_series("T10Y2Y", start, end)


def get_unemployment(start=None, end=None):
    return get_series("UNRATE", start, end)


def get_sahm_rule(start=None, end=None):
    """Sahm rule recession indicator. >=0.5 historically signals recession start."""
    return get_series("SAHMREALTIME", start, end)


# ── Regime classifiers ──────────────────────────────────────────

def vix_regime(value, low=20.0, high=30.0):
    """Bucket a single VIX value into low/med/high vol."""
    if value is None or pd.isna(value):
        return "unknown"
    if value < low:
        return "low_vol"
    if value > high:
        return "high_vol"
    return "med_vol"


def yield_curve_regime(slope, flat_band=0.25):
    """Bucket the 2s10s slope into positive / flat / inverted."""
    if slope is None or pd.isna(slope):
        return "unknown"
    if slope > flat_band:
        return "positive"
    if slope < -flat_band:
        return "inverted"
    return "flat"
