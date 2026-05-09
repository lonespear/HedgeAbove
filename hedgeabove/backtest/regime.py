"""
Regime-conditional analysis. Tag each rule fire (or trade) with the
prevailing macro regime on its date and aggregate forward returns per
regime — answers "does this signal still work in high-vol environments?"

Currently supported regimes:
  vix          : low_vol (<20) / med_vol (20-30) / high_vol (>30) via FRED VIXCLS
  yield_curve  : positive / flat / inverted (10y-2y) via FRED T10Y2Y

Both use forward-fill alignment so weekend/holiday gaps in the macro
series don't cause "unknown" labels for trading-day fires.
"""
import pandas as pd

from hedgeabove.data import macro


_REGIME_CONFIG = {
    "vix":         (macro.get_vix,                 macro.vix_regime),
    "yield_curve": (macro.get_yield_curve_slope,   macro.yield_curve_regime),
}


def available_regimes():
    return sorted(_REGIME_CONFIG)


def classify_dates(dates, regime_name):
    """Return a list of regime labels for a list of dates, using the named
    indicator. Forward-fill alignment: each date gets the most recent value
    on or before it (handles weekends/holidays gracefully)."""
    if regime_name not in _REGIME_CONFIG:
        raise ValueError(f"Unknown regime: {regime_name!r}. "
                         f"Available: {available_regimes()}")
    fetcher, classifier = _REGIME_CONFIG[regime_name]
    series = fetcher().sort_index()
    series.index = pd.to_datetime(series.index)

    labels = []
    for d in dates:
        d = pd.to_datetime(d)
        idx = series.index.searchsorted(d, side="right") - 1
        if idx < 0:
            labels.append("unknown")
        else:
            labels.append(classifier(series.iloc[idx]))
    return labels


def by_regime_summary(fires_or_trades, regime_name, horizon=20):
    """Aggregate forward returns by regime label.

    Accepts either a list of FireEvent (uses fwd_returns[horizon]) or
    a list of Trade (uses return_pct, ignores horizon arg). Returns a
    DataFrame with columns: regime, n, hit_rate, avg, median, std.
    """
    if not fires_or_trades:
        return pd.DataFrame()

    is_fire = hasattr(fires_or_trades[0], "fwd_returns")
    dates = [item.fire_date if is_fire else item.entry_date for item in fires_or_trades]
    labels = classify_dates(dates, regime_name)

    if is_fire:
        rets = [item.fwd_returns.get(horizon) for item in fires_or_trades]
    else:
        rets = [item.return_pct for item in fires_or_trades]

    df = pd.DataFrame({"regime": labels, "ret": rets}).dropna(subset=["ret"])
    if df.empty:
        return df

    grouped = df.groupby("regime")["ret"]
    agg = pd.DataFrame({
        "n": grouped.size(),
        "hit_rate": grouped.apply(lambda x: float((x > 0).mean())),
        "avg": grouped.mean(),
        "median": grouped.median(),
        "std": grouped.apply(lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
    }).reset_index()
    # Order columns so high-conviction (high n) regimes show first
    return agg.sort_values("n", ascending=False).reset_index(drop=True)
