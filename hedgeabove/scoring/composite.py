"""
Cross-sectional composite scoring: rank a universe of tickers by a
weighted combination of factors.

Each factor extracts a number from a ticker's data (fundamentals via
EDGAR + bars via yfinance). Within the universe, factors are Z-scored
so disparate units (P/E vs profit margin vs momentum) combine on the
same scale. The composite score is then a weighted sum:

    composite[i] = sum_f  w[f] * z[f, i]

Negative weights signal "lower is better" — pe with weight ``-0.4``
contributes most to the score for the cheapest names.

Output: a DataFrame indexed by symbol, ranked by composite score
descending. Pair with ``db.create_watchlist_group`` + ``add_ticker_to_group``
to convert the top-N into a scanner-monitored watchlist.

Design notes:
- Fundamentals come from ``data/edgar.py`` so scores are point-in-time
  if ``as_of`` is supplied (defaults to today).
- Tickers missing any factor in the weight set are dropped — composite
  ranks shouldn't be polluted by partial-data names.
- Parallel fetching uses a ThreadPoolExecutor capped at 8 workers to
  stay under the SEC's 10 req/s soft cap and yfinance's de-facto rate
  limit. ~5-10 minutes is realistic for the full S&P 500 cold; warm
  cache is seconds.
"""
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd
import yfinance as yf

from hedgeabove.data.edgar import get_fundamentals_as_of
from hedgeabove.indicators.technical import flatten_columns


# ── Factor library ──────────────────────────────────────────────

def _factor_pe(ticker, info, bars):
    return info.get("pe_ratio") if info else None


def _factor_pb(ticker, info, bars):
    return info.get("price_to_book") if info else None


def _factor_roe(ticker, info, bars):
    return info.get("roe") if info else None


def _factor_profit_margin(ticker, info, bars):
    return info.get("profit_margin") if info else None


def _factor_debt_to_equity(ticker, info, bars):
    return info.get("debt_to_equity") if info else None


def _factor_revenue_growth(ticker, info, bars):
    return info.get("revenue_growth") if info else None


def _factor_earnings_growth(ticker, info, bars):
    return info.get("earnings_growth") if info else None


def _factor_momentum_12_1(ticker, info, bars):
    """12-month return excluding the most recent ~21 trading days.
    Classic Asness/Moskowitz momentum factor — last month is dropped to
    avoid one-month reversal contamination."""
    if bars is None or len(bars) < 252:
        return None
    end_idx = len(bars) - 22
    start_idx = end_idx - 230
    if start_idx < 0:
        return None
    p_end = float(bars.iloc[end_idx]["Close"])
    p_start = float(bars.iloc[start_idx]["Close"])
    if p_start <= 0:
        return None
    return p_end / p_start - 1


def _factor_realized_vol_60d(ticker, info, bars):
    """Annualized 60-day realized volatility from daily log returns."""
    if bars is None or len(bars) < 61:
        return None
    rets = bars["Close"].pct_change().dropna().tail(60)
    if len(rets) < 30:
        return None
    return float(rets.std() * math.sqrt(252))


FACTORS = {
    "pe": _factor_pe,
    "pb": _factor_pb,
    "roe": _factor_roe,
    "profit_margin": _factor_profit_margin,
    "debt_to_equity": _factor_debt_to_equity,
    "revenue_growth": _factor_revenue_growth,
    "earnings_growth": _factor_earnings_growth,
    "momentum_12_1": _factor_momentum_12_1,
    "realized_vol_60d": _factor_realized_vol_60d,
}


# ── Presets ─────────────────────────────────────────────────────

PRESETS = {
    "quality": {
        "roe": 0.30,
        "profit_margin": 0.30,
        "debt_to_equity": -0.20,
        "revenue_growth": 0.20,
    },
    "value": {
        "pe": -0.40,
        "pb": -0.30,
        "roe": 0.30,
    },
    "growth": {
        "revenue_growth": 0.40,
        "earnings_growth": 0.30,
        "roe": 0.30,
    },
    "momentum": {
        "momentum_12_1": 1.00,
    },
    "low_vol": {
        "realized_vol_60d": -1.00,
    },
    "quality_value": {
        "pe": -0.20,
        "pb": -0.20,
        "roe": 0.30,
        "profit_margin": 0.30,
    },
}


# ── Fetcher + scorer ────────────────────────────────────────────

def _fetch_one(ticker, as_of):
    """Pull bars + fundamentals for one ticker. Returns (ticker, info, bars).
    Defensive — any exception yields (ticker, None, None) so a single
    flaky ticker doesn't kill a 500-name run."""
    try:
        bars = yf.download(ticker, period="2y", interval="1d",
                           progress=False, auto_adjust=True)
        bars = flatten_columns(bars)
        if bars.empty:
            bars = None
            current_price = None
        else:
            current_price = float(bars.iloc[-1]["Close"])
        info = None
        try:
            info = get_fundamentals_as_of(ticker, as_of, current_price=current_price)
        except Exception:
            pass
        return ticker, info, bars
    except Exception:
        return ticker, None, None


def score_universe(symbols, weights, as_of=None, max_workers=4, progress=None,
                   sector_neutral=False, attach_sector=False):
    """Score a list of symbols. See module docstring for details.

    Args:
      symbols: list of tickers
      weights: dict {factor_name: float}. Use negative weights for lower-better factors.
      as_of: date or ISO string for fundamentals lookback (default today).
      max_workers: parallel fetcher pool size.
      progress: optional callable(i, n, ticker) for progress reporting.
      sector_neutral: if True, Z-score each factor *within* sector instead of
        across the whole universe. Tech P/E gets compared to other tech P/Es
        (not utilities); useful for cross-sector strategies that don't want
        to over-pick e.g. cheap energy names just because energy P/Es are
        structurally lower than tech.
      attach_sector: if True, include a 'sector' column in the output even
        when sector_neutral is False. Implied True when sector_neutral=True.

    Returns:
      DataFrame ranked by composite_score descending. Index is symbol.
      Columns include the raw factor values + composite_score (and 'sector'
      when attached). Empty DataFrame if no ticker has all required factors.
    """
    as_of = as_of or date.today()

    unknown = set(weights) - set(FACTORS)
    if unknown:
        raise ValueError(f"Unknown factor(s): {sorted(unknown)}. "
                         f"Available: {list(FACTORS)}")

    # De-dupe the input so universes with repeats (universe.py has some)
    # don't double-fetch.
    seen = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    raw = {}
    n = len(symbols)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, sym, as_of): sym for sym in symbols}
        for i, future in enumerate(as_completed(futures), start=1):
            sym = futures[future]
            if progress is not None:
                progress(i, n, sym)
            try:
                _, info, bars = future.result()
            except Exception:
                info, bars = None, None
            raw[sym] = (info, bars)

    rows = []
    for sym, (info, bars) in raw.items():
        row = {"symbol": sym}
        all_present = True
        for factor in weights:
            val = FACTORS[factor](sym, info, bars)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                all_present = False
                break
            row[factor] = val
        if all_present:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    # Optionally tag with sector. Always required for sector-neutral mode.
    if sector_neutral or attach_sector:
        from hedgeabove.data.sectors import get_sector
        df["sector"] = df.index.to_series().apply(get_sector)

    composite = pd.Series(0.0, index=df.index)
    if sector_neutral:
        # Z-score each factor *within* its sector group.
        for factor, w in weights.items():
            zs = pd.Series(index=df.index, dtype=float)
            for sec, sub in df.groupby("sector"):
                vals = sub[factor]
                if len(vals) < 2:
                    zs.loc[sub.index] = 0.0
                    continue
                mean = vals.mean()
                std = vals.std(ddof=1)
                if std == 0 or pd.isna(std):
                    zs.loc[sub.index] = 0.0
                else:
                    zs.loc[sub.index] = (vals - mean) / std
            df[f"{factor}_z"] = zs
            composite = composite + w * zs
    else:
        # Z-score each factor across the whole universe.
        for factor, w in weights.items():
            vals = df[factor]
            mean = vals.mean()
            std = vals.std(ddof=1)
            if std == 0 or pd.isna(std):
                z = pd.Series(0.0, index=df.index)
            else:
                z = (vals - mean) / std
            df[f"{factor}_z"] = z
            composite = composite + w * z

    df["composite_score"] = composite
    df = df.sort_values("composite_score", ascending=False)
    return df


def score_with_preset(symbols, preset_name, **kwargs):
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name!r}. "
                         f"Available: {list(PRESETS)}")
    return score_universe(symbols, PRESETS[preset_name], **kwargs)
