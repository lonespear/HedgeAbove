"""
SEC EDGAR data layer — point-in-time fundamentals for backtesting.

Defeats the look-ahead bias inherent in yfinance's "today's snapshot" of
fundamentals. Every fact returned here is filtered to ``filing_date <=
as_of``, with restated/amended values deduped to the *earliest* filing per
(concept, period_end) so backtests don't accidentally use a value that
wasn't available on the as-of date.

Source: SEC EDGAR XBRL companyfacts API (free, no key, only an Identity
header per SEC fair-use policy). Set ``EDGAR_IDENTITY`` in .env to your
"Name email@example.com" string; required by the SEC.

Performance: ~25k facts per large filer. We index by concept once per
ticker (LRU-cached) so per-as-of queries iterate only the small concept
slice.

Example::

    from hedgeabove.data.edgar import get_fundamentals_as_of
    info = get_fundamentals_as_of("AAPL", "2024-06-30", current_price=215.20)
    # info['pe_ratio'] is the trailing P/E using only data filed by 2024-06-30.
"""
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional

_IDENTITY_SET = False


def _ensure_identity():
    """Set EDGAR identity header. SEC requires it on every request."""
    global _IDENTITY_SET
    if _IDENTITY_SET:
        return
    from edgar import set_identity
    identity = os.getenv("EDGAR_IDENTITY", "HedgeAbove jonathan.day808@gmail.com")
    set_identity(identity)
    _IDENTITY_SET = True


@lru_cache(maxsize=128)
def _get_company_facts(ticker):
    """Fetch + cache the EntityFacts object for a ticker."""
    _ensure_identity()
    from edgar import Company
    return Company(ticker).get_facts()


@lru_cache(maxsize=128)
def _facts_by_concept(ticker):
    """Group all facts for a ticker by concept (e.g. 'us-gaap:Revenues').
    Cached per-ticker so we only build the index once per process."""
    facts = _get_company_facts(ticker)
    by_concept: dict = {}
    for f in facts.get_all_facts():
        by_concept.setdefault(f.concept, []).append(f)
    return by_concept


def _to_date(d) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.fromisoformat(d.replace("Z", "")).date()
    return None


def _filter_as_of(ticker, concept, as_of):
    """Return facts for `concept` with filing_date <= as_of, deduped to the
    earliest filing per (period_start, period_end). Earliest filing wins so
    later amendments don't leak corrected numbers into a historical backtest.
    """
    as_of = _to_date(as_of)
    if as_of is None:
        return []
    facts = _facts_by_concept(ticker).get(concept, [])

    by_period: dict = {}
    for f in facts:
        fd = _to_date(getattr(f, "filing_date", None))
        if fd is None or fd > as_of:
            continue
        key = (getattr(f, "period_start", None), getattr(f, "period_end", None))
        prev = by_period.get(key)
        if prev is None or fd < _to_date(prev.filing_date):
            by_period[key] = f
    return sorted(by_period.values(), key=lambda f: _to_date(f.period_end) or date.min)


def _quarterly_only(facts):
    """Filter a fact list to quarterly duration entries (~91-day spans)."""
    out = []
    for f in facts:
        if getattr(f, "period_type", None) != "duration":
            continue
        ps = _to_date(f.period_start)
        pe = _to_date(f.period_end)
        if ps is None or pe is None:
            continue
        span = (pe - ps).days
        if 80 <= span <= 100:
            out.append(f)
    return out


def _instant_only(facts):
    return [f for f in facts if getattr(f, "period_type", None) == "instant"]


def _try_concepts(ticker, concepts, as_of, prefer_quarterly=True):
    """Return facts for the candidate concept that has the deepest history
    (most quarterly facts when ``prefer_quarterly``, else most facts of any
    kind). EDGAR uses different us-gaap concepts for the same economic
    quantity across filers — e.g. NVDA reports revenue under ``us-gaap:Revenues``
    while AAPL uses ``us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax``.
    Returning the deepest series picks the right one per filer instead of the
    first one alphabetically.
    """
    best_concept = None
    best_facts: list = []
    best_score = 0
    for c in concepts:
        facts = _filter_as_of(ticker, c, as_of)
        if not facts:
            continue
        if prefer_quarterly:
            score = len(_quarterly_only(facts))
        else:
            score = len(facts)
        if score > best_score:
            best_score = score
            best_concept = c
            best_facts = facts
    return best_concept, best_facts


# Concept priority lists. Order matters: we use the first one that has data.
_REVENUE_CONCEPTS = (
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "us-gaap:SalesRevenueGoodsNet",
)
_NET_INCOME_CONCEPTS = (
    "us-gaap:NetIncomeLoss",
    "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
)
_EPS_CONCEPTS = (
    "us-gaap:EarningsPerShareDiluted",
    "us-gaap:EarningsPerShareBasic",
)
_EQUITY_CONCEPTS = (
    "us-gaap:StockholdersEquity",
    "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
_DEBT_CONCEPTS = (
    "us-gaap:LongTermDebt",
    "us-gaap:LongTermDebtNoncurrent",
)
_SHARES_CONCEPTS = (
    "dei:EntityCommonStockSharesOutstanding",
    "us-gaap:CommonStockSharesOutstanding",
)


def _ttm_sum(facts, n=4):
    """Sum the n most-recent quarterly values. Returns None if fewer than n quarters."""
    qs = _quarterly_only(facts)
    if len(qs) < n:
        return None
    qs_sorted = sorted(qs, key=lambda f: _to_date(f.period_end))
    return float(sum(f.numeric_value for f in qs_sorted[-n:] if f.numeric_value is not None))


def _ttm_yoy_growth(facts):
    """(TTM ending most recent quarter) / (TTM ending 4 quarters prior) - 1.
    Both windows are derived only from facts in `facts`, so look-ahead-safe."""
    qs = _quarterly_only(facts)
    if len(qs) < 8:
        return None
    qs_sorted = sorted(qs, key=lambda f: _to_date(f.period_end))
    recent4 = qs_sorted[-4:]
    prior4 = qs_sorted[-8:-4]
    ttm_now = sum(f.numeric_value for f in recent4 if f.numeric_value is not None)
    ttm_prior = sum(f.numeric_value for f in prior4 if f.numeric_value is not None)
    if ttm_prior is None or ttm_prior <= 0:
        return None
    return float(ttm_now) / float(ttm_prior) - 1.0


def _latest_instant(facts):
    """Most-recent instant-type value (balance-sheet items)."""
    insts = _instant_only(facts)
    if not insts:
        return None
    insts.sort(key=lambda f: _to_date(f.period_end))
    return float(insts[-1].numeric_value) if insts[-1].numeric_value is not None else None


def get_fundamentals_as_of(ticker, as_of, current_price=None):
    """Build a yfinance-compatible fundamentals dict using only data filed by
    ``as_of``. Same key shape as ``data/market.py``\\'s ``get_stock_info`` so
    the existing fundamental rules consume it without changes.

    Args:
      ticker: ticker symbol (e.g. 'AAPL').
      as_of: date or ISO string. All facts must be filed on or before this date.
      current_price: historical close on as_of_date (optional). Required for
        ratio-shaped rules (P/E, P/B, dividend_yield). Without it those keys
        will be None.

    Returns:
      dict with keys: pe_ratio, price_to_book, revenue_growth, earnings_growth,
      roe, profit_margin, debt_to_equity, current_price, target_mean_price,
      dividend_yield, plus ttm_revenue / ttm_net_income / shares_outstanding /
      stockholders_equity for transparency.

    Some keys are deliberately None because they aren't recoverable from
    EDGAR alone: ``target_mean_price`` (analyst data isn't filed with the SEC)
    and ``dividend_yield`` (a future enhancement; needs TTM dividends paid).
    """
    info = {
        "symbol": ticker,
        "current_price": current_price,
        "pe_ratio": None,
        "price_to_book": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "roe": None,
        "profit_margin": None,
        "debt_to_equity": None,
        "dividend_yield": None,
        "target_mean_price": None,
        # Raw transparency
        "ttm_revenue": None,
        "ttm_net_income": None,
        "ttm_eps": None,
        "stockholders_equity": None,
        "long_term_debt": None,
        "shares_outstanding": None,
    }

    try:
        # Income-statement TTMs
        _, rev_facts = _try_concepts(ticker, _REVENUE_CONCEPTS, as_of)
        info["ttm_revenue"] = _ttm_sum(rev_facts)
        info["revenue_growth"] = _ttm_yoy_growth(rev_facts)

        _, ni_facts = _try_concepts(ticker, _NET_INCOME_CONCEPTS, as_of)
        info["ttm_net_income"] = _ttm_sum(ni_facts)
        info["earnings_growth"] = _ttm_yoy_growth(ni_facts)

        _, eps_facts = _try_concepts(ticker, _EPS_CONCEPTS, as_of)
        info["ttm_eps"] = _ttm_sum(eps_facts)

        # Balance sheet (instant) — score by total fact count, not quarterly
        _, eq_facts = _try_concepts(ticker, _EQUITY_CONCEPTS, as_of, prefer_quarterly=False)
        info["stockholders_equity"] = _latest_instant(eq_facts)

        _, debt_facts = _try_concepts(ticker, _DEBT_CONCEPTS, as_of, prefer_quarterly=False)
        info["long_term_debt"] = _latest_instant(debt_facts)

        _, sh_facts = _try_concepts(ticker, _SHARES_CONCEPTS, as_of, prefer_quarterly=False)
        info["shares_outstanding"] = _latest_instant(sh_facts)

        # Derived ratios
        eps = info["ttm_eps"]
        if current_price is not None and eps is not None and eps > 0:
            info["pe_ratio"] = current_price / eps

        equity = info["stockholders_equity"]
        shares = info["shares_outstanding"]
        if (current_price is not None and shares is not None and shares > 0
                and equity is not None and equity > 0):
            book_value_per_share = equity / shares
            if book_value_per_share > 0:
                info["price_to_book"] = current_price / book_value_per_share

        ttm_ni = info["ttm_net_income"]
        if ttm_ni is not None and equity is not None and equity > 0:
            info["roe"] = ttm_ni / equity

        ttm_rev = info["ttm_revenue"]
        if ttm_ni is not None and ttm_rev is not None and ttm_rev > 0:
            info["profit_margin"] = ttm_ni / ttm_rev

        ltd = info["long_term_debt"]
        if ltd is not None and equity is not None and equity > 0:
            # yfinance reports debt_to_equity as percent (50 = 50%);
            # match that convention so our existing rules' thresholds work.
            info["debt_to_equity"] = ltd / equity * 100

    except Exception:
        # Defensive: never let a single fact-extraction failure kill a backtest run.
        pass

    return info
