"""
Fundamental rule evaluators.

Each rule signature: ``fn(stock_info, params) -> str | None``

`stock_info` is the dict returned by ``hedgeabove.data.market.get_stock_info``.
Returning a string means the rule fired. Returning None means quiet (or that
the underlying data was missing — fundamentals aren't always populated for
every ticker, e.g. crypto won't have P/E).

Defaults are sensible v1 thresholds for screening; override per-rule via the
``alert_rules.params_json`` column, e.g. ``{"threshold": 25}``.
"""
REGISTRY = {}


def rule(key):
    def deco(fn):
        REGISTRY[key] = fn
        return fn
    return deco


def _pos(v):
    """Return v if it's a positive finite number, else None."""
    try:
        if v is None:
            return None
        v = float(v)
        if v != v or v <= 0:  # NaN or non-positive
            return None
        return v
    except (TypeError, ValueError):
        return None


def _frac_to_pct(v):
    """yfinance often reports growth/margins as fractions (0.15 = 15%).
    Sometimes already as percent. Treat |v| > 1 as already percent."""
    if v is None:
        return None
    return v if abs(v) > 1 else v * 100


@rule("pe_below")
def pe_below(info, params):
    """Fires when trailing P/E is below threshold. Params: threshold (default 15)."""
    threshold = float(params.get("threshold", 15.0))
    pe = _pos(info.get("pe_ratio"))
    if pe is None:
        return None
    if pe < threshold:
        return f"P/E {pe:.1f} below threshold {threshold:.1f}"
    return None


@rule("pe_above")
def pe_above(info, params):
    """Fires when trailing P/E is above threshold. Params: threshold (default 40)."""
    threshold = float(params.get("threshold", 40.0))
    pe = _pos(info.get("pe_ratio"))
    if pe is None:
        return None
    if pe > threshold:
        return f"P/E {pe:.1f} above threshold {threshold:.1f}"
    return None


@rule("pb_below")
def pb_below(info, params):
    """Fires when P/B is below threshold. Params: threshold (default 1.5)."""
    threshold = float(params.get("threshold", 1.5))
    pb = _pos(info.get("price_to_book"))
    if pb is None:
        return None
    if pb < threshold:
        return f"P/B {pb:.2f} below threshold {threshold:.2f}"
    return None


@rule("dividend_yield_above")
def dividend_yield_above(info, params):
    """Fires when dividend yield (in %) is above threshold. Params: threshold (default 3.0 = 3%)."""
    threshold_pct = float(params.get("threshold", 3.0))
    dy = info.get("dividend_yield")
    if dy is None or dy == 0:
        return None
    dy_pct = _frac_to_pct(dy)
    if dy_pct is not None and dy_pct > threshold_pct:
        return f"Dividend yield {dy_pct:.2f}% above {threshold_pct:.1f}%"
    return None


@rule("revenue_growth_above")
def revenue_growth_above(info, params):
    """Fires when YoY revenue growth (%) is above threshold. Params: threshold (default 15)."""
    threshold_pct = float(params.get("threshold", 15.0))
    g = info.get("revenue_growth")
    if g is None:
        return None
    g_pct = _frac_to_pct(g)
    if g_pct > threshold_pct:
        return f"Revenue growth {g_pct:.1f}% above {threshold_pct:.1f}%"
    return None


@rule("earnings_growth_above")
def earnings_growth_above(info, params):
    """Fires when earnings growth (%) is above threshold. Params: threshold (default 20)."""
    threshold_pct = float(params.get("threshold", 20.0))
    g = info.get("earnings_growth")
    if g is None:
        return None
    g_pct = _frac_to_pct(g)
    if g_pct > threshold_pct:
        return f"Earnings growth {g_pct:.1f}% above {threshold_pct:.1f}%"
    return None


@rule("roe_above")
def roe_above(info, params):
    """Fires when return on equity (%) is above threshold. Params: threshold (default 20)."""
    threshold_pct = float(params.get("threshold", 20.0))
    roe = info.get("roe")
    if roe is None:
        return None
    roe_pct = _frac_to_pct(roe)
    if roe_pct > threshold_pct:
        return f"ROE {roe_pct:.1f}% above {threshold_pct:.1f}%"
    return None


@rule("profit_margin_above")
def profit_margin_above(info, params):
    """Fires when profit margin (%) is above threshold. Params: threshold (default 20)."""
    threshold_pct = float(params.get("threshold", 20.0))
    pm = info.get("profit_margin")
    if pm is None:
        return None
    pm_pct = _frac_to_pct(pm)
    if pm_pct > threshold_pct:
        return f"Profit margin {pm_pct:.1f}% above {threshold_pct:.1f}%"
    return None


@rule("debt_to_equity_below")
def debt_to_equity_below(info, params):
    """Fires when debt/equity is below threshold. Params: threshold (default 50)."""
    threshold = float(params.get("threshold", 50.0))
    de = info.get("debt_to_equity")
    if de is None:
        return None
    if de < threshold:
        return f"D/E {de:.1f} below threshold {threshold:.1f}"
    return None


@rule("analyst_upside_above")
def analyst_upside_above(info, params):
    """Fires when analyst mean target / current price - 1 (%) is above threshold. Params: threshold (default 20)."""
    threshold_pct = float(params.get("threshold", 20.0))
    target = _pos(info.get("target_mean_price"))
    price = _pos(info.get("current_price"))
    if target is None or price is None:
        return None
    upside_pct = (target / price - 1) * 100
    if upside_pct > threshold_pct:
        return (f"Analyst upside {upside_pct:.1f}% above {threshold_pct:.1f}% "
                f"(target ${target:.2f} vs current ${price:.2f})")
    return None


def evaluate(rule_type, stock_info, params=None):
    """Run one fundamental rule. Returns alert message if fired, else None."""
    fn = REGISTRY.get(rule_type)
    if fn is None or stock_info is None:
        return None
    return fn(stock_info, params or {})


def available_rules():
    return sorted(REGISTRY.keys())


def get_doc(rule_type):
    """Return the rule's docstring (first line) — used for UI hints + CLI help."""
    fn = REGISTRY.get(rule_type)
    if fn is None or not fn.__doc__:
        return ""
    return fn.__doc__.strip().split("\n", 1)[0]
