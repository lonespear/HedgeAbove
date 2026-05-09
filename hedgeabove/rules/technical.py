"""
Technical rule evaluators.

Each rule is registered via @rule(key) and has signature:
    fn(latest_row, prev_row, params_dict) -> message_str | None

Scanner iterates each enabled rule against the most recent two daily bars
(latest, prev) of an indicator-augmented DataFrame. Returning a string means
the rule fired and should produce an alert; returning None means quiet.
"""
from hedgeabove import config

REGISTRY = {}


def rule(key):
    def deco(fn):
        REGISTRY[key] = fn
        return fn
    return deco


@rule("rsi_oversold")
def rsi_oversold(latest, prev, params):
    threshold = params.get("threshold", config.RSI_OVERSOLD)
    period = params.get("period", config.RSI_PERIOD)
    rsi = latest.get(f"RSI_{period}")
    if rsi is None or rsi >= threshold:
        return None
    return f"RSI OVERSOLD ({rsi:.1f} < {threshold})"


@rule("rsi_overbought")
def rsi_overbought(latest, prev, params):
    threshold = params.get("threshold", config.RSI_OVERBOUGHT)
    period = params.get("period", config.RSI_PERIOD)
    rsi = latest.get(f"RSI_{period}")
    if rsi is None or rsi <= threshold:
        return None
    return f"RSI OVERBOUGHT ({rsi:.1f} > {threshold})"


def _macd_pair(latest, prev):
    return (
        latest.get("MACD_12_26_9"),
        latest.get("MACDs_12_26_9"),
        prev.get("MACD_12_26_9"),
        prev.get("MACDs_12_26_9"),
    )


@rule("macd_bullish_cross")
def macd_bullish_cross(latest, prev, params):
    macd, sig, pmacd, psig = _macd_pair(latest, prev)
    if None in (macd, sig, pmacd, psig):
        return None
    if pmacd < psig and macd > sig:
        return "MACD BULLISH CROSSOVER"
    return None


@rule("macd_bearish_cross")
def macd_bearish_cross(latest, prev, params):
    macd, sig, pmacd, psig = _macd_pair(latest, prev)
    if None in (macd, sig, pmacd, psig):
        return None
    if pmacd > psig and macd < sig:
        return "MACD BEARISH CROSSOVER"
    return None


def _ma_pair(latest, prev, fast, slow):
    return (
        latest.get(f"SMA_{fast}"),
        latest.get(f"SMA_{slow}"),
        prev.get(f"SMA_{fast}"),
        prev.get(f"SMA_{slow}"),
    )


@rule("golden_cross")
def golden_cross(latest, prev, params):
    fast = params.get("ma_fast", config.MA_FAST)
    slow = params.get("ma_slow", config.MA_SLOW)
    f, s, pf, ps = _ma_pair(latest, prev, fast, slow)
    if None in (f, s, pf, ps):
        return None
    if pf < ps and f > s:
        return f"GOLDEN CROSS ({fast}MA crossed above {slow}MA)"
    return None


@rule("death_cross")
def death_cross(latest, prev, params):
    fast = params.get("ma_fast", config.MA_FAST)
    slow = params.get("ma_slow", config.MA_SLOW)
    f, s, pf, ps = _ma_pair(latest, prev, fast, slow)
    if None in (f, s, pf, ps):
        return None
    if pf > ps and f < s:
        return f"DEATH CROSS ({fast}MA crossed below {slow}MA)"
    return None


# ── Bollinger Band rules ────────────────────────────────────────

def _bb_cols(params):
    length = int(params.get("length", 20))
    std = float(params.get("std", 2.0))
    std_str = f"{std:.1f}"
    return f"BBL_{length}_{std_str}", f"BBU_{length}_{std_str}", f"BBB_{length}_{std_str}"


@rule("bollinger_breakout_up")
def bollinger_breakout_up(latest, prev, params):
    bbl, bbu, _ = _bb_cols(params)
    pu, u = prev.get(bbu), latest.get(bbu)
    pp, p = prev.get("Close"), latest.get("Close")
    if None in (pu, u, pp, p):
        return None
    if pp <= pu and p > u:
        return f"BOLLINGER BREAKOUT UP (close ${p:.2f} crossed above upper band ${u:.2f})"
    return None


@rule("bollinger_breakout_down")
def bollinger_breakout_down(latest, prev, params):
    bbl, _, _ = _bb_cols(params)
    pl, l = prev.get(bbl), latest.get(bbl)
    pp, p = prev.get("Close"), latest.get("Close")
    if None in (pl, l, pp, p):
        return None
    if pp >= pl and p < l:
        return f"BOLLINGER BREAKOUT DOWN (close ${p:.2f} crossed below lower band ${l:.2f})"
    return None


# ── Volume / range rules ────────────────────────────────────────

@rule("volume_spike")
def volume_spike(latest, prev, params):
    multiplier = float(params.get("multiplier", 2.0))
    period = int(params.get("period", 20))
    vol = latest.get("Volume")
    avg = latest.get(f"VOL_MA_{period}")
    if vol is None or avg is None or avg <= 0:
        return None
    ratio = vol / avg
    if ratio >= multiplier:
        return f"VOLUME SPIKE ({ratio:.1f}x avg{period})"
    return None


@rule("high_52w_breakout")
def high_52w_breakout(latest, prev, params):
    lookback = int(params.get("lookback", 252))
    col = f"HIGH_{lookback}"
    h = latest.get(col)
    p = latest.get("Close")
    pp = prev.get("Close")
    ph = prev.get(col)
    if None in (h, p, pp, ph):
        return None
    # Today's close breaks above the prior-{lookback}-day high, and yesterday's didn't
    if pp <= ph and p > h:
        return f"52-WEEK HIGH BREAKOUT (close ${p:.2f} > prior {lookback}d high ${h:.2f})"
    return None


@rule("low_52w_breakout")
def low_52w_breakout(latest, prev, params):
    lookback = int(params.get("lookback", 252))
    col = f"LOW_{lookback}"
    l = latest.get(col)
    p = latest.get("Close")
    pp = prev.get("Close")
    pl = prev.get(col)
    if None in (l, p, pp, pl):
        return None
    if pp >= pl and p < l:
        return f"52-WEEK LOW BREAKDOWN (close ${p:.2f} < prior {lookback}d low ${l:.2f})"
    return None


# ── Price-target rules ──────────────────────────────────────────

@rule("price_above")
def price_above(latest, prev, params):
    target = params.get("target_price")
    if target is None:
        return None
    target = float(target)
    p = latest.get("Close")
    pp = prev.get("Close")
    if p is None or pp is None:
        return None
    # Fire on the crossing day, not every day above
    if pp <= target and p > target:
        return f"PRICE CROSSED ABOVE ${target:.2f} (close ${p:.2f})"
    return None


@rule("price_below")
def price_below(latest, prev, params):
    target = params.get("target_price")
    if target is None:
        return None
    target = float(target)
    p = latest.get("Close")
    pp = prev.get("Close")
    if p is None or pp is None:
        return None
    if pp >= target and p < target:
        return f"PRICE CROSSED BELOW ${target:.2f} (close ${p:.2f})"
    return None


def evaluate(rule_type, latest, prev, params=None):
    """Run one rule. Returns alert message if fired, else None."""
    fn = REGISTRY.get(rule_type)
    if fn is None:
        return None
    return fn(latest, prev, params or {})


def available_rules():
    return sorted(REGISTRY.keys())
