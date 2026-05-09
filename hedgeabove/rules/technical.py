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


def evaluate(rule_type, latest, prev, params=None):
    """Run one rule. Returns alert message if fired, else None."""
    fn = REGISTRY.get(rule_type)
    if fn is None:
        return None
    return fn(latest, prev, params or {})


def available_rules():
    return sorted(REGISTRY.keys())
