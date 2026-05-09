"""
Walk-forward parameter optimization.

The point: a single backtest with the best-looking threshold over the
full history is the canonical way to fool yourself. Walk-forward
splits history into K time-ordered folds; for each fold the param is
chosen on the *training* portion and then evaluated on the
*out-of-sample* portion. Aggregating the OOS evaluations gives a
realistic estimate of the rule's edge — and the in-sample-vs-OOS gap
is the overfit indicator.

Currently single-rule, single-numeric-param, technical rules only.
Fundamental rules require EDGAR and are slower; can be added later
if needed.
"""
import numpy as np
import pandas as pd
import yfinance as yf

from hedgeabove import config
from hedgeabove.indicators.technical import add_indicators, flatten_columns
from hedgeabove.rules import technical as tech_rules


def _fires_on_window(df, start_idx, end_idx, rule_type, params, horizon):
    """Evaluate rule across df[start_idx:end_idx], returning the list of
    (fire_bar_idx, fwd_return_at_horizon). Forward returns are looked up
    from the same df, so they may extend past end_idx by up to `horizon`
    bars (a fire at the last training day still has a real OOS return
    available in the data — but we don't use those here, the OOS scoring
    handles its own fires). Returns (idx, ret) pairs only when the
    forward bar exists.
    """
    fires = []
    n = len(df)
    lo = max(start_idx, 1)
    hi = min(end_idx, n)
    for i in range(lo, hi):
        latest = df.iloc[i]
        prev = df.iloc[i - 1]
        msg = tech_rules.evaluate(rule_type, latest, prev, params)
        if msg is None:
            continue
        j = i + horizon
        if j >= n:
            continue
        price = float(latest["Close"])
        fwd = float(df.iloc[j]["Close"]) / price - 1
        fires.append((i, fwd))
    return fires


def _score_fires(fires, score="sharpe"):
    """Aggregate (idx, ret) pairs into a single score. Returns -inf when
    there are no fires so optimizer ignores the candidate."""
    if not fires:
        return float("-inf")
    rets = np.array([r for _, r in fires], dtype=float)
    if score == "hit_rate":
        return float((rets > 0).mean())
    if score == "avg_return":
        return float(rets.mean())
    # default: per-fire Sharpe (mean / std). Stable for small samples.
    m = float(rets.mean())
    s = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    return (m / s) if s > 0 else 0.0


def walk_forward_optimize(symbol, rule_type, param_name, param_grid,
                          period="10y", horizon=20, n_folds=5,
                          score="sharpe"):
    """Walk-forward parameter selection.

    Args:
      symbol, rule_type: what to optimize
      param_name: rule param to vary (e.g. "threshold")
      param_grid: iterable of candidate values
      period: yfinance lookback (default 10y for enough folds)
      horizon: forward-return horizon used for both scoring and OOS
      n_folds: number of time-ordered folds; each fold uses its first
               half as in-sample and second half as OOS
      score: "sharpe" (default), "hit_rate", or "avg_return"

    Returns dict:
      fold_records: list of per-fold dicts
        (fold, train_start/end, test_start/end, best_param,
         train_scores, oos_n_fires, oos_score)
      walk_forward_summary: aggregate OOS stats across all folds
      n_folds, param_name, param_grid, horizon, score_metric
    None if there's not enough data.
    """
    if rule_type not in tech_rules.REGISTRY:
        raise ValueError(f"walk-forward currently supports technical rules only; "
                         f"got {rule_type!r}")

    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    df = flatten_columns(df)
    if df.empty or len(df) < config.MA_SLOW + 5:
        return None
    df = add_indicators(df,
                        rsi_period=config.RSI_PERIOD,
                        ma_fast=config.MA_FAST,
                        ma_slow=config.MA_SLOW)

    n = len(df)
    fold_size = n // n_folds
    if fold_size < 60:
        # Need enough room for both train and test halves to have ~30 bars min.
        return None

    fold_records = []
    all_oos_fires = []
    for fold_i in range(n_folds):
        fold_start = fold_i * fold_size
        fold_end = (fold_i + 1) * fold_size if fold_i < n_folds - 1 else n
        train_end = fold_start + (fold_end - fold_start) // 2

        train_scores = {}
        for pv in param_grid:
            params = {param_name: pv}
            fires_train = _fires_on_window(df, fold_start, train_end,
                                           rule_type, params, horizon)
            train_scores[pv] = _score_fires(fires_train, score)

        # If every train_score is -inf, no candidate fired — skip fold.
        finite = {pv: s for pv, s in train_scores.items() if s != float("-inf")}
        if not finite:
            best_pv = list(param_grid)[0]
            oos_fires = []
            oos_score = float("-inf")
        else:
            best_pv = max(finite, key=finite.get)
            oos_fires = _fires_on_window(df, train_end, fold_end,
                                         rule_type, {param_name: best_pv}, horizon)
            oos_score = _score_fires(oos_fires, score)

        fold_records.append({
            "fold": fold_i + 1,
            "train_start": df.index[fold_start].date()
                if fold_start < n else None,
            "train_end": df.index[train_end - 1].date()
                if train_end > 0 and train_end <= n else None,
            "test_start": df.index[train_end].date()
                if train_end < n else None,
            "test_end": df.index[fold_end - 1].date()
                if fold_end > 0 and fold_end <= n else None,
            "best_param": best_pv,
            "train_scores": train_scores,
            "oos_n_fires": len(oos_fires),
            "oos_score": oos_score,
        })
        all_oos_fires.extend(oos_fires)

    if all_oos_fires:
        oos_rets = np.array([r for _, r in all_oos_fires], dtype=float)
        s_arr = float(oos_rets.std(ddof=1)) if len(oos_rets) > 1 else 0.0
        if s_arr > 0:
            sharpe_per_fire = float(oos_rets.mean()) / s_arr
            # Annualize by sqrt of (252 / horizon)
            sharpe_ann = sharpe_per_fire * np.sqrt(252.0 / horizon)
        else:
            sharpe_ann = None
        wf_summary = {
            "n_oos_fires": int(len(oos_rets)),
            "oos_hit_rate": float((oos_rets > 0).mean()),
            "oos_avg_return": float(oos_rets.mean()),
            "oos_median_return": float(np.median(oos_rets)),
            "oos_sharpe_ann": sharpe_ann,
        }
    else:
        wf_summary = {"n_oos_fires": 0}

    return {
        "fold_records": fold_records,
        "walk_forward_summary": wf_summary,
        "n_folds": n_folds,
        "param_name": param_name,
        "param_grid": list(param_grid),
        "horizon": horizon,
        "score_metric": score,
    }
