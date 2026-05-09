"""
Strategy Lab — cross-sectional scoring + strategy backtesting in one place.

Surfaces the same engines used by `cli score` and `cli strategy` in a
dashboard form: pick a preset, score a universe, optionally save the
top-N as a watchlist; or pick a rule + basket and replay it as a
single-position trader to get an equity curve + Sharpe + max DD.
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from hedgeabove import db
from hedgeabove.rules import technical as tech_rules
from hedgeabove.rules import fundamental as fund_rules


_UNBACKTESTABLE = {"analyst_upside_above", "dividend_yield_above"}


def _all_backtestable_rules():
    return sorted(
        (set(tech_rules.REGISTRY) | set(fund_rules.REGISTRY)) - _UNBACKTESTABLE
    )


def _rule_kind(rt):
    return "technical" if rt in tech_rules.REGISTRY else "fundamental"


def _rule_doc(rt):
    if rt in tech_rules.REGISTRY:
        return tech_rules.get_doc(rt)
    if rt in fund_rules.REGISTRY:
        return fund_rules.get_doc(rt)
    return ""


# ── Cross-sectional Scorer section ──────────────────────────────

def _render_scorer():
    from hedgeabove.scoring.composite import (
        score_universe, score_with_preset, PRESETS, FACTORS,
    )

    st.subheader("Cross-sectional scorer")
    st.caption(
        "Z-score every ticker in a universe on a weighted set of factors, "
        "rank by composite score, optionally save the top-N as a watchlist "
        "the scanner will alert on."
    )

    c1, c2 = st.columns([1, 1])
    mode = c1.radio("Weighting", ["Preset", "Custom weights (JSON)"],
                    horizontal=True, key="sl_score_mode")
    universe_mode = c2.radio("Universe", ["S&P 500 (~500 tickers)", "Custom symbols"],
                             horizontal=True, key="sl_score_univ_mode")

    if mode == "Preset":
        preset = st.selectbox("Preset", list(PRESETS.keys()),
                              index=0, key="sl_score_preset")
        weights = PRESETS[preset]
        st.caption(f"Weights: `{weights}`  (negative = lower-is-better)")
    else:
        default = json.dumps({"roe": 0.4, "pe": -0.4, "revenue_growth": 0.2}, indent=2)
        weights_text = st.text_area("Weights (JSON dict)", value=default, height=80,
                                    key="sl_score_weights")
        try:
            weights = json.loads(weights_text)
            unknown = set(weights) - set(FACTORS)
            if unknown:
                st.error(f"Unknown factors: {sorted(unknown)}. "
                         f"Available: {sorted(FACTORS)}")
                weights = None
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            weights = None

    if universe_mode.startswith("S&P 500"):
        symbols_input = None
    else:
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="AAPL, MSFT, NVDA, GOOGL, META, AMZN, AVGO, ORCL, CRM, ADBE",
            key="sl_score_symbols",
        )

    cc1, cc2, cc3 = st.columns([1, 1, 2])
    top_n = cc1.number_input("Top N", min_value=5, max_value=200, value=20, step=5,
                             key="sl_score_topn")
    save_as = cc2.text_input("Save as watchlist (optional)",
                             placeholder="e.g. quality_top20",
                             key="sl_score_save_as")
    run = cc3.button("Run scoring", type="primary", key="sl_score_run",
                     disabled=(weights is None))

    if not run or weights is None:
        return

    if universe_mode.startswith("S&P 500"):
        from hedgeabove.data.universe import get_sp500_tickers
        symbols = get_sp500_tickers()
    else:
        symbols = [s.strip().upper() for s in (symbols_input or "").split(",") if s.strip()]
        if not symbols:
            st.error("Need at least one symbol.")
            return

    progress_bar = st.progress(0.0, text=f"Fetching 0/{len(symbols)}...")
    last_ticker = {"v": ""}

    def progress(i, n, sym):
        last_ticker["v"] = sym
        progress_bar.progress(min(i / max(n, 1), 1.0),
                              text=f"Fetching {i}/{n}: {sym}")

    with st.spinner("Scoring..."):
        df = score_universe(symbols, weights, progress=progress)
    progress_bar.empty()

    if df.empty:
        st.warning("No results — every ticker is missing at least one required factor.")
        return

    cols = list(weights.keys()) + ["composite_score"]
    top = df.head(int(top_n))[cols].copy()
    st.success(f"Scored {len(df)} eligible tickers; showing top {len(top)}.")
    st.dataframe(top, use_container_width=True)

    if save_as.strip():
        existing = db.get_watchlist_group_by_name(save_as.strip())
        if existing:
            gid = existing[0]
            for sym in top.index:
                db.add_ticker_to_group(gid, sym)
            st.info(f"Appended {len(top)} ticker(s) to existing watchlist '{save_as.strip()}'.")
        else:
            gid = db.create_watchlist_group(save_as.strip())
            for sym in top.index:
                db.add_ticker_to_group(gid, sym)
            st.success(f"Saved as new watchlist '{save_as.strip()}' "
                       f"({len(top)} tickers). Scanner will alert on these next run.")


# ── Strategy Backtester section ─────────────────────────────────

def _render_strategy_backtester():
    from hedgeabove.backtest.strategy import simulate_basket, trades_to_dataframe

    st.subheader("Strategy backtester")
    st.caption(
        "Single-position long-only walk through a rule's fires across a basket. "
        "Enter the next fire when flat, hold N trading days, exit, repeat. "
        "Cash earns 0%, no costs/slippage. Compare against buy-and-hold context "
        "via the Recent Alerts page or external benchmarks."
    )

    backtestable = _all_backtestable_rules()
    c1, c2, c3 = st.columns([2, 1, 1])
    rule_type = c1.selectbox("Rule", backtestable, key="sl_strat_rule",
                             format_func=lambda rt: f"{rt}  ({_rule_kind(rt)})")
    period = c2.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2,
                          key="sl_strat_period")
    hold_days = c3.number_input("Hold (days)", min_value=1, max_value=120, value=20,
                                key="sl_strat_hold")

    if rule_type:
        st.caption(_rule_doc(rule_type))

    s1, s2 = st.columns([1, 2])
    source = s1.radio("Basket source", ["Watchlist", "Custom symbols"],
                      horizontal=True, key="sl_strat_source")
    if source == "Watchlist":
        groups = db.list_watchlist_groups()
        if not groups:
            s2.info("No watchlists exist yet. Use Alerts page or `cli init` to create one.")
            return
        names = [n for _, n in groups]
        chosen = s2.selectbox("Watchlist", names, key="sl_strat_watchlist")
        gid = next(g for g, n in groups if n == chosen)
        symbols = db.get_watchlist_group_tickers(gid)
    else:
        sym_input = s2.text_input("Symbols (comma-separated)",
                                  value="SPY, QQQ, IWM",
                                  key="sl_strat_symbols")
        symbols = [s.strip().upper() for s in sym_input.split(",") if s.strip()]

    params_text = st.text_input("Optional params (JSON)", value="{}",
                                key="sl_strat_params",
                                placeholder='e.g. {"threshold": 25}')

    if st.button("Run backtest", type="primary", key="sl_strat_run", disabled=not symbols):
        try:
            params = json.loads(params_text or "{}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        with st.spinner(f"Replaying {rule_type} on {len(symbols)} ticker(s)..."):
            res = simulate_basket(symbols, rule_type, params,
                                  period=period, hold_days=int(hold_days))

        s = res["summary"]
        if not s.get("n_trades"):
            st.warning("No trades — rule never fired (or no future bars to close).")
            return

        # Headline metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total return", f"{s['total_return']*100:+.1f}%")
        m2.metric("Annualized", f"{s['ann_return']*100:+.1f}%")
        m3.metric("Sharpe (ann)",
                  f"{s['sharpe_ann']:.2f}" if s.get("sharpe_ann") is not None else "—")
        m4.metric("Max DD", f"{s['max_drawdown']*100:.1f}%")
        m5.metric("Trades", f"{s['n_trades']}")

        sub_cols = st.columns(4)
        sub_cols[0].metric("Win rate", f"{s['win_rate']*100:.1f}%")
        sub_cols[1].metric("Avg trade", f"{s['avg_trade_return']*100:+.2f}%")
        sub_cols[2].metric("Median trade", f"{s['median_trade_return']*100:+.2f}%")
        sub_cols[3].metric("Trades / yr", f"{s['trades_per_year']:.1f}")

        # Equity curve
        eq = res["equity_curve"]
        st.markdown("**Equity curve** (NAV at each trade exit, starts at 1.0):")
        st.line_chart(eq, use_container_width=True)

        # Trades table
        st.markdown("**Trades:**")
        trades_df = trades_to_dataframe(res["trades"])
        # Pre-scale return column for display
        trades_df["return %"] = trades_df["return_pct"] * 100
        trades_df = trades_df.drop(columns=["return_pct"])
        st.dataframe(trades_df, use_container_width=True, hide_index=True,
                     column_config={
                         "return %": st.column_config.NumberColumn(format="%+.2f"),
                         "entry_price": st.column_config.NumberColumn(format="$%.2f"),
                         "exit_price": st.column_config.NumberColumn(format="$%.2f"),
                     })


def render():
    db.init_db()
    st.header("Strategy Lab")
    st.caption(
        "Score a universe to generate a watchlist, then backtest a rule on it "
        "as a real strategy. Both engines share the same DB the scanner uses, "
        "so a watchlist created here is immediately monitored by `scan.py`."
    )

    tab_score, tab_strat = st.tabs(["Cross-sectional Scorer", "Strategy Backtester"])
    with tab_score:
        _render_scorer()
    with tab_strat:
        _render_strategy_backtester()
