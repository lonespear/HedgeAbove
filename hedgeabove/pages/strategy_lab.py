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
    mode = st.radio(
        "Rule mode",
        ["Single rule", "Composite (multi-rule AND/OR)"],
        horizontal=True, key="sl_strat_mode",
    )

    rule_type = None
    composite_rules: list = []
    combiner = "all"

    if mode == "Single rule":
        c1, c2, c3 = st.columns([2, 1, 1])
        rule_type = c1.selectbox("Rule", backtestable, key="sl_strat_rule",
                                 format_func=lambda rt: f"{rt}  ({_rule_kind(rt)})")
        period = c2.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2,
                              key="sl_strat_period")
        hold_days = c3.number_input("Hold (days)", min_value=1, max_value=120, value=20,
                                    key="sl_strat_hold")
        if rule_type:
            st.caption(_rule_doc(rule_type))
    else:
        c1, c2 = st.columns([1, 1])
        composite_rules = c1.multiselect(
            "Rules to combine", backtestable,
            default=backtestable[:2] if len(backtestable) >= 2 else backtestable,
            key="sl_strat_composite_rules",
            format_func=lambda rt: f"{rt}  ({_rule_kind(rt)})",
        )
        combiner = c2.selectbox(
            "Combiner", ["all", "any", "majority"],
            key="sl_strat_combiner",
            help="all=AND (every rule fires), any=OR (any rule fires), majority=>50% fire",
        )
        c3, c4 = st.columns([1, 1])
        period = c3.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2,
                              key="sl_strat_period")
        hold_days = c4.number_input("Hold (days)", min_value=1, max_value=120, value=20,
                                    key="sl_strat_hold")
        if composite_rules:
            st.caption("Rules in this composite:")
            for rt in composite_rules:
                st.caption(f"  - `{rt}` _({_rule_kind(rt)})_: {_rule_doc(rt)}")
        else:
            st.caption("Pick at least one rule above to enable backtest.")

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

    p1, p2 = st.columns([3, 1])
    if mode == "Single rule":
        params_text = p1.text_input("Optional params (JSON)", value="{}",
                                    key="sl_strat_params",
                                    placeholder='e.g. {"threshold": 25}')
    else:
        p1.caption("Composite mode uses each rule's default params. "
                   "For per-rule overrides, use the CLI: "
                   "`hedgeabove.cli strategy --rules ... --combine ...`")
        params_text = "{}"
    benchmark = p2.text_input("Benchmark", value="SPY", key="sl_strat_bench",
                              help="Buy-and-hold ticker for comparison. Empty = skip.")

    can_run = bool(symbols) and (
        (mode == "Single rule" and rule_type)
        or (mode != "Single rule" and len(composite_rules) >= 1)
    )

    if st.button("Run backtest", type="primary", key="sl_strat_run", disabled=not can_run):
        try:
            params = json.loads(params_text or "{}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        if mode == "Single rule":
            label = rule_type
            spinner_msg = f"Replaying {rule_type} on {len(symbols)} ticker(s)..."
            with st.spinner(spinner_msg):
                res = simulate_basket(symbols, rule_type, params,
                                      period=period, hold_days=int(hold_days),
                                      benchmark=benchmark.strip() or None)
        else:
            rules_list = [(rt, {}) for rt in composite_rules]
            label = f"[{combiner.upper()}: {','.join(composite_rules)}]"
            spinner_msg = (f"Replaying composite ({combiner.upper()}) of "
                           f"{len(composite_rules)} rule(s) on "
                           f"{len(symbols)} ticker(s)...")
            with st.spinner(spinner_msg):
                res = simulate_basket(symbols, rules=rules_list, combiner=combiner,
                                      period=period, hold_days=int(hold_days),
                                      benchmark=benchmark.strip() or None)

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

        # Benchmark comparison row
        bm = s.get("benchmark")
        if bm and s.get("benchmark_total_return") is not None:
            st.markdown(f"**vs benchmark `{bm}` buy-and-hold:**")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric(f"{bm} total", f"{s['benchmark_total_return']*100:+.1f}%")
            b2.metric(f"{bm} ann", f"{s['benchmark_ann_return']*100:+.1f}%")
            b3.metric(f"{bm} max DD", f"{s['benchmark_max_drawdown']*100:.1f}%")
            b4.metric("Alpha (total)", f"{s['alpha_total']*100:+.1f}%")
            b5.metric("Alpha (ann)", f"{s['alpha_ann']*100:+.1f}%")

        # Equity curve with benchmark overlay
        eq = res["equity_curve"]
        bench_eq = res.get("benchmark_curve", pd.Series(dtype=float))
        st.markdown("**Equity curve** (daily mark-to-market in trade, flat in cash):")
        if not bench_eq.empty:
            chart_df = pd.DataFrame({"strategy": eq, bm or "benchmark": bench_eq})
            st.line_chart(chart_df, use_container_width=True)
        else:
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

        # ── Tearsheet ──
        from hedgeabove.backtest.tearsheet import tearsheet
        ts = tearsheet(eq, bench_eq if not bench_eq.empty else None)

        st.markdown("---")
        st.markdown("### Tearsheet")

        # Calendar-year returns
        yr = ts["calendar_year_returns"]
        if not yr.empty:
            st.markdown("**Calendar-year returns:**")
            yr_df = pd.DataFrame({
                "Year": [d.year for d in yr.index],
                "Return %": (yr.values * 100).round(2),
            })
            st.dataframe(yr_df, use_container_width=False, hide_index=True,
                         column_config={"Return %": st.column_config.NumberColumn(format="%+.2f")})

        # Monthly returns heatmap (year x month grid)
        mr = ts["monthly_returns"]
        if not mr.empty:
            mr_df = pd.DataFrame({
                "year": [d.year for d in mr.index],
                "month": [d.month for d in mr.index],
                "ret_pct": mr.values * 100,
            })
            pivot = mr_df.pivot(index="year", columns="month", values="ret_pct")
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                           7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            pivot = pivot.rename(columns=month_names)
            st.markdown("**Monthly returns (%):**")
            try:
                styled = pivot.style.format("{:+.2f}", na_rep="—") \
                                    .background_gradient(cmap="RdYlGn", vmin=-5, vmax=5)
                st.dataframe(styled, use_container_width=True)
            except Exception:
                # Fallback if Styler unsupported in this Streamlit version
                st.dataframe(pivot.round(2), use_container_width=True)

        # Drawdown chart + summary
        dd = ts["drawdown_series"]
        if not dd.empty:
            st.markdown("**Drawdown (% below all-time peak NAV):**")
            st.area_chart(dd * 100, use_container_width=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("Max drawdown", f"{ts['max_drawdown']*100:.2f}%")
            d2.metric("Bottomed on",
                      str(ts['max_drawdown_date'].date())
                      if hasattr(ts['max_drawdown_date'], 'date')
                      else str(ts['max_drawdown_date']))
            d3.metric("Longest underwater", f"{ts['longest_drawdown_days']} days")

        # Rolling Sharpe chart + summary
        rs = ts["rolling_sharpe_252d"].dropna()
        if not rs.empty:
            st.markdown("**Rolling 252-day Sharpe:**")
            st.line_chart(rs, use_container_width=True)
            rs1, rs2, rs3, rs4 = st.columns(4)
            rs1.metric("Latest", f"{rs.iloc[-1]:.2f}")
            rs2.metric("Mean",   f"{rs.mean():.2f}")
            rs3.metric("Max",    f"{rs.max():.2f}")
            rs4.metric("Min",    f"{rs.min():.2f}")

        # Beta to benchmark
        if ts.get("beta") is not None and bm:
            st.metric(f"Beta to {bm}", f"{ts['beta']:.2f}",
                      help="OLS slope of strategy daily returns regressed on benchmark daily returns. "
                           "1.0 = moves with the market; 0 = uncorrelated; negative = inverse.")


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
