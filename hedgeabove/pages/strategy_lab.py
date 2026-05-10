"""
Strategy Lab — cross-sectional scoring, strategy backtesting,
walk-forward optimization, and factor IC in one place.

Tabs:
  1. Cross-sectional Scorer — rank a universe by weighted Z-scored factors;
     optional sector-neutral mode; save top-N as a watchlist.
  2. Strategy Backtester    — single-rule or composite, single- or multi-
     position, equal or inverse-vol sizing, full tearsheet inline.
  3. Walk-Forward           — out-of-sample param optimization over K folds.
  4. Factor IC              — the canonical Spearman IC test for factor
     predictiveness with IR / t-stat / verdict.
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from hedgeabove import db
from hedgeabove.rules import technical as tech_rules
from hedgeabove.rules import fundamental as fund_rules


_UNBACKTESTABLE = {"analyst_upside_above", "dividend_yield_above"}


# ── Basket presets (UX: pick a curated list instead of typing tickers) ──
_BASKET_PRESETS = {
    "Index ETFs":          ["SPY", "QQQ", "IWM", "DIA"],
    "Tech megacaps":       ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL"],
    "Banks":               ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
    "Energy":              ["XOM", "CVX", "COP", "EOG", "PSX", "MPC", "VLO"],
    "Healthcare":          ["JNJ", "UNH", "LLY", "MRK", "ABBV", "PFE", "TMO", "ABT"],
    "Consumer staples":    ["WMT", "PG", "COST", "KO", "PEP", "PM", "CL", "MDLZ"],
    "Mega-cap diversified": ["AAPL", "MSFT", "JNJ", "JPM", "WMT", "XOM", "BRK-B", "UNH", "V"],
    "Crypto (USD)":        ["BTC-USD", "ETH-USD", "SOL-USD"],
}


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
        # Basket preset dropdown — quick-pick instead of typing tickers
        preset_choice = st.selectbox(
            "Basket preset",
            ["Custom"] + list(_BASKET_PRESETS.keys()),
            key="sl_score_basket_preset",
            help="Pick a curated basket or 'Custom' to type your own symbols",
        )
        default_syms = (", ".join(_BASKET_PRESETS[preset_choice])
                        if preset_choice != "Custom"
                        else "AAPL, MSFT, NVDA, GOOGL, META, AMZN, AVGO, ORCL, CRM, ADBE")
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value=default_syms,
            key=f"sl_score_symbols_{preset_choice}",  # forces refresh on preset change
        )

    # Advanced options
    with st.expander("Advanced (sector neutralization)"):
        ac1, ac2 = st.columns(2)
        sector_neutral = ac1.checkbox(
            "Sector-neutral",
            value=False, key="sl_score_secneu",
            help="Z-score factors within each sector instead of across the whole "
                 "universe. Stops high-growth sectors (tech) from dominating every score.",
        )
        by_sector = ac2.checkbox(
            "Group output by sector",
            value=False, key="sl_score_bysec",
            help="Show top-N picks per sector for diversified watchlist construction.",
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
        df = score_universe(
            symbols, weights, progress=progress,
            sector_neutral=sector_neutral,
            attach_sector=sector_neutral or by_sector,
        )
    progress_bar.empty()

    if df.empty:
        st.warning("No results — every ticker is missing at least one required factor.")
        return

    cols = list(weights.keys()) + ["composite_score"]
    if (sector_neutral or by_sector) and "sector" in df.columns:
        cols = ["sector"] + cols

    if by_sector and "sector" in df.columns:
        per = max(1, int(top_n) // max(1, df["sector"].nunique()))
        st.success(f"Scored {len(df)} eligible tickers; showing top {per} per sector.")
        for sec, sub in df.groupby("sector"):
            sub_top = sub.sort_values("composite_score", ascending=False).head(per)[cols]
            with st.expander(f"**{sec}** ({len(sub_top)} of {len(sub)})", expanded=True):
                st.dataframe(sub_top, use_container_width=True)
        # For watchlist save, take overall top-N
        top = df.head(int(top_n))[cols].copy()
    else:
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
        # Default to rsi_oversold so first-time "Run backtest" produces a
        # meaningful result; falls back to first available if not present.
        default_idx = backtestable.index("rsi_oversold") if "rsi_oversold" in backtestable else 0
        rule_type = c1.selectbox("Rule", backtestable, key="sl_strat_rule",
                                 index=default_idx,
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

    # Basket source: dropdown with watchlist + presets + custom
    groups = db.list_watchlist_groups()
    group_names = [n for _, n in groups]
    source_choices = (["Custom symbols"] +
                      [f"Watchlist: {n}" for n in group_names] +
                      [f"Preset: {name}" for name in _BASKET_PRESETS])
    source_choice = st.selectbox(
        "Basket source", source_choices,
        index=source_choices.index("Preset: Index ETFs") if "Preset: Index ETFs" in source_choices else 0,
        key="sl_strat_source_dd",
    )

    if source_choice.startswith("Watchlist: "):
        chosen = source_choice[len("Watchlist: "):]
        gid = next(g for g, n in groups if n == chosen)
        symbols = db.get_watchlist_group_tickers(gid)
        st.caption(f"Watchlist `{chosen}`: {', '.join(symbols) or '(empty)'}")
    elif source_choice.startswith("Preset: "):
        name = source_choice[len("Preset: "):]
        symbols = _BASKET_PRESETS[name]
        st.caption(f"Preset `{name}`: {', '.join(symbols)}")
    else:
        sym_input = st.text_input("Symbols (comma-separated)",
                                  value="SPY, QQQ, IWM",
                                  key="sl_strat_symbols_custom")
        symbols = [s.strip().upper() for s in sym_input.split(",") if s.strip()]

    # Sizing & risk options moved into an expander to declutter the form.
    with st.expander("Sizing & risk", expanded=False):
        sr1, sr2, sr3 = st.columns(3)
        max_concurrent = sr1.number_input(
            "Max concurrent positions",
            min_value=1, max_value=20, value=1, step=1,
            key="sl_strat_max_concurrent",
            help="1 = sequential single-position. Higher values diversify "
                 "and typically improve Sharpe at the cost of less compounding.",
        )
        sizing_choice = sr2.selectbox(
            "Position sizing", ["equal", "inverse_vol"],
            key="sl_strat_sizing",
            help="equal = 1/max_concurrent of NAV per position. "
                 "inverse_vol = scale by target_vol/realized_vol (clamped 0.5-2x).",
        )
        target_vol = sr3.number_input(
            "Target vol (annualized)", min_value=0.05, max_value=0.60,
            value=0.20, step=0.05, format="%.2f",
            key="sl_strat_target_vol",
            help="Used only when sizing = inverse_vol.",
        )
        sb1, sb2 = st.columns(2)
        benchmark = sb1.text_input(
            "Benchmark", value="SPY", key="sl_strat_bench",
            help="Buy-and-hold ticker for comparison. Empty = skip.",
        )
        ee1, ee2 = sb2.columns(2)
        excl_before = ee1.number_input(
            "Excl earnings days before", min_value=0, max_value=30, value=0,
            key="sl_strat_excl_before",
        )
        excl_after = ee2.number_input(
            "Excl earnings days after", min_value=0, max_value=30, value=0,
            key="sl_strat_excl_after",
        )
        if mode == "Single rule":
            params_text = st.text_input(
                "Optional rule params (JSON, single-rule mode only)",
                value="{}", key="sl_strat_params",
                placeholder='e.g. {"threshold": 25}',
            )
        else:
            st.caption("Composite mode uses each rule's default params. "
                       "For per-rule overrides, use the CLI.")
            params_text = "{}"
    excl_win = (
        (int(excl_before), int(excl_after))
        if (excl_before > 0 or excl_after > 0) else None
    )

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

        common_kwargs = dict(
            period=period, hold_days=int(hold_days),
            benchmark=benchmark.strip() or None,
            max_concurrent=int(max_concurrent),
            sizing=sizing_choice, target_vol=float(target_vol),
            exclude_earnings_window=excl_win,
        )
        if mode == "Single rule":
            label = rule_type
            spinner_msg = f"Replaying {rule_type} on {len(symbols)} ticker(s)..."
            with st.spinner(spinner_msg):
                res = simulate_basket(symbols, rule_type, params, **common_kwargs)
        else:
            rules_list = [(rt, {}) for rt in composite_rules]
            label = f"[{combiner.upper()}: {','.join(composite_rules)}]"
            spinner_msg = (f"Replaying composite ({combiner.upper()}) of "
                           f"{len(composite_rules)} rule(s) on "
                           f"{len(symbols)} ticker(s)...")
            with st.spinner(spinner_msg):
                res = simulate_basket(symbols, rules=rules_list, combiner=combiner,
                                      **common_kwargs)

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


def _render_walk_forward():
    from hedgeabove.backtest.walkforward import walk_forward_optimize

    st.subheader("Walk-forward parameter optimization")
    st.caption(
        "Splits history into K folds. For each fold, optimize the rule's "
        "param on the *training* half, evaluate on the *out-of-sample* half. "
        "Aggregating OOS performance gives an honest estimate of edge — "
        "and the in-sample-vs-OOS gap is the overfit indicator."
    )

    backtestable = sorted(tech_rules.REGISTRY.keys())  # technical-only for now
    c1, c2, c3 = st.columns([2, 1, 1])
    wf_default_idx = backtestable.index("rsi_oversold") if "rsi_oversold" in backtestable else 0
    rt = c1.selectbox("Rule (technical only)", backtestable,
                      index=wf_default_idx, key="sl_wf_rule")
    sym = c2.text_input("Symbol", value="SPY", key="sl_wf_sym")
    period = c3.selectbox("Period", ["5y", "10y", "max"], index=1, key="sl_wf_period")

    p1, p2, p3 = st.columns([2, 1, 1])
    param_name = p1.text_input("Param to optimize", value="threshold", key="sl_wf_param")
    horizon = p2.number_input("Horizon (days)", min_value=1, max_value=120,
                              value=20, key="sl_wf_horizon")
    folds = p3.number_input("Folds", min_value=2, max_value=10, value=5, key="sl_wf_folds")

    g1, g2 = st.columns([3, 1])
    grid_text = g1.text_input("Param grid (comma-separated values)",
                              value="20, 25, 30, 35, 40", key="sl_wf_grid")
    score = g2.selectbox("Score metric",
                         ["sharpe", "hit_rate", "avg_return"],
                         key="sl_wf_score")

    if rt:
        st.caption(f"`{rt}`: {tech_rules.get_doc(rt)}")

    if st.button("Run walk-forward", type="primary", key="sl_wf_run"):
        try:
            grid = []
            for x in grid_text.split(","):
                x = x.strip()
                if not x:
                    continue
                grid.append(int(x) if x.isdigit() else float(x))
        except ValueError as e:
            st.error(f"Invalid grid: {e}")
            return
        if len(grid) < 2:
            st.error("Need at least 2 grid values to optimize over.")
            return

        with st.spinner(f"Running {folds} folds..."):
            res = walk_forward_optimize(
                sym.upper(), rt, param_name, grid,
                period=period, horizon=int(horizon),
                n_folds=int(folds), score=score,
            )
        if not res:
            st.warning("Insufficient data (need ≥ 60 bars per fold).")
            return

        s = res["walk_forward_summary"]
        if s.get("n_oos_fires", 0) == 0:
            st.warning("No out-of-sample fires across any fold — rule too strict for this window.")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("OOS fires", s["n_oos_fires"])
        m2.metric("OOS hit rate", f"{s['oos_hit_rate']*100:.1f}%")
        m3.metric("OOS avg return", f"{s['oos_avg_return']*100:+.2f}%")
        sa = s.get("oos_sharpe_ann")
        m4.metric("OOS Sharpe (ann)", f"{sa:.2f}" if sa is not None else "—")

        st.markdown("**Fold-by-fold:**")
        rows = []
        for f in res["fold_records"]:
            oos_score = f["oos_score"]
            oos_str = "—" if oos_score == float("-inf") else f"{oos_score:+.3f}"
            rows.append({
                "Fold": f["fold"],
                "Train": f"{f['train_start']} → {f['train_end']}",
                "Test": f"{f['test_start']} → {f['test_end']}",
                "Best param": f["best_param"],
                "OOS score": oos_str,
                "OOS n": f["oos_n_fires"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Visual: the best param across folds (instability indicator)
        best_params = [f["best_param"] for f in res["fold_records"]]
        unique_params = len(set(best_params))
        if unique_params > 1:
            st.caption(
                f"Param chosen across folds: {best_params}. "
                f"{unique_params} different values — instability suggests the "
                f"rule's optimal threshold isn't stable; treat OOS Sharpe as the honest estimate."
            )


def _render_factor_ic():
    from hedgeabove.scoring.composite import FACTORS
    from hedgeabove.scoring.ic import factor_ic, factor_ic_summary

    st.subheader("Factor IC (Information Coefficient)")
    st.caption(
        "The canonical factor-predictiveness test: at each rebalance date, "
        "Spearman rank-correlate factor values across the cross-section vs "
        "forward returns. Mean IC > 0.05 = real factor. IR > 0.5 = actionable."
    )

    factor_choices = sorted(FACTORS.keys())
    c1, c2, c3 = st.columns([2, 1, 1])
    # Default to momentum_12_1 — a well-known factor that produces a
    # tangible IC quickly on the default basket.
    ic_default_idx = factor_choices.index("momentum_12_1") if "momentum_12_1" in factor_choices else 0
    factor = c1.selectbox("Factor", factor_choices, index=ic_default_idx,
                          key="sl_ic_factor")
    horizon = c2.number_input("Horizon (days)", min_value=5, max_value=63,
                              value=21, key="sl_ic_horizon")
    rebal = c3.selectbox("Rebalance", ["ME", "W", "QE"], key="sl_ic_rebal",
                         help="ME=monthly, W=weekly, QE=quarterly")

    u1, u2 = st.columns([1, 3])
    universe_mode = u1.radio("Universe", ["Preset", "Custom"],
                              horizontal=True, key="sl_ic_universe_mode")
    if universe_mode == "Preset":
        preset = u2.selectbox("Basket", list(_BASKET_PRESETS.keys()),
                              key="sl_ic_preset",
                              index=list(_BASKET_PRESETS.keys()).index("Tech megacaps"))
        symbols = _BASKET_PRESETS[preset]
        st.caption(f"`{preset}`: {', '.join(symbols)}")
    else:
        sym_input = u2.text_input(
            "Symbols (comma-separated)",
            value="AAPL, MSFT, NVDA, GOOGL, META, AMZN, JPM, BAC, XOM, CVX, WMT, KO",
            key="sl_ic_symbols",
        )
        symbols = [s.strip().upper() for s in sym_input.split(",") if s.strip()]

    period = st.selectbox("Period", ["2y", "5y", "10y", "max"], index=1, key="sl_ic_period")

    if st.button("Run IC analysis", type="primary", key="sl_ic_run",
                 disabled=not symbols):
        with st.spinner(f"Computing IC across {len(symbols)} tickers..."):
            ic_df = factor_ic(factor, symbols, period=period,
                              horizon=int(horizon), rebalance_freq=rebal)
        if ic_df.empty:
            st.warning("No IC data — insufficient cross-section coverage at any rebalance date.")
            return
        ppy = {"ME": 12, "W": 52, "QE": 4}[rebal]
        s = factor_ic_summary(ic_df, periods_per_year=ppy)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean IC", f"{s['mean_ic']:+.4f}")
        m2.metric("IR (ann)", f"{s.get('ir_annualized'):.2f}"
                  if s.get("ir_annualized") is not None else "—")
        m3.metric("t-stat", f"{s.get('t_stat'):.2f}"
                  if s.get("t_stat") is not None else "—")
        m4.metric("% positive", f"{s['pct_positive']*100:.1f}%")

        # Verdict
        abs_ic = abs(s["mean_ic"])
        if abs_ic >= 0.05:
            verdict_msg = "**Real factor** (|mean IC| ≥ 0.05)"
            st.success(verdict_msg)
        elif abs_ic >= 0.03:
            verdict_msg = "**Weak signal** (0.03 ≤ |mean IC| < 0.05)"
            st.info(verdict_msg)
        else:
            verdict_msg = "**No edge** (|mean IC| < 0.03)"
            st.warning(verdict_msg)

        # IC time series chart
        ts = ic_df.set_index("rebalance_date")["ic"]
        st.markdown("**IC over time:**")
        st.line_chart(ts, use_container_width=True)

        with st.expander("Per-rebalance IC table"):
            st.dataframe(ic_df, use_container_width=True, hide_index=True,
                         column_config={
                             "ic": st.column_config.NumberColumn(format="%+.4f"),
                             "p_value": st.column_config.NumberColumn(format="%.4f"),
                         })


def render():
    db.init_db()
    st.header("Strategy Lab")
    st.caption(
        "Cross-sectional scoring, strategy backtesting, walk-forward optimization, "
        "and factor IC analysis — all sharing the same SQLite DB the scanner uses."
    )

    tabs = st.tabs([
        "Cross-sectional Scorer",
        "Strategy Backtester",
        "Walk-Forward",
        "Factor IC",
    ])
    with tabs[0]:
        _render_scorer()
    with tabs[1]:
        _render_strategy_backtester()
    with tabs[2]:
        _render_walk_forward()
    with tabs[3]:
        _render_factor_ic()
