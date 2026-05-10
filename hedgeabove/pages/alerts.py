"""
Alerts page — manage watchlist groups, rules, and view fire history.

Reads/writes the same SQLite tables as the headless scanner (scan.py / cron),
so configuration changes here take effect on the next scan run.
"""
import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from hedgeabove import config, db
from hedgeabove.rules import technical as tech_rules
from hedgeabove.rules import fundamental as fund_rules
from hedgeabove.alerts.telegram import send as send_telegram


def _all_rule_types():
    return sorted(set(tech_rules.REGISTRY) | set(fund_rules.REGISTRY))


def _rule_kind(rt):
    if rt in tech_rules.REGISTRY:
        return "technical"
    if rt in fund_rules.REGISTRY:
        return "fundamental"
    return "unknown"


def _rule_doc(rt):
    if rt in tech_rules.REGISTRY:
        return tech_rules.get_doc(rt)
    if rt in fund_rules.REGISTRY:
        return fund_rules.get_doc(rt)
    return ""


def _todays_alert_count():
    today = str(datetime.utcnow().date())
    return sum(1 for r in db.recent_alerts(500) if r[2][:10] == today)


def _render_top(groups):
    total_tickers = sum(len(db.get_watchlist_group_tickers(gid)) for gid, _ in groups)
    total_rules = sum(len(db.list_alert_rules(gid, enabled_only=True)) for gid, _ in groups)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Watchlists", len(groups))
    c2.metric("Tickers", total_tickers)
    c3.metric("Active Rules", total_rules)
    c4.metric("Alerts Today", _todays_alert_count())

    tele_ok = bool(config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID)
    if tele_ok:
        st.success("Telegram is configured. Alerts from `Run scan now` will be delivered.")
    else:
        st.warning(
            "Telegram not configured. Alerts will print to the server log instead. "
            "Set `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` to enable delivery."
        )

    a, b = st.columns(2)
    if a.button("Run scan now", type="primary", use_container_width=True,
                disabled=not groups):
        from hedgeabove.scanner import run
        with st.spinner("Scanning watchlists…"):
            n = run(verbose=False)
        st.success(f"Scan complete — {n} alert(s) fired.")
        st.rerun()

    if b.button("Send test Telegram message", use_container_width=True, disabled=not tele_ok):
        ok = send_telegram(
            f"HedgeAbove test message — {datetime.now():%Y-%m-%d %H:%M:%S}"
        )
        if ok:
            st.success("Test message sent.")
        else:
            st.error("Test failed — check Telegram token and chat ID.")


def _render_group(gid, name):
    tickers = db.get_watchlist_group_tickers(gid)
    rules = db.list_alert_rules(gid, enabled_only=False)

    st.markdown(f"### `{name}`  ·  {len(tickers)} ticker(s)  ·  "
                f"{sum(1 for r in rules if r[3])} active rule(s)")

    # ── Tickers ──
    st.markdown("**Tickers**")
    if tickers:
        cols = st.columns(8)
        for i, t in enumerate(tickers):
            with cols[i % 8]:
                if st.button(f"× {t}", key=f"rm_{gid}_{t}", help=f"Remove {t}"):
                    db.remove_ticker_from_group(gid, t)
                    st.rerun()
    else:
        st.caption("(empty)")

    add_cols = st.columns([3, 1])
    new_ticks = add_cols[0].text_input(
        "Add tickers (comma-separated)", key=f"add_tick_{gid}",
        placeholder="e.g. MSFT, AMZN, GOOG",
        label_visibility="collapsed",
    )
    if add_cols[1].button("Add tickers", key=f"add_tick_btn_{gid}", use_container_width=True):
        symbols = [s.strip().upper() for s in new_ticks.split(",") if s.strip()]
        for sym in symbols:
            db.add_ticker_to_group(gid, sym)
        if symbols:
            st.rerun()

    # ── Rules ──
    st.markdown("**Rules**")
    if rules:
        for rid, rt, params, enabled in rules:
            r1, r2, r3, r4 = st.columns([4, 1, 1, 1])
            badge = " · `" + params + "`" if params and params != "{}" else ""
            kind = _rule_kind(rt)
            r1.write(f"**{rt}** _({kind})_{badge}")
            r2.write("✅ ON" if enabled else "⏸ OFF")
            tog_label = "Disable" if enabled else "Enable"
            if r3.button(tog_label, key=f"tog_{rid}"):
                db.set_alert_rule_enabled(rid, not enabled)
                st.rerun()
            if r4.button("Delete", key=f"del_rule_{rid}"):
                db.delete_alert_rule(rid)
                st.rerun()
            with st.expander(f"Edit params · {rt}", expanded=False):
                doc = _rule_doc(rt)
                if doc:
                    st.caption(doc)
                edited = st.text_area(
                    "Params (JSON)", value=params or "{}",
                    key=f"params_edit_{rid}", height=80,
                )
                if st.button("Save params", key=f"params_save_{rid}"):
                    try:
                        parsed = json.loads(edited)
                        if not isinstance(parsed, dict):
                            raise ValueError("Params must be a JSON object.")
                        db.set_alert_rule_params(rid, json.dumps(parsed))
                        st.success("Params saved.")
                        st.rerun()
                    except (ValueError, json.JSONDecodeError) as e:
                        st.error(f"Invalid JSON: {e}")
    else:
        st.caption("(no rules — add one below)")

    existing_types = {row[1] for row in rules}
    available = [r for r in _all_rule_types() if r not in existing_types]
    if available:
        new_rule = st.selectbox(
            "Add rule", available, key=f"new_rule_{gid}",
            format_func=lambda rt: f"{rt}  ({_rule_kind(rt)})",
        )
        doc = _rule_doc(new_rule)
        if doc:
            st.caption(doc)
        new_params = st.text_input(
            "Optional params (JSON)", value="{}",
            key=f"new_rule_params_{gid}",
            placeholder='e.g. {"threshold": 25} or {"target_price": 300}',
        )
        if st.button("Add rule", key=f"add_rule_btn_{gid}"):
            try:
                parsed = json.loads(new_params or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("Params must be a JSON object.")
                db.add_alert_rule(gid, new_rule, json.dumps(parsed))
                st.rerun()
            except (ValueError, json.JSONDecodeError) as e:
                st.error(f"Invalid JSON: {e}")
    else:
        st.caption("All registered rule types are already attached to this group.")

    with st.expander("Danger zone"):
        confirm = st.text_input(
            f"Type the group name (`{name}`) to confirm deletion",
            key=f"del_grp_input_{gid}",
        )
        if st.button("Delete this watchlist group", key=f"del_grp_btn_{gid}"):
            if confirm == name:
                db.delete_watchlist_group(gid)
                st.success(f"Deleted '{name}'.")
                st.rerun()
            else:
                st.error("Confirmation text didn't match — group not deleted.")


def render():
    db.init_db()

    st.header("Alerts & Watchlists")
    st.caption(
        "Configure technical-signal alerts. The same SQLite DB drives the "
        "headless scanner (`python scan.py` / cron), so changes here apply "
        "to all deployments."
    )

    groups = db.list_watchlist_groups()
    _render_top(groups)

    st.markdown("---")
    st.subheader("Watchlist groups")

    if not groups:
        st.info(
            "No watchlists yet. Create one below, then attach tickers and rules. "
            "Or run `python -m hedgeabove.cli init` from a terminal to seed a default."
        )
    else:
        names = [n for _, n in groups]
        selected = st.selectbox("Select group", names, key="alerts_selected_group")
        gid = next(g for g, n in groups if n == selected)
        _render_group(gid, selected)

    with st.expander("Create new watchlist group"):
        c1, c2 = st.columns([3, 1])
        new_name = c1.text_input("Group name", key="new_group_name",
                                 label_visibility="collapsed",
                                 placeholder="e.g. tech-megacaps")
        if c2.button("Create group", use_container_width=True):
            if new_name.strip():
                if db.get_watchlist_group_by_name(new_name.strip()):
                    st.error(f"Group '{new_name}' already exists.")
                else:
                    db.create_watchlist_group(new_name.strip())
                    st.rerun()
            else:
                st.error("Name cannot be empty.")

    st.markdown("---")
    st.subheader("Snoozed tickers")
    st.caption("Snoozed tickers are skipped by the scanner until their snooze expires. "
               "Useful for muting noisy signals without removing the ticker from a watchlist.")
    snoozes = db.list_snoozes(active_only=False)
    if snoozes:
        today = str(datetime.utcnow().date())
        for symbol, until, reason, created in snoozes:
            sc1, sc2, sc3, sc4 = st.columns([1, 2, 3, 1])
            sc1.write(f"**{symbol}**")
            status = "ACTIVE" if until >= today else "EXPIRED"
            sc2.write(f"until {until} _({status})_")
            sc3.write(reason or "_(no reason)_")
            if sc4.button("Remove", key=f"unsnooze_{symbol}"):
                db.unsnooze_ticker(symbol)
                st.rerun()
    else:
        st.caption("(no snoozes)")
    with st.expander("Add snooze"):
        sn1, sn2, sn3, sn4 = st.columns([2, 1, 3, 1])
        new_sym = sn1.text_input("Symbol", key="snooze_sym",
                                 placeholder="AAPL",
                                 label_visibility="collapsed")
        days = sn2.number_input("Days", min_value=1, max_value=365, value=7,
                                key="snooze_days", label_visibility="collapsed")
        reason = sn3.text_input("Reason (optional)", key="snooze_reason",
                                placeholder="e.g. earnings noise",
                                label_visibility="collapsed")
        if sn4.button("Snooze", key="snooze_add", use_container_width=True):
            if new_sym.strip():
                until = (datetime.utcnow().date() + timedelta(days=int(days))).isoformat()
                db.snooze_ticker(new_sym.strip(), until, reason.strip())
                st.rerun()
            else:
                st.error("Symbol cannot be empty.")

    st.markdown("---")
    st.subheader("Rule analytics — does this signal actually work?")
    st.caption(
        "Replays a rule over historical bars and measures forward returns at "
        "5/10/20 day horizons. Fundamental rules use SEC EDGAR for point-in-time "
        "data (filing-date filtered, amendments deduped); `analyst_upside_above` "
        "and `dividend_yield_above` aren't yet backtestable."
    )
    _UNBACKTESTABLE = {"analyst_upside_above", "dividend_yield_above"}
    backtestable = [r for r in _all_rule_types() if r not in _UNBACKTESTABLE]

    ana_mode = st.radio(
        "Rule mode",
        ["Single rule", "Composite (multi-rule AND/OR)"],
        horizontal=True, key="ana_mode",
    )

    ana_rule = None
    ana_composite_rules: list = []
    ana_combiner = "all"

    if ana_mode == "Single rule":
        ra1, ra2, ra3 = st.columns([2, 1, 1])
        ana_rule = ra1.selectbox("Rule", backtestable, key="ana_rule",
                                 format_func=lambda rt: f"{rt}  ({_rule_kind(rt)})")
        ana_symbol = ra2.text_input("Symbol", value="SPY", key="ana_sym")
        ana_period = ra3.selectbox("Period", ["1y", "2y", "5y", "10y", "max"],
                                   index=2, key="ana_period")
        ana_params = st.text_input("Optional params (JSON)", value="{}",
                                   key="ana_params",
                                   placeholder='e.g. {"threshold": 25}')
        if ana_rule:
            st.caption(_rule_doc(ana_rule))
    else:
        ra1, ra2 = st.columns([2, 1])
        ana_composite_rules = ra1.multiselect(
            "Rules", backtestable,
            default=backtestable[:2] if len(backtestable) >= 2 else backtestable,
            key="ana_composite_rules",
            format_func=lambda rt: f"{rt}  ({_rule_kind(rt)})",
        )
        ana_combiner = ra2.selectbox(
            "Combiner", ["all", "any", "majority"],
            key="ana_combiner",
            help="all=AND, any=OR, majority=>50%",
        )
        ra3, ra4 = st.columns([1, 1])
        ana_symbol = ra3.text_input("Symbol", value="SPY", key="ana_sym_c")
        ana_period = ra4.selectbox("Period", ["1y", "2y", "5y", "10y", "max"],
                                   index=2, key="ana_period_c")
        ana_params = "{}"
        if ana_composite_rules:
            for rt in ana_composite_rules:
                st.caption(f"  - `{rt}` _({_rule_kind(rt)})_: {_rule_doc(rt)}")

    with st.expander("Advanced filters", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        ana_by_regime = adv1.selectbox(
            "Macro regime breakdown",
            ["(none)", "vix", "yield_curve"],
            key="ana_by_regime",
            help="Split forward returns by FRED VIX or 2s10s slope buckets",
        )
        ana_by_year = adv2.checkbox(
            "Per-year breakdown + decay check", value=False, key="ana_by_year",
            help="Show fires + hit rates by calendar year, with z-score vs prior years",
        )
        eb1, eb2 = adv3.columns(2)
        ana_excl_before = eb1.number_input(
            "Excl earnings days before", min_value=0, max_value=30, value=0,
            key="ana_excl_before",
        )
        ana_excl_after = eb2.number_input(
            "Excl earnings days after", min_value=0, max_value=30, value=0,
            key="ana_excl_after",
        )
    excl_win = (
        (int(ana_excl_before), int(ana_excl_after))
        if (ana_excl_before > 0 or ana_excl_after > 0) else None
    )

    can_run = bool(ana_symbol.strip()) and (
        (ana_mode == "Single rule" and ana_rule)
        or (ana_mode != "Single rule" and len(ana_composite_rules) >= 1)
    )

    if st.button("Run backtest", type="secondary", key="ana_run", disabled=not can_run):
        try:
            params_dict = json.loads(ana_params or "{}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            from hedgeabove.backtest.signals import (
                summarize_rule, summarize_composite, fires_to_dataframe,
                DEFAULT_HORIZONS,
            )
            sym = ana_symbol.upper().strip()
            if ana_mode == "Single rule":
                spinner_msg = f"Replaying {ana_rule} on {sym}..."
            else:
                spinner_msg = (f"Replaying composite ({ana_combiner.upper()}) of "
                               f"{len(ana_composite_rules)} rule(s) on {sym}...")
            with st.spinner(spinner_msg):
                if ana_mode == "Single rule":
                    summary, fires = summarize_rule(
                        sym, ana_rule, params_dict, ana_period,
                        exclude_earnings_window=excl_win,
                    )
                else:
                    rules_list = [(rt, {}) for rt in ana_composite_rules]
                    summary, fires = summarize_composite(
                        sym, rules_list, ana_combiner, ana_period,
                        exclude_earnings_window=excl_win,
                    )

            label = summary["rule_type"]
            st.write(f"**{summary['symbol']}** :: `{label}`  "
                     f"params={summary['params']}  period={summary['period']}"
                     + (f"  exclude_earnings={excl_win}" if excl_win else ""))
            if summary["n_fires"] == 0:
                st.warning("Rule never fired in this window. Try a different "
                           "threshold, longer period, or composite combiner.")
            else:
                metric_cols = st.columns(len(DEFAULT_HORIZONS) + 1)
                metric_cols[0].metric("Total fires", summary["n_fires"])
                for i, h in enumerate(DEFAULT_HORIZONS, start=1):
                    hr = summary[f"hit_rate_{h}d"]
                    avg = summary[f"avg_return_{h}d"]
                    if hr is None:
                        metric_cols[i].metric(f"{h}d hit rate", "—")
                    else:
                        metric_cols[i].metric(
                            f"{h}d hit rate",
                            f"{hr:.0%}",
                            delta=f"avg {avg:+.2%}",
                        )

                # Per-horizon detail table (pre-scaled to %)
                rows = []
                for h in DEFAULT_HORIZONS:
                    hr = summary[f"hit_rate_{h}d"]
                    avg = summary[f"avg_return_{h}d"]
                    med = summary[f"median_return_{h}d"]
                    std = summary[f"std_return_{h}d"]
                    rows.append({
                        "horizon": f"{h}d",
                        "fires_w_fwd": summary[f"n_with_fwd_{h}d"],
                        "hit_rate %": hr * 100 if hr is not None else None,
                        "avg %": avg * 100 if avg is not None else None,
                        "median %": med * 100 if med is not None else None,
                        "std %": std * 100 if std is not None else None,
                        "sharpe": summary[f"sharpe_{h}d"],
                    })
                stats_df = pd.DataFrame(rows)
                st.dataframe(stats_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "hit_rate %": st.column_config.NumberColumn(format="%.1f"),
                                 "avg %": st.column_config.NumberColumn(format="%+.2f"),
                                 "median %": st.column_config.NumberColumn(format="%+.2f"),
                                 "std %": st.column_config.NumberColumn(format="%.2f"),
                                 "sharpe": st.column_config.NumberColumn(format="%.2f"),
                             })

                # Recent fires
                st.markdown("**Last 10 fires:**")
                fires_df = fires_to_dataframe(fires).tail(10).reset_index(drop=True)
                for col in [c for c in fires_df.columns if c.startswith("fwd_")]:
                    fires_df[col] = fires_df[col] * 100
                fires_df = fires_df.rename(columns={
                    "fwd_5d": "5d %", "fwd_10d": "10d %", "fwd_20d": "20d %",
                })
                st.dataframe(fires_df, use_container_width=True, hide_index=True,
                             column_config={
                                 "5d %": st.column_config.NumberColumn(format="%+.2f"),
                                 "10d %": st.column_config.NumberColumn(format="%+.2f"),
                                 "20d %": st.column_config.NumberColumn(format="%+.2f"),
                                 "price": st.column_config.NumberColumn(format="$%.2f"),
                             })

                # Macro regime breakdown
                if ana_by_regime != "(none)":
                    from hedgeabove.backtest.regime import by_regime_summary
                    st.markdown(f"**Forward returns by `{ana_by_regime}` regime:**")
                    reg_tabs = st.tabs([f"{h}d" for h in DEFAULT_HORIZONS])
                    for tab, h in zip(reg_tabs, DEFAULT_HORIZONS):
                        with tab:
                            agg = by_regime_summary(fires, ana_by_regime, horizon=h)
                            if agg.empty:
                                st.caption("(no data for this horizon)")
                                continue
                            disp = agg.copy()
                            disp["hit_rate"] = disp["hit_rate"] * 100
                            disp["avg"] = disp["avg"] * 100
                            disp["median"] = disp["median"] * 100
                            disp["std"] = disp["std"] * 100
                            st.dataframe(disp, use_container_width=True, hide_index=True,
                                         column_config={
                                             "hit_rate": st.column_config.NumberColumn(format="%.1f%%", label="hit %"),
                                             "avg": st.column_config.NumberColumn(format="%+.2f%%"),
                                             "median": st.column_config.NumberColumn(format="%+.2f%%"),
                                             "std": st.column_config.NumberColumn(format="%.2f%%"),
                                         })

                # Per-year breakdown + decay check
                if ana_by_year:
                    import numpy as np
                    rows_y = [{"year": f.fire_date.year,
                               **{f"fwd_{h}d": f.fwd_returns.get(h) for h in DEFAULT_HORIZONS}}
                              for f in fires]
                    ydf = pd.DataFrame(rows_y)
                    if not ydf.empty:
                        st.markdown("**Per-year breakdown:**")
                        yr_tabs = st.tabs([f"{h}d" for h in DEFAULT_HORIZONS])
                        for tab, h in zip(yr_tabs, DEFAULT_HORIZONS):
                            with tab:
                                col = f"fwd_{h}d"
                                valid = ydf[ydf[col].notna()]
                                if valid.empty:
                                    st.caption("(no data)")
                                    continue
                                agg = valid.groupby("year")[col].agg(
                                    n="count",
                                    hit_rate=lambda x: float((x > 0).mean()) * 100,
                                    avg=lambda x: float(x.mean()) * 100,
                                    median=lambda x: float(x.median()) * 100,
                                ).reset_index()
                                st.dataframe(agg, use_container_width=True, hide_index=True,
                                             column_config={
                                                 "hit_rate": st.column_config.NumberColumn(format="%.1f%%", label="hit %"),
                                                 "avg": st.column_config.NumberColumn(format="%+.2f%%"),
                                                 "median": st.column_config.NumberColumn(format="%+.2f%%"),
                                             })
                                yearly_hr = dict(zip(agg["year"], agg["hit_rate"]))
                                years = sorted(yearly_hr.keys())
                                if len(years) >= 4:
                                    last_y = years[-1]
                                    prior = [yearly_hr[y] for y in years[:-1]]
                                    mean_p = float(np.mean(prior))
                                    std_p = float(np.std(prior, ddof=1)) if len(prior) > 1 else 0.0
                                    z = (yearly_hr[last_y] - mean_p) / std_p if std_p > 0 else 0.0
                                    tag = ""
                                    if z < -1.5:
                                        tag = " ⚠ possible alpha decay"
                                    elif z > 1.5:
                                        tag = " ↑ recent surge"
                                    st.caption(f"Decay check: {last_y} hit {yearly_hr[last_y]:.1f}% "
                                               f"vs prior mean {mean_p:.1f}% (z={z:+.2f}){tag}")

    st.markdown("---")
    st.subheader("Recent alert history")
    history = db.recent_alerts(50)
    if history:
        df = pd.DataFrame(history, columns=["Symbol", "Rule", "Fired At (UTC)", "Message"])
        df["Fired At (UTC)"] = df["Fired At (UTC)"].str[:19].str.replace("T", " ")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts fired yet. Tickers, rules, and a scan run will populate this.")
