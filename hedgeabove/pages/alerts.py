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
    st.subheader("Recent alert history")
    history = db.recent_alerts(50)
    if history:
        df = pd.DataFrame(history, columns=["Symbol", "Rule", "Fired At (UTC)", "Message"])
        df["Fired At (UTC)"] = df["Fired At (UTC)"].str[:19].str.replace("T", " ")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts fired yet. Tickers, rules, and a scan run will populate this.")
