"""
HedgeAbove command-line interface.

Usage:
    python -m hedgeabove.cli init
    python -m hedgeabove.cli watchlist list
    python -m hedgeabove.cli watchlist add <name>
    python -m hedgeabove.cli watchlist add-ticker <name> <SYMBOL> [<SYMBOL> ...]
    python -m hedgeabove.cli watchlist remove-ticker <name> <SYMBOL>
    python -m hedgeabove.cli watchlist export <file>
    python -m hedgeabove.cli watchlist import <file> [--mode merge|replace]
    python -m hedgeabove.cli rule list
    python -m hedgeabove.cli rule add <group_name> <rule_type> [--param k=v ...]
    python -m hedgeabove.cli rule disable <rule_id>
    python -m hedgeabove.cli rule enable <rule_id>
    python -m hedgeabove.cli rule delete <rule_id>
    python -m hedgeabove.cli scan-once [--group <name>] [--ticker <SYMBOL>]
    python -m hedgeabove.cli rules-available
    python -m hedgeabove.cli snooze add <SYMBOL> --days N [--reason "text"]
    python -m hedgeabove.cli snooze remove <SYMBOL>
    python -m hedgeabove.cli snooze list
    python -m hedgeabove.cli analyze <SYMBOL> <rule_type> [--period 5y] [--param k=v ...] [--show-fires]
"""
import argparse
import json
import sys
from datetime import datetime, timedelta

from hedgeabove import db
from hedgeabove.rules import technical as tech_rules
from hedgeabove.rules import fundamental as fund_rules


def _all_rule_types():
    return sorted(set(tech_rules.REGISTRY) | set(fund_rules.REGISTRY))


def _parse_param(s):
    """Parse a 'k=v' string. Tries int, float, then leaves as string."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--param expects k=v, got '{s}'")
    k, v = s.split("=", 1)
    for cast in (int, float):
        try:
            return k, cast(v)
        except ValueError:
            continue
    return k, v


_DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "BTC-USD", "ETH-USD"]
_DEFAULT_RULES = [
    "rsi_oversold", "rsi_overbought",
    "macd_bullish_cross", "macd_bearish_cross",
    "golden_cross", "death_cross",
]


def _resolve_group_or_die(name):
    g = db.get_watchlist_group_by_name(name)
    if not g:
        print(f"No such watchlist group: {name}", file=sys.stderr)
        sys.exit(1)
    return g[0]


def cmd_init(args):
    db.init_db()
    if db.get_watchlist_group_by_name("default"):
        print("Default watchlist already exists. Nothing to do.")
        return
    gid = db.create_watchlist_group("default")
    for sym in _DEFAULT_TICKERS:
        db.add_ticker_to_group(gid, sym)
    for rt in _DEFAULT_RULES:
        db.add_alert_rule(gid, rt)
    print(f"Seeded 'default' watchlist with {len(_DEFAULT_TICKERS)} tickers "
          f"and {len(_DEFAULT_RULES)} rules.")


def cmd_watchlist(args):
    db.init_db()
    if args.action == "list":
        groups = db.list_watchlist_groups()
        if not groups:
            print("(no watchlist groups — run `cli init`)")
            return
        for gid, name in groups:
            tickers = db.get_watchlist_group_tickers(gid)
            print(f"  [{gid}] {name}: {', '.join(tickers) or '(empty)'}")
    elif args.action == "add":
        gid = db.create_watchlist_group(args.name)
        print(f"Created watchlist group '{args.name}' (id={gid})")
    elif args.action == "add-ticker":
        gid = _resolve_group_or_die(args.name)
        for sym in args.symbols:
            db.add_ticker_to_group(gid, sym.upper())
        print(f"Added {len(args.symbols)} ticker(s) to '{args.name}'")
    elif args.action == "remove-ticker":
        gid = _resolve_group_or_die(args.name)
        db.remove_ticker_from_group(gid, args.symbol.upper())
        print(f"Removed {args.symbol.upper()} from '{args.name}'")
    elif args.action == "export":
        groups = db.list_watchlist_groups()
        payload = {
            "version": 1,
            "exported_at": datetime.utcnow().isoformat(),
            "groups": [
                {
                    "name": name,
                    "tickers": db.get_watchlist_group_tickers(gid),
                    "rules": [
                        {
                            "rule_type": rt,
                            "params": json.loads(params) if params else {},
                            "enabled": bool(enabled),
                        }
                        for rid, rt, params, enabled in db.list_alert_rules(gid, enabled_only=False)
                    ],
                }
                for gid, name in groups
            ],
        }
        with open(args.file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        n_rules = sum(len(g["rules"]) for g in payload["groups"])
        n_tickers = sum(len(g["tickers"]) for g in payload["groups"])
        print(f"Exported {len(payload['groups'])} group(s), {n_tickers} ticker(s), "
              f"{n_rules} rule(s) -> {args.file}")
    elif args.action == "import":
        with open(args.file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != 1:
            print(f"Unsupported export version: {payload.get('version')}", file=sys.stderr)
            sys.exit(1)
        if args.mode == "replace":
            for gid, _ in db.list_watchlist_groups():
                db.delete_watchlist_group(gid)
        n_groups = n_tickers = n_rules = 0
        for g in payload["groups"]:
            existing = db.get_watchlist_group_by_name(g["name"])
            if existing:
                gid = existing[0]
            else:
                gid = db.create_watchlist_group(g["name"])
                n_groups += 1
            for sym in g.get("tickers", []):
                db.add_ticker_to_group(gid, sym.upper())
                n_tickers += 1
            existing_types = {row[1] for row in db.list_alert_rules(gid, enabled_only=False)}
            for r in g.get("rules", []):
                if r["rule_type"] in existing_types:
                    continue
                params_str = json.dumps(r.get("params", {}))
                rid = db.add_alert_rule(gid, r["rule_type"], params_str)
                if not r.get("enabled", True):
                    db.set_alert_rule_enabled(rid, False)
                n_rules += 1
        print(f"Imported: +{n_groups} group(s), +{n_tickers} ticker(s), +{n_rules} new rule(s) "
              f"(mode={args.mode})")


def cmd_rule(args):
    db.init_db()
    if args.action == "list":
        groups = db.list_watchlist_groups()
        for gid, name in groups:
            rules = db.list_alert_rules(gid, enabled_only=False)
            print(f"  Group '{name}':")
            if not rules:
                print("    (no rules)")
                continue
            for row in rules:
                rid, rt, params, enabled = row
                flag = "ON " if enabled else "OFF"
                params_str = "" if params == "{}" else f" {params}"
                print(f"    [{rid}] {flag} {rt}{params_str}")
    elif args.action == "add":
        if args.rule_type not in tech_rules.REGISTRY and args.rule_type not in fund_rules.REGISTRY:
            print(f"Unknown rule type: {args.rule_type}", file=sys.stderr)
            print(f"Available: {_all_rule_types()}", file=sys.stderr)
            sys.exit(1)
        gid = _resolve_group_or_die(args.group_name)
        params_dict = dict(args.param) if args.param else {}
        params_str = json.dumps(params_dict)
        rid = db.add_alert_rule(gid, args.rule_type, params_str)
        suffix = f" with params {params_dict}" if params_dict else ""
        print(f"Added rule '{args.rule_type}' to '{args.group_name}' (id={rid}){suffix}")
    elif args.action == "disable":
        db.set_alert_rule_enabled(args.rule_id, False)
        print(f"Disabled rule {args.rule_id}")
    elif args.action == "enable":
        db.set_alert_rule_enabled(args.rule_id, True)
        print(f"Enabled rule {args.rule_id}")
    elif args.action == "delete":
        db.delete_alert_rule(args.rule_id)
        print(f"Deleted rule {args.rule_id}")


def cmd_scan_once(args):
    from hedgeabove.scanner import run
    run(verbose=True, group=args.group, ticker=args.ticker)


def cmd_rules_available(args):
    print("Technical rules (need price history):")
    for rt in tech_rules.available_rules():
        doc = tech_rules.get_doc(rt)
        print(f"  - {rt:30s} {doc}")
    print()
    print("Fundamental rules (need ticker.info):")
    for rt in fund_rules.available_rules():
        doc = fund_rules.get_doc(rt)
        print(f"  - {rt:30s} {doc}")


def cmd_analyze(args):
    from hedgeabove.backtest.signals import summarize_rule, DEFAULT_HORIZONS
    if args.rule_type not in tech_rules.REGISTRY and args.rule_type not in fund_rules.REGISTRY:
        print(f"Unknown rule type: {args.rule_type}", file=sys.stderr)
        sys.exit(1)
    if args.rule_type in fund_rules.REGISTRY:
        if args.rule_type in {"analyst_upside_above", "dividend_yield_above"}:
            print(f"'{args.rule_type}' isn't yet backtestable (no point-in-time source for "
                  f"analyst targets / dividends).", file=sys.stderr)
            sys.exit(1)
    params_dict = dict(args.param) if args.param else {}
    sym = args.symbol.upper()
    print(f"\nAnalyzing {sym} :: {args.rule_type}  params={params_dict}  period={args.period}")
    summary, fires = summarize_rule(sym, args.rule_type, params_dict, args.period)
    print(f"  Total fires: {summary['n_fires']}")
    if summary["n_fires"] == 0:
        print("  Rule never fired in this window. Try a different threshold or longer period.")
        return
    print()
    print(f"  {'horizon':>10s}  {'fires':>6s}  {'hit_rate':>9s}  {'avg':>8s}  {'median':>8s}  {'std':>8s}  {'sharpe':>7s}")
    for h in DEFAULT_HORIZONS:
        n_w = summary[f"n_with_fwd_{h}d"]
        hr = summary[f"hit_rate_{h}d"]
        avg = summary[f"avg_return_{h}d"]
        med = summary[f"median_return_{h}d"]
        std = summary[f"std_return_{h}d"]
        sh = summary[f"sharpe_{h}d"]
        if hr is None:
            print(f"  {h:>9d}d  {n_w:>6d}  (insufficient forward data)")
            continue
        sh_str = f"{sh:>7.2f}" if sh is not None else "    -  "
        print(f"  {h:>9d}d  {n_w:>6d}  {hr:>8.1%}  {avg:>+7.2%}  {med:>+7.2%}  {std:>7.2%}  {sh_str}")
    if args.show_fires:
        print("\n  Last 10 fires:")
        for f in fires[-10:]:
            r5 = f"{f.fwd_returns.get(5)*100:+6.2f}%" if f.fwd_returns.get(5) is not None else "   -  "
            r20 = f"{f.fwd_returns.get(20)*100:+6.2f}%" if f.fwd_returns.get(20) is not None else "   -  "
            print(f"    {f.fire_date.date()}  ${f.price_at_fire:>8.2f}  "
                  f"5d={r5}  20d={r20}  {f.message}")


def cmd_snooze(args):
    db.init_db()
    if args.action == "add":
        until = (datetime.utcnow().date() + timedelta(days=args.days)).isoformat()
        db.snooze_ticker(args.symbol, until, args.reason or "")
        print(f"Snoozed {args.symbol.upper()} until {until} "
              f"(reason: {args.reason or 'unspecified'})")
    elif args.action == "remove":
        db.unsnooze_ticker(args.symbol)
        print(f"Removed snooze on {args.symbol.upper()}")
    elif args.action == "list":
        rows = db.list_snoozes(active_only=False)
        if not rows:
            print("(no snoozes)")
            return
        today = str(datetime.utcnow().date())
        for symbol, until, reason, created in rows:
            status = "ACTIVE" if until >= today else "EXPIRED"
            r = f"  ({reason})" if reason else ""
            print(f"  {symbol:8s}  until {until}  [{status}]{r}")


def _build_parser():
    p = argparse.ArgumentParser(prog="hedgeabove.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize DB and seed default watchlist")

    pw = sub.add_parser("watchlist", help="Manage watchlist groups")
    psw = pw.add_subparsers(dest="action", required=True)
    psw.add_parser("list")
    pw_add = psw.add_parser("add"); pw_add.add_argument("name")
    pw_addt = psw.add_parser("add-ticker")
    pw_addt.add_argument("name"); pw_addt.add_argument("symbols", nargs="+")
    pw_rm = psw.add_parser("remove-ticker")
    pw_rm.add_argument("name"); pw_rm.add_argument("symbol")
    pw_exp = psw.add_parser("export", help="Dump all groups, tickers, and rules to JSON")
    pw_exp.add_argument("file")
    pw_imp = psw.add_parser("import", help="Load groups/tickers/rules from a JSON file")
    pw_imp.add_argument("file")
    pw_imp.add_argument("--mode", choices=["merge", "replace"], default="merge",
                        help="merge: add to existing (default). replace: wipe all groups first.")

    pr = sub.add_parser("rule", help="Manage alert rules")
    psr = pr.add_subparsers(dest="action", required=True)
    psr.add_parser("list")
    pr_add = psr.add_parser("add", help="Attach a rule to a group, optionally with --param k=v")
    pr_add.add_argument("group_name"); pr_add.add_argument("rule_type")
    pr_add.add_argument("--param", type=_parse_param, action="append",
                        help="Override default rule params, e.g. --param threshold=25 (repeat for multiple)")
    pr_dis = psr.add_parser("disable"); pr_dis.add_argument("rule_id", type=int)
    pr_en = psr.add_parser("enable"); pr_en.add_argument("rule_id", type=int)
    pr_del = psr.add_parser("delete"); pr_del.add_argument("rule_id", type=int)

    sc = sub.add_parser("scan-once", help="Run a single scan now")
    sc.add_argument("--group", help="Limit scan to one watchlist group")
    sc.add_argument("--ticker", help="Limit scan to one symbol")
    sub.add_parser("rules-available", help="List registered rule types with descriptions")

    pa = sub.add_parser("analyze", help="Backtest a technical rule on a symbol")
    pa.add_argument("symbol")
    pa.add_argument("rule_type")
    pa.add_argument("--period", default="5y", help="yfinance period (e.g. 1y, 5y, max). Default: 5y")
    pa.add_argument("--param", type=_parse_param, action="append",
                    help="Override default rule params, e.g. --param threshold=25")
    pa.add_argument("--show-fires", action="store_true",
                    help="Also print the last 10 fire events with forward returns")

    ps = sub.add_parser("snooze", help="Mute alerts for a ticker")
    pss = ps.add_subparsers(dest="action", required=True)
    ps_add = pss.add_parser("add")
    ps_add.add_argument("symbol")
    ps_add.add_argument("--days", type=int, required=True,
                        help="Number of days to snooze (from today, inclusive)")
    ps_add.add_argument("--reason", default="", help="Optional human-readable reason")
    ps_rm = pss.add_parser("remove"); ps_rm.add_argument("symbol")
    pss.add_parser("list")
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    {
        "init": cmd_init,
        "watchlist": cmd_watchlist,
        "rule": cmd_rule,
        "scan-once": cmd_scan_once,
        "rules-available": cmd_rules_available,
        "snooze": cmd_snooze,
        "analyze": cmd_analyze,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
