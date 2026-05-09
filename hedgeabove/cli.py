"""
HedgeAbove command-line interface.

Usage:
    python -m hedgeabove.cli init
    python -m hedgeabove.cli watchlist list
    python -m hedgeabove.cli watchlist add <name>
    python -m hedgeabove.cli watchlist add-ticker <name> <SYMBOL> [<SYMBOL> ...]
    python -m hedgeabove.cli watchlist remove-ticker <name> <SYMBOL>
    python -m hedgeabove.cli rule list
    python -m hedgeabove.cli rule add <group_name> <rule_type>
    python -m hedgeabove.cli rule disable <rule_id>
    python -m hedgeabove.cli rule enable <rule_id>
    python -m hedgeabove.cli rule delete <rule_id>
    python -m hedgeabove.cli scan-once
    python -m hedgeabove.cli rules-available
"""
import argparse
import sys

from hedgeabove import db
from hedgeabove.rules import technical as tech_rules


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
        if args.rule_type not in tech_rules.REGISTRY:
            print(f"Unknown rule type: {args.rule_type}", file=sys.stderr)
            print(f"Available: {tech_rules.available_rules()}", file=sys.stderr)
            sys.exit(1)
        gid = _resolve_group_or_die(args.group_name)
        rid = db.add_alert_rule(gid, args.rule_type)
        print(f"Added rule '{args.rule_type}' to '{args.group_name}' (id={rid})")
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
    run(verbose=True)


def cmd_rules_available(args):
    print("Registered rule types:")
    for rt in tech_rules.available_rules():
        print(f"  - {rt}")


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

    pr = sub.add_parser("rule", help="Manage alert rules")
    psr = pr.add_subparsers(dest="action", required=True)
    psr.add_parser("list")
    pr_add = psr.add_parser("add")
    pr_add.add_argument("group_name"); pr_add.add_argument("rule_type")
    pr_dis = psr.add_parser("disable"); pr_dis.add_argument("rule_id", type=int)
    pr_en = psr.add_parser("enable"); pr_en.add_argument("rule_id", type=int)
    pr_del = psr.add_parser("delete"); pr_del.add_argument("rule_id", type=int)

    sub.add_parser("scan-once", help="Run a single scan now")
    sub.add_parser("rules-available", help="List registered rule types")
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
    }[args.cmd](args)


if __name__ == "__main__":
    main()
