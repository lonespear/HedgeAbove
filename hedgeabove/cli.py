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
    python -m hedgeabove.cli score <preset|--weights k=v,...> [--symbols A,B | --universe sp500] [--top N] [--save-as <name>]
    python -m hedgeabove.cli presets
    python -m hedgeabove.cli strategy <rule_type> [--symbols A,B | --watchlist <name>] [--period 5y] [--hold-days 20] [--param k=v ...]
    python -m hedgeabove.cli walk-forward <SYMBOL> <rule_type> --param-name <k> --param-grid v1,v2,v3 [--period 10y] [--horizon 20] [--folds 5] [--score sharpe|hit_rate|avg_return]
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


def cmd_strategy(args):
    from hedgeabove.backtest.strategy import simulate_basket, trades_to_dataframe

    rules_list = None
    if args.rules:
        rules_list = [(rt.strip(), {}) for rt in args.rules.split(",") if rt.strip()]
        for rt, _ in rules_list:
            if rt not in tech_rules.REGISTRY and rt not in fund_rules.REGISTRY:
                print(f"Unknown rule type: {rt}", file=sys.stderr)
                sys.exit(1)
            if rt in {"analyst_upside_above", "dividend_yield_above"}:
                print(f"'{rt}' isn't yet backtestable.", file=sys.stderr)
                sys.exit(1)
    else:
        if not args.rule_type:
            print("Need rule_type or --rules", file=sys.stderr)
            sys.exit(1)
        if args.rule_type not in tech_rules.REGISTRY and args.rule_type not in fund_rules.REGISTRY:
            print(f"Unknown rule type: {args.rule_type}", file=sys.stderr)
            sys.exit(1)
        if args.rule_type in {"analyst_upside_above", "dividend_yield_above"}:
            print(f"'{args.rule_type}' isn't yet backtestable.", file=sys.stderr)
            sys.exit(1)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.watchlist:
        g = db.get_watchlist_group_by_name(args.watchlist)
        if not g:
            print(f"No watchlist named '{args.watchlist}'.", file=sys.stderr)
            sys.exit(1)
        symbols = db.get_watchlist_group_tickers(g[0])
        if not symbols:
            print(f"Watchlist '{args.watchlist}' is empty.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Need --symbols or --watchlist", file=sys.stderr)
        sys.exit(1)

    params = dict(args.param) if args.param else {}
    if rules_list is not None:
        label = f"[{args.combine.upper()}: {','.join(rt for rt, _ in rules_list)}]"
        print(f"\nSimulating composite {label} on {len(symbols)} ticker(s)  "
              f"period={args.period}  hold={args.hold_days}d")
        res = simulate_basket(symbols, rules=rules_list, combiner=args.combine,
                              period=args.period, hold_days=args.hold_days,
                              benchmark=args.benchmark)
    else:
        print(f"\nSimulating '{args.rule_type}' on {len(symbols)} ticker(s)  "
              f"period={args.period}  hold={args.hold_days}d  params={params}")
        res = simulate_basket(symbols, args.rule_type, params,
                              period=args.period, hold_days=args.hold_days,
                              benchmark=args.benchmark)
    s = res["summary"]
    if not s.get("n_trades"):
        print("No trades — rule never fired (or no future bars to close).")
        return

    print(f"\n  Span: {s['span_years']:.1f}y   Trades: {s['n_trades']}   "
          f"Trades/yr: {s['trades_per_year']:.1f}")
    print(f"  Win rate:      {s['win_rate']:>7.1%}")
    print(f"  Avg trade:     {s['avg_trade_return']:>+7.2%}")
    print(f"  Median trade:  {s['median_trade_return']:>+7.2%}")
    print(f"  Total return:  {s['total_return']:>+7.1%}")
    print(f"  Ann return:    {s['ann_return']:>+7.1%}")
    sh = s.get("sharpe_ann")
    print(f"  Sharpe (ann):  {sh:>7.2f}" if sh is not None else "  Sharpe (ann):    -  ")
    print(f"  Max drawdown:  {s['max_drawdown']:>7.1%}")

    bm = s.get("benchmark")
    if bm:
        print(f"\n  Benchmark ({bm}, buy-and-hold over same span):")
        print(f"    Total return:  {s['benchmark_total_return']:>+7.1%}")
        print(f"    Ann return:    {s['benchmark_ann_return']:>+7.1%}")
        print(f"    Max drawdown:  {s['benchmark_max_drawdown']:>7.1%}")
        print(f"    Alpha (total): {s['alpha_total']:>+7.1%}")
        print(f"    Alpha (ann):   {s['alpha_ann']:>+7.1%}")

    if args.show_trades:
        df = trades_to_dataframe(res["trades"]).tail(15)
        print("\n  Last 15 trades:")
        for _, r in df.iterrows():
            sign = "+" if r.return_pct >= 0 else ""
            print(f"    {r.entry}  ->  {r.exit}  {r.symbol:6s}  "
                  f"${r.entry_price:>8.2f} -> ${r.exit_price:>8.2f}  "
                  f"{sign}{r.return_pct*100:6.2f}%")

    if args.tearsheet:
        from hedgeabove.backtest.tearsheet import tearsheet
        ts = tearsheet(res["equity_curve"], res.get("benchmark_curve"))

        yr = ts["calendar_year_returns"]
        if not yr.empty:
            print("\n  Calendar-year returns:")
            for d, r in yr.items():
                print(f"    {d.year}:  {r*100:+7.2f}%")

        dd_max = ts.get("max_drawdown")
        if dd_max is not None:
            print(f"\n  Drawdown analysis:")
            print(f"    Max drawdown:        {dd_max*100:7.2f}%")
            print(f"    Max DD bottomed on:  {ts['max_drawdown_date'].date()}")
            print(f"    Longest underwater:  {ts['longest_drawdown_days']} trading days")

        rs = ts["rolling_sharpe_252d"].dropna()
        if not rs.empty:
            print(f"\n  Rolling 252-day Sharpe:")
            print(f"    Latest: {rs.iloc[-1]:5.2f}")
            print(f"    Min:    {rs.min():5.2f}")
            print(f"    Max:    {rs.max():5.2f}")
            print(f"    Mean:   {rs.mean():5.2f}")

        beta = ts.get("beta")
        if beta is not None:
            bm = res["summary"].get("benchmark", "benchmark")
            print(f"\n  Beta to {bm}: {beta:.2f}")


def cmd_walk_forward(args):
    from hedgeabove.backtest.walkforward import walk_forward_optimize
    if args.rule_type not in tech_rules.REGISTRY:
        print(f"walk-forward currently supports only technical rules; got {args.rule_type!r}",
              file=sys.stderr)
        sys.exit(1)
    grid = []
    for x in args.param_grid.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            grid.append(int(x))
        except ValueError:
            try:
                grid.append(float(x))
            except ValueError:
                print(f"Invalid grid value: {x!r}", file=sys.stderr)
                sys.exit(1)
    if len(grid) < 2:
        print("--param-grid needs at least 2 values to optimize over.", file=sys.stderr)
        sys.exit(1)

    sym = args.symbol.upper()
    print(f"\nWalk-forward {args.rule_type} on {sym}")
    print(f"  Param: {args.param_name} in {grid}")
    print(f"  Folds: {args.folds}  Horizon: {args.horizon}d  Period: {args.period}  Score: {args.score_metric}")

    res = walk_forward_optimize(sym, args.rule_type, args.param_name, grid,
                                period=args.period, horizon=args.horizon,
                                n_folds=args.folds, score=args.score_metric)
    if not res:
        print("Insufficient data for walk-forward (need >= 60 bars per fold).")
        return

    print()
    print("  Fold-by-fold (train -> chosen param -> OOS):")
    print(f"    {'fold':>4s}  {'train start':>11s} {'-> end':>11s}  "
          f"{'test start':>11s} {'-> end':>11s}  {'best':>6s}  {'OOS score':>9s}  {'OOS n':>6s}")
    for f in res["fold_records"]:
        bp = f["best_param"]
        bp_str = str(bp)
        oos = f["oos_score"]
        oos_str = "-inf" if oos == float("-inf") else f"{oos:>9.3f}"
        print(f"    {f['fold']:>4d}  {str(f['train_start']):>11s} -> {str(f['train_end']):<11s}  "
              f"{str(f['test_start']):>11s} -> {str(f['test_end']):<11s}  "
              f"{bp_str:>6s}  {oos_str}  {f['oos_n_fires']:>6d}")

    s = res["walk_forward_summary"]
    print(f"\n  Aggregate out-of-sample:")
    print(f"    Total OOS fires: {s.get('n_oos_fires', 0)}")
    if s.get("n_oos_fires", 0) > 0:
        print(f"    Hit rate:        {s['oos_hit_rate']:>7.1%}")
        print(f"    Avg return:      {s['oos_avg_return']:>+7.2%}")
        print(f"    Median return:   {s['oos_median_return']:>+7.2%}")
        sh = s.get("oos_sharpe_ann")
        if sh is not None:
            print(f"    Sharpe (ann):    {sh:>7.2f}")


def cmd_presets(args):
    from hedgeabove.scoring.composite import PRESETS, FACTORS
    print("Available factors:")
    for name in FACTORS:
        print(f"  - {name}")
    print()
    print("Built-in presets (negative weight = lower-is-better):")
    for name, w in PRESETS.items():
        weight_str = ", ".join(f"{k}={v:+.2f}" for k, v in w.items())
        print(f"  - {name:15s} {{ {weight_str} }}")


def cmd_score(args):
    from hedgeabove.scoring.composite import (
        score_universe, score_with_preset, PRESETS, FACTORS,
    )
    # Resolve weights
    if args.preset:
        if args.preset not in PRESETS:
            print(f"Unknown preset: {args.preset!r}. Available: {list(PRESETS)}",
                  file=sys.stderr)
            sys.exit(1)
        weights = PRESETS[args.preset]
        label = f"preset '{args.preset}'"
    elif args.weights:
        weights = {}
        for kv in args.weights.split(","):
            if "=" not in kv:
                print(f"Bad --weights entry: {kv!r}. Expected k=v.", file=sys.stderr)
                sys.exit(1)
            k, v = kv.split("=", 1)
            weights[k.strip()] = float(v)
        unknown = set(weights) - set(FACTORS)
        if unknown:
            print(f"Unknown factor(s): {sorted(unknown)}. "
                  f"Available: {list(FACTORS)}", file=sys.stderr)
            sys.exit(1)
        label = "custom weights"
    else:
        print("Need --preset or --weights k1=v1,k2=v2", file=sys.stderr)
        sys.exit(1)

    # Resolve universe
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.universe == "sp500":
        from hedgeabove.data.universe import get_sp500_tickers
        symbols = get_sp500_tickers()
    else:
        print("Need --symbols A,B,C or --universe sp500", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(symbols)} ticker(s) on {label}: {weights}")
    print("(parallel fetch — this can take a while; cold S&P 500 is ~5-10 min)")

    def progress(i, n, sym):
        if i == n or i % 5 == 0:
            print(f"  [{i:>4d}/{n}] {sym:8s}", end="\r", flush=True)

    df = score_universe(symbols, weights, progress=progress)
    print()  # newline after progress

    if df.empty:
        print("No results — every ticker is missing at least one required factor.")
        return

    cols_to_show = list(weights.keys()) + ["composite_score"]
    top = df.head(args.top)[cols_to_show]
    print(f"\nTop {len(top)} of {len(df)} (eligible after factor-completeness filter):")
    print(top.to_string(float_format=lambda x: f"{x:>9.4f}"))

    if args.save_as:
        from hedgeabove import db
        db.init_db()
        existing = db.get_watchlist_group_by_name(args.save_as)
        if existing:
            gid = existing[0]
            print(f"\nWatchlist group '{args.save_as}' exists; appending to it.")
        else:
            gid = db.create_watchlist_group(args.save_as)
            print(f"\nCreated watchlist group '{args.save_as}' (id={gid})")
        for sym in top.index:
            db.add_ticker_to_group(gid, sym)
        print(f"Added {len(top)} ticker(s) to '{args.save_as}'.")


def cmd_analyze(args):
    from hedgeabove.backtest.signals import (
        summarize_rule, summarize_composite, DEFAULT_HORIZONS,
    )
    sym = args.symbol.upper()
    params_dict = dict(args.param) if args.param else {}

    if args.rules:
        rules_list = [(rt.strip(), {}) for rt in args.rules.split(",") if rt.strip()]
        for rt, _ in rules_list:
            if rt not in tech_rules.REGISTRY and rt not in fund_rules.REGISTRY:
                print(f"Unknown rule type: {rt}", file=sys.stderr)
                sys.exit(1)
            if rt in {"analyst_upside_above", "dividend_yield_above"}:
                print(f"'{rt}' isn't yet backtestable.", file=sys.stderr)
                sys.exit(1)
        label = f"[{args.combine.upper()}: {','.join(rt for rt, _ in rules_list)}]"
        print(f"\nAnalyzing {sym} :: {label}  period={args.period}")
        summary, fires = summarize_composite(sym, rules_list, args.combine, args.period)
    else:
        if not args.rule_type:
            print("Need rule_type or --rules", file=sys.stderr)
            sys.exit(1)
        if args.rule_type not in tech_rules.REGISTRY and args.rule_type not in fund_rules.REGISTRY:
            print(f"Unknown rule type: {args.rule_type}", file=sys.stderr)
            sys.exit(1)
        if args.rule_type in {"analyst_upside_above", "dividend_yield_above"}:
            print(f"'{args.rule_type}' isn't yet backtestable.", file=sys.stderr)
            sys.exit(1)
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

    if args.by_regime:
        from hedgeabove.backtest.regime import by_regime_summary, available_regimes
        if args.by_regime not in available_regimes():
            print(f"\nUnknown regime: {args.by_regime}. "
                  f"Available: {available_regimes()}", file=sys.stderr)
            return
        for h in DEFAULT_HORIZONS:
            agg = by_regime_summary(fires, args.by_regime, horizon=h)
            if agg.empty:
                continue
            print(f"\n  Forward returns by {args.by_regime} regime ({h}d horizon):")
            print(f"    {'regime':>12s}  {'n':>5s}  {'hit_rate':>9s}  {'avg':>9s}  {'median':>9s}  {'std':>8s}")
            for _, r in agg.iterrows():
                # Use bracket access — `median` and `std` collide with pandas methods
                print(f"    {r['regime']:>12s}  {int(r['n']):>5d}  {r['hit_rate']:>8.1%}  "
                      f"{r['avg']:>+8.2%}  {r['median']:>+8.2%}  {r['std']:>7.2%}")


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

    sub.add_parser("presets", help="List built-in factor presets")

    pwf = sub.add_parser("walk-forward",
                         help="Walk-forward parameter optimization (out-of-sample validation)")
    pwf.add_argument("symbol")
    pwf.add_argument("rule_type")
    pwf.add_argument("--param-name", required=True,
                     help="Rule param to vary (e.g. threshold)")
    pwf.add_argument("--param-grid", required=True,
                     help="Comma-separated candidate values, e.g. 20,25,30,35,40")
    pwf.add_argument("--period", default="10y", help="yfinance period (default 10y)")
    pwf.add_argument("--horizon", type=int, default=20,
                     help="Forward-return horizon for scoring + OOS (default 20 days)")
    pwf.add_argument("--folds", type=int, default=5,
                     help="Number of time-ordered folds (default 5)")
    pwf.add_argument("--score-metric", choices=["sharpe", "hit_rate", "avg_return"],
                     default="sharpe", help="Optimizer objective (default sharpe)")

    pst = sub.add_parser("strategy", help="Backtest a single-position long basket strategy (single rule or composite)")
    pst.add_argument("rule_type", nargs="?", help="Single rule (or use --rules)")
    pst.add_argument("--rules", help="Comma-separated rules for composite strategies")
    pst.add_argument("--combine", choices=["all", "any", "majority"], default="all",
                     help="Combiner for --rules: all=AND (default), any=OR, majority=>50%%")
    pstg = pst.add_mutually_exclusive_group(required=True)
    pstg.add_argument("--symbols", help="Comma-separated tickers")
    pstg.add_argument("--watchlist", help="Use this watchlist group's tickers")
    pst.add_argument("--period", default="5y", help="yfinance period (default 5y)")
    pst.add_argument("--hold-days", type=int, default=20, help="Trading days to hold each position (default 20)")
    pst.add_argument("--param", type=_parse_param, action="append",
                     help="Override default rule params, e.g. --param threshold=25")
    pst.add_argument("--benchmark", default="SPY",
                     help="Benchmark symbol for buy-and-hold comparison (default SPY)")
    pst.add_argument("--no-benchmark", dest="benchmark", action="store_const", const=None,
                     help="Skip benchmark comparison")
    pst.add_argument("--show-trades", action="store_true", help="Print last 15 trades")
    pst.add_argument("--tearsheet", action="store_true",
                     help="Print calendar-year returns, drawdown stats, rolling Sharpe, beta")

    psc = sub.add_parser("score", help="Cross-sectional rank a universe by weighted Z-scored factors")
    pscg = psc.add_mutually_exclusive_group(required=True)
    pscg.add_argument("--preset", help="Use a built-in preset (see `cli presets`)")
    pscg.add_argument("--weights", help="Custom weights, e.g. roe=0.4,pe=-0.6")
    psug = psc.add_mutually_exclusive_group(required=True)
    psug.add_argument("--symbols", help="Comma-separated tickers")
    psug.add_argument("--universe", choices=["sp500"], help="Built-in universe")
    psc.add_argument("--top", type=int, default=20, help="Number of top names to show / save")
    psc.add_argument("--save-as", help="Save the top-N as a watchlist group with this name")

    pa = sub.add_parser("analyze", help="Backtest a single rule, or a composite via --rules")
    pa.add_argument("symbol")
    pa.add_argument("rule_type", nargs="?", help="Single rule (or use --rules a,b for composite)")
    pa.add_argument("--rules", help="Comma-separated rules for composite analysis (replaces rule_type)")
    pa.add_argument("--combine", choices=["all", "any", "majority"], default="all",
                    help="Combiner for --rules: all=AND (default), any=OR, majority=>50%%")
    pa.add_argument("--period", default="5y", help="yfinance period (e.g. 1y, 5y, max). Default: 5y")
    pa.add_argument("--param", type=_parse_param, action="append",
                    help="Override default rule params for single-rule mode, e.g. --param threshold=25")
    pa.add_argument("--show-fires", action="store_true",
                    help="Also print the last 10 fire events with forward returns")
    pa.add_argument("--by-regime", choices=["vix", "yield_curve"],
                    help="Break down forward returns by macro regime (FRED VIXCLS or 2s10s slope)")

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
        "score": cmd_score,
        "presets": cmd_presets,
        "strategy": cmd_strategy,
        "walk-forward": cmd_walk_forward,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
