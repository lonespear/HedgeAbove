"""Home page."""

import streamlit as st


def render():
    st.header("HedgeAbove — Quant Research Platform")
    st.caption("Rise above market uncertainty. Free data only. No paid APIs.")

    # ── Top-level capability cards ──────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Signal scanner")
        st.write(
            "23 rules (technical + fundamental). Telegram alerts via cron on a Pi, "
            "or in-browser via the Alerts page. Watchlist groups, snoozes, "
            "earnings-window filters, dedup-per-day."
        )
    with c2:
        st.markdown("### Strategy Lab")
        st.write(
            "Cross-sectional scoring (6 presets, sector-neutral), strategy "
            "backtesting (multi-position, inverse-vol sizing, full tearsheet), "
            "walk-forward optimization, factor IC analysis."
        )
    with c3:
        st.markdown("### Free data layer")
        st.write(
            "yfinance bars + earnings dates + sectors. SEC EDGAR XBRL for "
            "*point-in-time* fundamentals (defeats look-ahead). FRED for "
            "macro regime conditioning (VIX, yield curve)."
        )

    st.markdown("---")

    # ── Two-pronged deployment story ────────────────────────────────
    st.subheader("Two-pronged deployment, one codebase")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### Lean Pi cron alerter")
        st.code(
            "pip install -r requirements-core.txt\n"
            "python -m hedgeabove.cli init\n"
            "python scan.py    # cron entry",
            language="bash",
        )
        st.caption(
            "Lean install, no Streamlit. Runs as a cron job on a Raspberry "
            "Pi (or any box) and pushes alerts to Telegram. Same SQLite the "
            "dashboard uses, so changes here reach the cron immediately."
        )
    with d2:
        st.markdown("##### Full dashboard")
        st.code(
            "pip install -r requirements.txt\n"
            "streamlit run app.py",
            language="bash",
        )
        st.caption(
            "Everything in this UI — Strategy Lab, Alerts management, "
            "Stock Screener, Portfolio Builder, Risk Analytics, "
            "Optimization, Predictions, Backtesting, Options Pricing. "
            "Optionally also runs the scanner in cron on the same box."
        )

    st.markdown("---")

    # ── Quick-start CTAs ────────────────────────────────────────────
    st.subheader("Where to start")
    cta_cols = st.columns(3)
    with cta_cols[0]:
        st.markdown("#### Try a backtest")
        st.write(
            "Pick a rule + a basket + a hold horizon. See an equity curve, "
            "Sharpe, max DD, monthly returns heatmap, drawdown chart, "
            "rolling Sharpe, and beta to SPY."
        )
        st.info("→ **Strategy Lab → Strategy Backtester** tab")
    with cta_cols[1]:
        st.markdown("#### Test a factor")
        st.write(
            "Compute Information Coefficient — Spearman correlation between "
            "factor ranks and forward returns. Mean IC > 0.05 = real factor; "
            "IR > 0.5 = actionable."
        )
        st.info("→ **Strategy Lab → Factor IC** tab")
    with cta_cols[2]:
        st.markdown("#### Manage live alerts")
        st.write(
            "Configure watchlists and rules that the cron scanner will fire "
            "on. Snooze noisy tickers. Backtest any rule with regime, year, "
            "and earnings filters."
        )
        st.info("→ **Alerts & Watchlists**")

    st.markdown("---")

    # ── What's where ────────────────────────────────────────────────
    st.subheader("All pages")
    pages = {
        "Stock Screener": "Filter 2,535+ stocks by 50+ fundamental metrics with preset strategies.",
        "Alerts & Watchlists": "Scanner config + Rule Analytics with composite mode, regime / year / earnings breakdowns.",
        "Strategy Lab": "Cross-sectional Scorer, Strategy Backtester, Walk-Forward, Factor IC.",
        "Portfolio Builder": "Track positions with real-time prices and P&L.",
        "Risk Analytics": "VaR (Historical / Parametric / Monte Carlo), CVaR, GARCH, copulas.",
        "Portfolio Optimization": "Efficient Frontier, Max Sharpe, Min Volatility, Risk Parity.",
        "AI Predictions": "ARIMA, GARCH, Prophet, ML models for return forecasting.",
        "Backtesting": "Portfolio-level backtests with comparison to benchmarks.",
        "Options Pricing": "Black-Scholes, Greeks, implied volatility surfaces.",
    }
    cols = st.columns(2)
    for i, (page, desc) in enumerate(pages.items()):
        with cols[i % 2]:
            st.markdown(f"**{page}** — {desc}")

    st.markdown("---")
    st.success(
        "Real-data sanity from this build: SPY rsi_oversold during VIX>30 "
        "regimes hits **84.9%** at 20d (vs 60.9% in normal vol). "
        "8-ticker basket with inverse-vol sizing: **Sharpe 2.13 / Max DD -11.0%**."
    )
