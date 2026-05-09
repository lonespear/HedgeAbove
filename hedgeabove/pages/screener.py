"""Stock Screener page."""

import streamlit as st
import pandas as pd
import plotly.express as px

from hedgeabove.data.market import get_stock_info
from hedgeabove.data.universe import get_sp500_tickers


def render():
    st.header("Comprehensive Stock Screener")
    st.caption("Screen 2,535+ global stocks with 50+ fundamental metrics")

    # Screening Presets
    st.subheader("Screening Strategy")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Value Stocks", use_container_width=True):
            st.session_state.screen_preset = "value"
    with col2:
        if st.button("Growth Stocks", use_container_width=True):
            st.session_state.screen_preset = "growth"
    with col3:
        if st.button("Quality Stocks", use_container_width=True):
            st.session_state.screen_preset = "quality"
    with col4:
        if st.button("Dividend Stocks", use_container_width=True):
            st.session_state.screen_preset = "dividend"

    if 'screen_preset' not in st.session_state:
        st.session_state.screen_preset = None

    st.markdown("---")

    # Filters Section
    with st.expander("Screening Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Basic Filters**")
            sector_filter = st.multiselect(
                "Sector",
                ["Technology", "Healthcare", "Financials", "Consumer Discretionary",
                 "Consumer Staples", "Energy", "Industrials", "Communication Services",
                 "Utilities", "Real Estate", "Materials", "All"],
                default=["All"],
            )
            results_limit = st.slider("Max Results", 10, 200, 50, step=10)

        with col2:
            st.markdown("**Valuation Filters**")
            pe_min = st.number_input("P/E Min", value=0.0, step=1.0)
            pe_max = st.number_input("P/E Max", value=100.0, step=1.0)
            pb_min = st.number_input("P/B Min", value=0.0, step=0.1)
            pb_max = st.number_input("P/B Max", value=10.0, step=0.1)

        with col3:
            st.markdown("**Profitability Filters**")
            roe_min = st.number_input("ROE Min (%)", value=0.0, step=1.0)
            roic_min = st.number_input("ROIC Min (%)", value=0.0, step=1.0)
            profit_margin_min = st.number_input("Profit Margin Min (%)", value=0.0, step=1.0)

    # Apply preset filters
    if st.session_state.screen_preset == "value":
        st.info("**Value Strategy**: Low P/E (<15), Low P/B (<3), High Dividend Yield (>2%)")
        pe_max = 15.0
        pb_max = 3.0
    elif st.session_state.screen_preset == "growth":
        st.info("**Growth Strategy**: High Earnings Growth (>15%), High Revenue Growth (>10%), High ROE (>15%)")
        roe_min = 15.0
    elif st.session_state.screen_preset == "quality":
        st.info("**Quality Strategy**: High ROIC (>15%), High Profit Margin (>10%), Low Debt/Equity (<50%)")
        roic_min = 15.0
        profit_margin_min = 10.0
    elif st.session_state.screen_preset == "dividend":
        st.info("**Dividend Strategy**: High Dividend Yield (>3%), Payout Ratio <70%, Positive Cash Flow")

    st.markdown("---")

    all_tickers = get_sp500_tickers()

    if len(all_tickers) > 0:
        with st.spinner(f"Fetching comprehensive data for up to {results_limit} stocks... (This may take 30-60 seconds)"):
            screener_data = []
            progress_bar = st.progress(0)

            for idx, ticker in enumerate(all_tickers[:results_limit]):
                stock_info = get_stock_info(ticker)
                if stock_info:
                    if "All" not in sector_filter:
                        if stock_info['sector'] not in sector_filter:
                            continue

                    if stock_info['pe_ratio']:
                        if stock_info['pe_ratio'] < pe_min or stock_info['pe_ratio'] > pe_max:
                            continue

                    if stock_info['price_to_book']:
                        if stock_info['price_to_book'] < pb_min or stock_info['price_to_book'] > pb_max:
                            continue

                    if stock_info['roe']:
                        if stock_info['roe'] * 100 < roe_min:
                            continue

                    if stock_info['roic']:
                        if stock_info['roic'] * 100 < roic_min:
                            continue

                    if stock_info['profit_margin']:
                        if stock_info['profit_margin'] * 100 < profit_margin_min:
                            continue

                    screener_data.append({
                        'Symbol': ticker,
                        'Company': stock_info['name'][:30],
                        'Sector': stock_info['sector'],
                        'Price': stock_info['current_price'],
                        'Mkt Cap': stock_info['market_cap'],
                        'P/E': stock_info['pe_ratio'],
                        'Fwd P/E': stock_info['forward_pe'],
                        'PEG': stock_info['peg_ratio'],
                        'P/B': stock_info['price_to_book'],
                        'P/S': stock_info['price_to_sales'],
                        'EV/EBITDA': stock_info['ev_to_ebitda'],
                        'BVPS': stock_info['book_value_per_share'],
                        'EPS (TTM)': stock_info['eps_ttm'],
                        'EPS (Fwd)': stock_info['eps_forward'],
                        'EPS Growth': stock_info['earnings_growth'] * 100 if stock_info['earnings_growth'] else None,
                        'Rev Growth': stock_info['revenue_growth'] * 100 if stock_info['revenue_growth'] else None,
                        'ROE': stock_info['roe'] * 100 if stock_info['roe'] else None,
                        'ROA': stock_info['roa'] * 100 if stock_info['roa'] else None,
                        'ROIC': stock_info['roic'] * 100 if stock_info['roic'] else None,
                        'Profit Margin': stock_info['profit_margin'] * 100 if stock_info['profit_margin'] else None,
                        'Operating Margin': stock_info['operating_margin'] * 100 if stock_info['operating_margin'] else None,
                        'Gross Margin': stock_info['gross_margin'] * 100 if stock_info['gross_margin'] else None,
                        'Div Yield': stock_info['dividend_yield'] * 100 if stock_info['dividend_yield'] else 0,
                        'Payout Ratio': stock_info['payout_ratio'] * 100 if stock_info['payout_ratio'] else None,
                        'Current Ratio': stock_info['current_ratio'],
                        'D/E Ratio': stock_info['debt_to_equity'],
                        'FCF': stock_info['free_cash_flow'],
                        'Target Price': stock_info['target_mean_price'],
                        'Upside': (
                            ((stock_info['target_mean_price'] / stock_info['current_price']) - 1) * 100
                            if stock_info['target_mean_price'] and stock_info['current_price'] else None
                        ),
                        'Rating': stock_info['recommendation'],
                        'Beta': stock_info['beta'],
                    })

                progress_bar.progress((idx + 1) / min(results_limit, len(all_tickers)))

            progress_bar.empty()

        if screener_data:
            screener_df = pd.DataFrame(screener_data)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stocks Found", len(screener_df))
            with col2:
                avg_pe = screener_df['P/E'].mean()
                st.metric("Avg P/E", f"{avg_pe:.2f}" if pd.notna(avg_pe) else "N/A")
            with col3:
                avg_roe = screener_df['ROE'].mean()
                st.metric("Avg ROE", f"{avg_roe:.1f}%" if pd.notna(avg_roe) else "N/A")
            with col4:
                avg_div = screener_df['Div Yield'].mean()
                st.metric("Avg Div Yield", f"{avg_div:.2f}%" if pd.notna(avg_div) else "N/A")

            st.markdown("---")

            _render_screener_tabs(screener_df)
            _render_sector_chart(screener_df)
            _render_quick_add(screener_df)
        else:
            st.info("No stocks match your filter criteria. Try adjusting filters or increasing results limit.")
    else:
        st.error("Could not load stock list")


def _render_screener_tabs(screener_df):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Valuation", "Growth & Profitability",
        "Dividends", "Financial Health", "Analyst Data",
    ])

    fmt_price = '${:.2f}'
    fmt_pct = lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    fmt_pct1 = lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    fmt_val = lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    fmt_cap = lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x > 0 else "N/A"
    fmt_fcf = lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x > 0 else "N/A"

    with tab1:
        st.subheader("Stock Overview")
        st.dataframe(
            screener_df[['Symbol', 'Company', 'Sector', 'Price', 'Mkt Cap', 'P/E', 'ROE', 'Div Yield', 'Beta']]
            .style.format({'Price': fmt_price, 'Mkt Cap': fmt_cap, 'P/E': fmt_val, 'ROE': fmt_pct1,
                           'Div Yield': fmt_pct, 'Beta': fmt_val}),
            use_container_width=True, height=500,
        )

    with tab2:
        st.subheader("Valuation Metrics")
        st.dataframe(
            screener_df[['Symbol', 'Company', 'Price', 'P/E', 'Fwd P/E', 'PEG', 'P/B', 'P/S', 'EV/EBITDA', 'BVPS']]
            .style.format({'Price': fmt_price, 'P/E': fmt_val, 'Fwd P/E': fmt_val, 'PEG': fmt_val,
                           'P/B': fmt_val, 'P/S': fmt_val, 'EV/EBITDA': fmt_val,
                           'BVPS': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"}),
            use_container_width=True, height=500,
        )

    with tab3:
        st.subheader("Growth & Profitability")
        st.dataframe(
            screener_df[['Symbol', 'Company', 'EPS (TTM)', 'EPS (Fwd)', 'EPS Growth', 'Rev Growth',
                         'ROE', 'ROA', 'ROIC', 'Profit Margin', 'Operating Margin', 'Gross Margin']]
            .style.format({
                'EPS (TTM)': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                'EPS (Fwd)': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                'EPS Growth': fmt_pct1, 'Rev Growth': fmt_pct1,
                'ROE': fmt_pct1, 'ROA': fmt_pct1, 'ROIC': fmt_pct1,
                'Profit Margin': fmt_pct1, 'Operating Margin': fmt_pct1, 'Gross Margin': fmt_pct1,
            }),
            use_container_width=True, height=500,
        )

    with tab4:
        st.subheader("Dividend Metrics")
        st.dataframe(
            screener_df[['Symbol', 'Company', 'Price', 'Div Yield', 'Payout Ratio', 'EPS (TTM)', 'FCF']]
            .style.format({'Price': fmt_price, 'Div Yield': fmt_pct,
                           'Payout Ratio': fmt_pct1,
                           'EPS (TTM)': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                           'FCF': fmt_fcf}),
            use_container_width=True, height=500,
        )

    with tab5:
        st.subheader("Financial Health")
        st.dataframe(
            screener_df[['Symbol', 'Company', 'Current Ratio', 'D/E Ratio', 'FCF', 'Beta', 'Mkt Cap']]
            .style.format({'Current Ratio': fmt_val, 'D/E Ratio': fmt_val,
                           'FCF': fmt_fcf, 'Beta': fmt_val, 'Mkt Cap': fmt_cap}),
            use_container_width=True, height=500,
        )

    with tab6:
        st.subheader("Analyst Targets & Recommendations")
        st.dataframe(
            screener_df[['Symbol', 'Company', 'Price', 'Target Price', 'Upside', 'Rating']]
            .style.format({'Price': fmt_price,
                           'Target Price': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                           'Upside': lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"}),
            use_container_width=True, height=500,
        )


def _render_sector_chart(screener_df):
    st.markdown("---")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Sector Distribution")
        sector_counts = screener_df['Sector'].value_counts()
        fig = px.bar(
            x=sector_counts.values, y=sector_counts.index, orientation='h',
            labels={'x': 'Count', 'y': 'Sector'}, title="Stocks by Sector",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("")
        st.write("")
        st.write("")
        for sector, count in sector_counts.items():
            st.write(f"**{sector}**: {count}")


def _render_quick_add(screener_df):
    st.markdown("---")
    st.subheader("Quick Add to Portfolio")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        quick_symbol = st.selectbox("Select Stock", screener_df['Symbol'].tolist())
        if quick_symbol:
            selected_company = screener_df[screener_df['Symbol'] == quick_symbol]['Company'].iloc[0]
            st.caption(f"**{selected_company}**")
    with col2:
        quick_shares = st.number_input("Shares", min_value=1.0, value=10.0, step=1.0, key="screener_shares")
    with col3:
        quick_price = screener_df[screener_df['Symbol'] == quick_symbol]['Price'].iloc[0] if quick_symbol else 100
        use_market_price = st.checkbox("Use market price", value=True, key="screener_use_market")
        if use_market_price:
            quick_avg_price = quick_price
            st.info(f"${quick_avg_price:.2f}")
        else:
            quick_avg_price = st.number_input("Custom Price", value=float(quick_price), step=0.01, key="screener_custom_price")
    with col4:
        st.write("")
        st.write("")
        total_cost = quick_shares * quick_avg_price
        st.metric("Total", f"${total_cost:,.2f}")

    if st.button("Add to Portfolio", type="primary", use_container_width=True, key="screener_add"):
        if quick_symbol not in st.session_state.portfolio['Symbol'].values:
            new_row = pd.DataFrame({
                'Symbol': [quick_symbol], 'Shares': [quick_shares], 'Avg Price': [quick_avg_price],
            })
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"Added {quick_shares} shares of {quick_symbol} ({selected_company}) at ${quick_avg_price:.2f}")
            st.balloons()
        else:
            st.warning(f"{quick_symbol} already in portfolio. Use Portfolio Builder to edit it.")
