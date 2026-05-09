"""Portfolio Builder page with SQLite persistence."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from hedgeabove.data.market import get_stock_info, get_current_price
from hedgeabove.db import (
    list_portfolios, create_portfolio, delete_portfolio,
    load_positions, upsert_position, delete_position, save_dataframe,
)


def _ensure_active_portfolio():
    """Make sure there's an active portfolio selected; create default if needed."""
    portfolios = list_portfolios()
    if not portfolios:
        pid = create_portfolio("My Portfolio")
        portfolios = [(pid, "My Portfolio", "")]

    if 'active_portfolio_id' not in st.session_state:
        st.session_state.active_portfolio_id = portfolios[0][0]

    return portfolios


def _sync_to_session(portfolio_id):
    """Load DB positions into session_state.portfolio."""
    st.session_state.portfolio = load_positions(portfolio_id)


def render():
    st.header("Portfolio Builder")

    # Portfolio selector
    portfolios = _ensure_active_portfolio()
    portfolio_names = {p[0]: p[1] for p in portfolios}

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        selected_id = st.selectbox(
            "Active Portfolio",
            options=[p[0] for p in portfolios],
            format_func=lambda x: portfolio_names[x],
            index=list(portfolio_names.keys()).index(st.session_state.active_portfolio_id)
            if st.session_state.active_portfolio_id in portfolio_names else 0,
            key="portfolio_selector",
        )
        if selected_id != st.session_state.active_portfolio_id:
            st.session_state.active_portfolio_id = selected_id
            _sync_to_session(selected_id)
            st.rerun()

    with col2:
        new_name = st.text_input("New portfolio name", key="new_portfolio_name")
    with col3:
        st.write("")
        st.write("")
        if st.button("Create", key="create_portfolio") and new_name.strip():
            pid = create_portfolio(new_name.strip())
            st.session_state.active_portfolio_id = pid
            _sync_to_session(pid)
            st.rerun()

    pid = st.session_state.active_portfolio_id

    # Load from DB if session is empty or stale
    if 'portfolio' not in st.session_state or st.session_state.portfolio is None:
        _sync_to_session(pid)

    # Add Position Section
    with st.expander("Add New Position", expanded=len(st.session_state.portfolio) == 0):
        st.subheader("Step 1: Enter Ticker Symbol")
        lookup_symbol = st.text_input("Search for a stock", placeholder="AAPL", key="lookup_symbol").upper()

        if lookup_symbol:
            with st.spinner(f"Fetching data for {lookup_symbol}..."):
                stock_info = get_stock_info(lookup_symbol)

            if stock_info:
                st.success(f"Found: **{stock_info['name']}** ({stock_info['symbol']})")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${stock_info['current_price']:.2f}",
                            f"{((stock_info['current_price'] - stock_info['previous_close']) / stock_info['previous_close'] * 100):.2f}%")
                col2.metric("Market Cap", f"${stock_info['market_cap']/1e9:.2f}B" if stock_info['market_cap'] > 0 else "N/A")
                col3.metric("P/E Ratio", f"{stock_info['pe_ratio']:.2f}" if stock_info['pe_ratio'] else "N/A")
                col4.metric("Div Yield", f"{stock_info['dividend_yield']*100:.2f}%" if stock_info['dividend_yield'] else "N/A")

                with st.expander("More Details"):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        st.write(f"**Today's Range:** ${stock_info['day_low']:.2f} - ${stock_info['day_high']:.2f}")
                        st.write(f"**Open:** ${stock_info['open']:.2f}")
                    with dc2:
                        st.write(f"**52-Week Range:** ${stock_info['52_week_low']:.2f} - ${stock_info['52_week_high']:.2f}")
                        pct_from_high = ((stock_info['current_price'] - stock_info['52_week_high']) / stock_info['52_week_high'] * 100)
                        st.write(f"**From High:** {pct_from_high:.1f}%")
                    with dc3:
                        st.write(f"**Sector:** {stock_info['sector']}")
                        st.write(f"**Industry:** {stock_info['industry']}")
                        st.write(f"**Volume:** {stock_info['volume']:,}")

                st.markdown("---")
                st.subheader("Step 2: Position Details")

                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    position_shares = st.number_input("Number of Shares", min_value=0.01, value=10.0, step=1.0, key="position_shares")

                with col2:
                    manual_price = st.checkbox("Enter price manually", key="manual_price_toggle")
                    if manual_price:
                        position_price = st.number_input("Purchase Price", min_value=0.01,
                                                         value=float(stock_info['current_price']),
                                                         step=0.01, key="position_price_manual")
                    else:
                        position_price = stock_info['current_price']
                        st.info(f"**Using current market price:** ${position_price:.2f}")

                with col3:
                    st.write("")
                    st.write("")
                    position_value = position_shares * position_price
                    st.metric("Total Cost", f"${position_value:,.2f}")

                if st.button("Add Position to Portfolio", type="primary", use_container_width=True):
                    if lookup_symbol in st.session_state.portfolio['Symbol'].values:
                        st.error(f"{lookup_symbol} is already in your portfolio. Use 'Edit' to update it.")
                    else:
                        upsert_position(pid, lookup_symbol, position_shares, position_price)
                        _sync_to_session(pid)
                        st.success(f"Added {position_shares} shares of {lookup_symbol} at ${position_price:.2f}")
                        st.balloons()
                        st.rerun()

            elif lookup_symbol:
                st.error(f"Could not find ticker '{lookup_symbol}'. Please verify the symbol and try again.")

    # Display Portfolio
    if len(st.session_state.portfolio) > 0:
        st.subheader("Your Portfolio")

        portfolio_display = st.session_state.portfolio.copy()
        portfolio_display['Current Price'] = portfolio_display['Symbol'].apply(get_current_price)
        portfolio_display['Value'] = portfolio_display['Shares'] * portfolio_display['Current Price']
        portfolio_display['Cost Basis'] = portfolio_display['Shares'] * portfolio_display['Avg Price']
        portfolio_display['P&L'] = portfolio_display['Value'] - portfolio_display['Cost Basis']
        portfolio_display['P&L %'] = ((portfolio_display['Current Price'] - portfolio_display['Avg Price']) / portfolio_display['Avg Price'] * 100).round(2)
        portfolio_display['Weight %'] = (portfolio_display['Value'] / portfolio_display['Value'].sum() * 100).round(2)

        st.dataframe(
            portfolio_display.style.format({
                'Shares': '{:.2f}', 'Avg Price': '${:.2f}', 'Current Price': '${:.2f}',
                'Value': '${:,.2f}', 'Cost Basis': '${:,.2f}', 'P&L': '${:,.2f}',
                'P&L %': '{:.2f}%', 'Weight %': '{:.2f}%',
            }).background_gradient(subset=['P&L %'], cmap='RdYlGn', vmin=-10, vmax=10),
            use_container_width=True,
        )

        # Edit/Delete Section
        with st.expander("Edit or Delete Positions"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Edit Position:**")
                edit_symbol = st.selectbox("Select Symbol to Edit", portfolio_display['Symbol'].tolist(), key="edit_symbol")
                if edit_symbol:
                    current_row = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == edit_symbol].iloc[0]
                    edit_shares = st.number_input("New Shares", value=float(current_row['Shares']), min_value=0.01, step=1.0, key="edit_shares")
                    edit_price = st.number_input("New Avg Price", value=float(current_row['Avg Price']), min_value=0.01, step=0.01, key="edit_price")

                    if st.button("Save Changes", type="primary"):
                        upsert_position(pid, edit_symbol, edit_shares, edit_price)
                        _sync_to_session(pid)
                        st.success(f"Updated {edit_symbol}")
                        st.rerun()

            with col2:
                st.write("**Delete Position:**")
                del_symbol = st.selectbox("Select Symbol to Delete", portfolio_display['Symbol'].tolist(), key="delete_symbol")
                if st.button("Delete Position", type="secondary"):
                    delete_position(pid, del_symbol)
                    _sync_to_session(pid)
                    st.success(f"Deleted {del_symbol}")
                    st.rerun()

        # Portfolio Summary
        st.subheader("Portfolio Summary")
        total_value = portfolio_display['Value'].sum()
        total_cost = portfolio_display['Cost Basis'].sum()
        total_pl = portfolio_display['P&L'].sum()
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric("Total Cost", f"${total_cost:,.2f}")
        col3.metric("P&L", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")
        col4.metric("Positions", len(portfolio_display))

        # Asset Allocation
        st.subheader("Asset Allocation")
        fig = px.pie(portfolio_display, values='Value', names='Symbol',
                     title='Portfolio Allocation by Value', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

        # Export
        st.subheader("Export Portfolio")
        csv = st.session_state.portfolio.to_csv(index=False)
        st.download_button(
            label="Download Portfolio as CSV", data=csv,
            file_name=f"hedgeabove_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Add your first position to get started!")
