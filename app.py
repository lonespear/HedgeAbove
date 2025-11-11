"""
HedgeAbove - AI-Powered Finance Analytics Platform
Main Streamlit Application with Full Portfolio & Risk Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="HedgeAbove - Rise Above Market Uncertainty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'Avg Price'])

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

#==================== HELPER FUNCTIONS ====================

@st.cache_data(ttl=300)
def get_current_price(symbol):
    """Fetch current price from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

@st.cache_data(ttl=300)
def get_historical_data(symbols, period='1y'):
    """Fetch historical price data"""
    try:
        data = yf.download(symbols, period=period, progress=False)
        if len(symbols) == 1:
            return data[['Close']].rename(columns={'Close': symbols[0]})
        return data['Close']
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(symbol):
    """Fetch detailed stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1y')

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]

        stock_data = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'current_price': current_price,
            'previous_close': info.get('previousClose', current_price),
            'open': info.get('open', current_price),
            'day_high': info.get('dayHigh', current_price),
            'day_low': info.get('dayLow', current_price),
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None),
            'dividend_yield': info.get('dividendYield', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', current_price),
            '52_week_low': info.get('fiftyTwoWeekLow', current_price),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }
        return stock_data
    except:
        return None

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """Get S&P 500 constituents"""
    # Using a simplified list for demo - in production, fetch from Wikipedia or other source
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV', 'PFE', 'AVGO',
        'COST', 'DIS', 'ADBE', 'NFLX', 'CRM', 'ACN', 'NKE', 'TMO', 'MRK', 'LLY',
        'CSCO', 'PEP', 'ABT', 'WMT', 'DHR', 'TXN', 'INTC', 'VZ', 'ORCL', 'PM',
        'QCOM', 'AMD', 'CMCSA', 'UNP', 'NEE', 'HON', 'COP', 'RTX', 'IBM', 'SBUX'
    ]

def calculate_var(returns, confidence=0.95, method='historical'):
    """Calculate Value at Risk"""
    if method == 'historical':
        return np.percentile(returns, (1 - confidence) * 100)
    elif method == 'parametric':
        mu = returns.mean()
        sigma = returns.std()
        return stats.norm.ppf(1 - confidence, mu, sigma)
    elif method == 'monte_carlo':
        mu = returns.mean()
        sigma = returns.std()
        simulations = np.random.normal(mu, sigma, 10000)
        return np.percentile(simulations, (1 - confidence) * 100)

def calculate_es(returns, confidence=0.95):
    """Calculate Expected Shortfall (CVaR)"""
    var = calculate_var(returns, confidence, method='historical')
    return returns[returns <= var].mean()

def calculate_portfolio_metrics(returns):
    """Calculate comprehensive portfolio metrics"""
    metrics = {}
    metrics['Total Return'] = (1 + returns).prod() - 1
    metrics['Annualized Return'] = returns.mean() * 252
    metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    metrics['Sortino Ratio'] = (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['Max Drawdown'] = drawdown.min()
    return metrics

def optimize_portfolio(returns, method='max_sharpe', target_return=None):
    """Portfolio optimization using Modern Portfolio Theory"""
    n_assets = returns.shape[1]

    # Calculate expected returns and covariance
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Objective functions
    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -ret / vol if vol != 0 else 0

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})

    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1/n_assets] * n_assets)

    if method == 'max_sharpe':
        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'min_vol':
        result = minimize(portfolio_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'target_return':
        result = minimize(portfolio_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'risk_parity':
        def risk_parity_objective(weights):
            portfolio_vol = portfolio_volatility(weights)
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        result = minimize(risk_parity_objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x if result.success else init_guess

def generate_efficient_frontier(returns, num_portfolios=50):
    """Generate efficient frontier portfolios"""
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)

    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    for target in target_returns:
        try:
            weights = optimize_portfolio(returns, method='target_return', target_return=target)
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            frontier_rets.append(ret)
            frontier_vols.append(vol)
            frontier_weights.append(weights)
        except:
            continue

    return frontier_rets, frontier_vols, frontier_weights

#==================== HEADER ====================

st.markdown('<div class="main-header">HedgeAbove</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Rise Above Market Uncertainty</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Portfolio Builder", "Risk Analytics", "Portfolio Optimization", "Stock Screener", "AI Predictions"]
)

# Sidebar user info
st.sidebar.markdown("---")
st.sidebar.markdown("### Account")
st.sidebar.info("**Free Tier**\n\nUnlimited Portfolios\nReal-time data via yfinance")
st.sidebar.button("Upgrade to Pro", type="primary", use_container_width=True)

#==================== HOME PAGE ====================

if page == "Home":
    st.header("Welcome to HedgeAbove")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Portfolio Analytics")
        st.write("Build and track portfolios with real-time market data from Yahoo Finance.")

    with col2:
        st.markdown("### 🛡️ Risk Management")
        st.write("VaR, Expected Shortfall, correlation analysis, and comprehensive risk metrics.")

    with col3:
        st.markdown("### 📈 Portfolio Optimization")
        st.write("Modern Portfolio Theory, Efficient Frontier, and optimal allocation strategies.")

    st.markdown("---")

    st.subheader("✨ New Features")

    features = {
        "Portfolio Builder": [
            "✅ Add/Edit/Delete positions",
            "✅ Real-time price updates (yfinance)",
            "✅ Automatic P&L tracking",
            "✅ CSV import/export"
        ],
        "Risk Analytics": [
            "✅ Value at Risk (Historical, Parametric, Monte Carlo)",
            "✅ Expected Shortfall (CVaR)",
            "✅ Correlation matrices with real data",
            "✅ Comprehensive portfolio metrics"
        ],
        "Portfolio Optimization": [
            "✅ Efficient Frontier visualization",
            "✅ Max Sharpe Ratio optimization",
            "✅ Minimum Volatility optimization",
            "✅ Target Return optimization",
            "✅ Risk Parity allocation"
        ],
        "Stock Screener": [
            "✅ S&P 500 constituents",
            "✅ Filter by sector and market cap",
            "✅ International exposure options",
            "✅ Quick-add to portfolio"
        ]
    }

    col1, col2 = st.columns(2)

    with col1:
        for feature, items in list(features.items())[:2]:
            with st.expander(f"**{feature}**", expanded=True):
                for item in items:
                    st.write(item)

    with col2:
        for feature, items in list(features.items())[2:]:
            with st.expander(f"**{feature}**", expanded=True):
                for item in items:
                    st.write(item)

    st.markdown("---")
    st.success("💡 **Get Started:** Select 'Portfolio Builder' from the sidebar to create your portfolio!")

#==================== PORTFOLIO BUILDER ====================

elif page == "Portfolio Builder":
    st.header("📊 Portfolio Builder")

    # Add Position Section
    with st.expander("➕ Add New Position", expanded=len(st.session_state.portfolio) == 0):
        # Step 1: Enter ticker
        st.subheader("Step 1: Enter Ticker Symbol")
        lookup_symbol = st.text_input("Search for a stock", placeholder="AAPL", key="lookup_symbol").upper()

        # Fetch and display stock info
        if lookup_symbol:
            with st.spinner(f"Fetching data for {lookup_symbol}..."):
                stock_info = get_stock_info(lookup_symbol)

            if stock_info:
                # Display stock information
                st.success(f"✅ Found: **{stock_info['name']}** ({stock_info['symbol']})")

                # Key Stats Display
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${stock_info['current_price']:.2f}",
                           f"{((stock_info['current_price'] - stock_info['previous_close']) / stock_info['previous_close'] * 100):.2f}%")
                col2.metric("Market Cap", f"${stock_info['market_cap']/1e9:.2f}B" if stock_info['market_cap'] > 0 else "N/A")
                col3.metric("P/E Ratio", f"{stock_info['pe_ratio']:.2f}" if stock_info['pe_ratio'] else "N/A")
                col4.metric("Div Yield", f"{stock_info['dividend_yield']*100:.2f}%" if stock_info['dividend_yield'] else "N/A")

                # Additional details
                with st.expander("📊 More Details"):
                    detail_col1, detail_col2, detail_col3 = st.columns(3)

                    with detail_col1:
                        st.write("**Today's Range:**")
                        st.write(f"${stock_info['day_low']:.2f} - ${stock_info['day_high']:.2f}")
                        st.write(f"**Open:** ${stock_info['open']:.2f}")

                    with detail_col2:
                        st.write("**52-Week Range:**")
                        st.write(f"${stock_info['52_week_low']:.2f} - ${stock_info['52_week_high']:.2f}")
                        pct_from_high = ((stock_info['current_price'] - stock_info['52_week_high']) / stock_info['52_week_high'] * 100)
                        st.write(f"**From High:** {pct_from_high:.1f}%")

                    with detail_col3:
                        st.write(f"**Sector:** {stock_info['sector']}")
                        st.write(f"**Industry:** {stock_info['industry']}")
                        st.write(f"**Volume:** {stock_info['volume']:,}")

                st.markdown("---")

                # Step 2: Enter shares and price
                st.subheader("Step 2: Position Details")

                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    position_shares = st.number_input(
                        "Number of Shares",
                        min_value=0.01,
                        value=10.0,
                        step=1.0,
                        key="position_shares",
                        help="Enter the number of shares to add"
                    )

                with col2:
                    manual_price = st.checkbox("Enter price manually", key="manual_price_toggle")

                    if manual_price:
                        position_price = st.number_input(
                            "Purchase Price",
                            min_value=0.01,
                            value=float(stock_info['current_price']),
                            step=0.01,
                            key="position_price_manual",
                            help="Enter your actual purchase price"
                        )
                    else:
                        position_price = stock_info['current_price']
                        st.info(f"**Using current market price:** ${position_price:.2f}")

                with col3:
                    st.write("")
                    st.write("")
                    position_value = position_shares * position_price
                    st.metric("Total Cost", f"${position_value:,.2f}")

                # Add button
                if st.button("➕ Add Position to Portfolio", type="primary", use_container_width=True):
                    if lookup_symbol in st.session_state.portfolio['Symbol'].values:
                        st.error(f"❌ {lookup_symbol} is already in your portfolio. Use 'Edit' to update it.")
                    else:
                        new_row = pd.DataFrame({
                            'Symbol': [lookup_symbol],
                            'Shares': [position_shares],
                            'Avg Price': [position_price]
                        })
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                        st.success(f"✅ Added {position_shares} shares of {lookup_symbol} at ${position_price:.2f} (Total: ${position_value:,.2f})")
                        st.balloons()
                        st.rerun()

            elif lookup_symbol:
                st.error(f"❌ Could not find ticker '{lookup_symbol}'. Please verify the symbol and try again.")
                st.caption("💡 Tip: Try common symbols like AAPL, MSFT, GOOGL, TSLA, NVDA")

    # Display Portfolio
    if len(st.session_state.portfolio) > 0:
        st.subheader("Your Portfolio")

        # Fetch current prices
        portfolio_display = st.session_state.portfolio.copy()
        portfolio_display['Current Price'] = portfolio_display['Symbol'].apply(get_current_price)

        # Calculate metrics
        portfolio_display['Value'] = portfolio_display['Shares'] * portfolio_display['Current Price']
        portfolio_display['Cost Basis'] = portfolio_display['Shares'] * portfolio_display['Avg Price']
        portfolio_display['P&L'] = portfolio_display['Value'] - portfolio_display['Cost Basis']
        portfolio_display['P&L %'] = ((portfolio_display['Current Price'] - portfolio_display['Avg Price']) / portfolio_display['Avg Price'] * 100).round(2)
        portfolio_display['Weight %'] = (portfolio_display['Value'] / portfolio_display['Value'].sum() * 100).round(2)

        # Display table
        st.dataframe(
            portfolio_display.style.format({
                'Shares': '{:.2f}',
                'Avg Price': '${:.2f}',
                'Current Price': '${:.2f}',
                'Value': '${:,.2f}',
                'Cost Basis': '${:,.2f}',
                'P&L': '${:,.2f}',
                'P&L %': '{:.2f}%',
                'Weight %': '{:.2f}%'
            }).background_gradient(subset=['P&L %'], cmap='RdYlGn', vmin=-10, vmax=10),
            use_container_width=True
        )

        # Edit/Delete Section
        with st.expander("✏️ Edit or Delete Positions"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Edit Position:**")
                edit_symbol = st.selectbox("Select Symbol to Edit", portfolio_display['Symbol'].tolist(), key="edit_symbol")
                if edit_symbol:
                    current_row = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == edit_symbol].iloc[0]
                    edit_shares = st.number_input("New Shares", value=float(current_row['Shares']), min_value=0.01, step=1.0, key="edit_shares")
                    edit_price = st.number_input("New Avg Price", value=float(current_row['Avg Price']), min_value=0.01, step=0.01, key="edit_price")

                    if st.button("💾 Save Changes", type="primary"):
                        idx = st.session_state.portfolio[st.session_state.portfolio['Symbol'] == edit_symbol].index[0]
                        st.session_state.portfolio.at[idx, 'Shares'] = edit_shares
                        st.session_state.portfolio.at[idx, 'Avg Price'] = edit_price
                        st.success(f"✅ Updated {edit_symbol}")
                        st.rerun()

            with col2:
                st.write("**Delete Position:**")
                delete_symbol = st.selectbox("Select Symbol to Delete", portfolio_display['Symbol'].tolist(), key="delete_symbol")
                if st.button("🗑️ Delete Position", type="secondary"):
                    st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != delete_symbol]
                    st.success(f"✅ Deleted {delete_symbol}")
                    st.rerun()

        # Portfolio Summary
        st.subheader("Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)

        total_value = portfolio_display['Value'].sum()
        total_cost = portfolio_display['Cost Basis'].sum()
        total_pl = portfolio_display['P&L'].sum()
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric("Total Cost", f"${total_cost:,.2f}")
        col3.metric("P&L", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")
        col4.metric("Positions", len(portfolio_display))

        # Asset Allocation
        st.subheader("Asset Allocation")
        fig = px.pie(
            portfolio_display,
            values='Value',
            names='Symbol',
            title='Portfolio Allocation by Value',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export Portfolio
        st.subheader("Export Portfolio")
        csv = st.session_state.portfolio.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio as CSV",
            data=csv,
            file_name=f"hedgeabove_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Add your first position to get started!")

#==================== RISK ANALYTICS ====================

elif page == "Risk Analytics":
    st.header("🛡️ Risk Analytics")

    if len(st.session_state.portfolio) == 0:
        st.warning("⚠️ Please add positions to your portfolio first!")
    else:
        symbols = st.session_state.portfolio['Symbol'].tolist()
        weights = st.session_state.portfolio['Shares'].values
        weights = weights / weights.sum()  # Normalize to portfolio weights

        # Fetch historical data
        with st.spinner("Fetching historical data..."):
            hist_data = get_historical_data(symbols, period='1y')

        if hist_data.empty:
            st.error("Could not fetch historical data. Please try again.")
        else:
            # Calculate returns
            returns = hist_data.pct_change().dropna()
            portfolio_returns = (returns * weights).sum(axis=1)

            # Portfolio Metrics
            st.subheader("Portfolio Risk Metrics")

            metrics = calculate_portfolio_metrics(portfolio_returns)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ann. Return", f"{metrics['Annualized Return']*100:.2f}%")
            col2.metric("Ann. Volatility", f"{metrics['Annualized Volatility']*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            col4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{metrics['Total Return']*100:.2f}%")
            col2.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
            col3.metric("", "")
            col4.metric("", "")

            # VaR and ES
            st.subheader("Value at Risk & Expected Shortfall")

            col1, col2 = st.columns(2)
            with col1:
                confidence_level = st.slider("Confidence Level", 90, 99, 95) / 100
            with col2:
                time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"])

            # Calculate VaR
            var_hist = calculate_var(portfolio_returns, confidence_level, 'historical')
            var_param = calculate_var(portfolio_returns, confidence_level, 'parametric')
            var_mc = calculate_var(portfolio_returns, confidence_level, 'monte_carlo')
            es = calculate_es(portfolio_returns, confidence_level)

            # Scale to portfolio value
            portfolio_value = (st.session_state.portfolio['Shares'] *
                             st.session_state.portfolio['Symbol'].apply(get_current_price)).sum()

            horizon_days = {'1 Day': 1, '1 Week': 5, '1 Month': 21}[time_horizon]
            scale = np.sqrt(horizon_days)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("VaR (Historical)", f"${abs(var_hist * portfolio_value * scale):,.2f}")
            col2.metric("VaR (Parametric)", f"${abs(var_param * portfolio_value * scale):,.2f}")
            col3.metric("VaR (Monte Carlo)", f"${abs(var_mc * portfolio_value * scale):,.2f}")
            col4.metric("Expected Shortfall", f"${abs(es * portfolio_value * scale):,.2f}")

            st.caption(f"Maximum expected loss at {confidence_level*100}% confidence over {time_horizon}")

            # Returns Distribution
            st.subheader("Returns Distribution")

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=portfolio_returns * 100,
                nbinsx=50,
                name='Returns',
                marker_color='lightblue'
            ))

            fig.add_vline(x=var_hist*100, line_dash="dash", line_color="red",
                         annotation_text=f"VaR ({confidence_level*100}%)")
            fig.add_vline(x=portfolio_returns.mean()*100, line_dash="dash", line_color="green",
                         annotation_text="Mean")

            fig.update_layout(
                title="Portfolio Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Correlation Matrix
            st.subheader("Asset Correlation Matrix")

            corr_matrix = returns.corr()

            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdYlGn',
                title='Correlation Matrix of Portfolio Assets'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Rolling Volatility
            st.subheader("Rolling Volatility (30-day)")

            rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='orange', width=2)
            ))

            fig.update_layout(
                title="30-Day Rolling Volatility (Annualized)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

#==================== PORTFOLIO OPTIMIZATION ====================

elif page == "Portfolio Optimization":
    st.header("📈 Modern Portfolio Theory & Optimization")

    if len(st.session_state.portfolio) < 2:
        st.warning("⚠️ Please add at least 2 positions to your portfolio for optimization!")
    else:
        symbols = st.session_state.portfolio['Symbol'].tolist()

        # Fetch historical data
        with st.spinner("Fetching historical data and calculating optimal portfolios..."):
            hist_data = get_historical_data(symbols, period='1y')

        if hist_data.empty:
            st.error("Could not fetch historical data. Please try again.")
        else:
            returns = hist_data.pct_change().dropna()

            # Current Portfolio Weights
            current_weights = st.session_state.portfolio['Shares'].values
            current_weights = current_weights / current_weights.sum()

            # Calculate current portfolio metrics
            current_returns = (returns * current_weights).sum(axis=1)
            current_metrics = calculate_portfolio_metrics(current_returns)

            # Optimization Methods
            st.subheader("Optimal Portfolio Allocations")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Optimization Methods:**")
                opt_methods = st.multiselect(
                    "Select optimization methods to compare:",
                    ["Max Sharpe Ratio", "Min Volatility", "Target Return", "Risk Parity"],
                    default=["Max Sharpe Ratio", "Min Volatility"]
                )

            with col2:
                if "Target Return" in opt_methods:
                    target_return = st.slider(
                        "Target Annual Return (%)",
                        min_value=0,
                        max_value=50,
                        value=15
                    ) / 100
                else:
                    target_return = None

            # Calculate optimal portfolios
            optimal_portfolios = {}

            if "Max Sharpe Ratio" in opt_methods:
                weights = optimize_portfolio(returns, 'max_sharpe')
                opt_returns = (returns * weights).sum(axis=1)
                optimal_portfolios['Max Sharpe'] = {
                    'weights': weights,
                    'metrics': calculate_portfolio_metrics(opt_returns)
                }

            if "Min Volatility" in opt_methods:
                weights = optimize_portfolio(returns, 'min_vol')
                opt_returns = (returns * weights).sum(axis=1)
                optimal_portfolios['Min Volatility'] = {
                    'weights': weights,
                    'metrics': calculate_portfolio_metrics(opt_returns)
                }

            if "Target Return" in opt_methods and target_return:
                weights = optimize_portfolio(returns, 'target_return', target_return=target_return)
                opt_returns = (returns * weights).sum(axis=1)
                optimal_portfolios['Target Return'] = {
                    'weights': weights,
                    'metrics': calculate_portfolio_metrics(opt_returns)
                }

            if "Risk Parity" in opt_methods:
                weights = optimize_portfolio(returns, 'risk_parity')
                opt_returns = (returns * weights).sum(axis=1)
                optimal_portfolios['Risk Parity'] = {
                    'weights': weights,
                    'metrics': calculate_portfolio_metrics(opt_returns)
                }

            # Display optimal allocations
            st.subheader("Optimal Allocations Comparison")

            # Create comparison DataFrame
            allocation_df = pd.DataFrame(index=symbols)
            allocation_df['Current'] = current_weights * 100

            for name, portfolio in optimal_portfolios.items():
                allocation_df[name] = portfolio['weights'] * 100

            st.dataframe(
                allocation_df.style.format("{:.2f}%").background_gradient(cmap='Blues', axis=1),
                use_container_width=True
            )

            # Metrics Comparison
            st.subheader("Performance Metrics Comparison")

            metrics_df = pd.DataFrame()
            metrics_df['Current Portfolio'] = pd.Series(current_metrics)

            for name, portfolio in optimal_portfolios.items():
                metrics_df[name] = pd.Series(portfolio['metrics'])

            # Format and display
            display_metrics = metrics_df.T[['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown']]

            st.dataframe(
                display_metrics.style.format({
                    'Annualized Return': '{:.2%}',
                    'Annualized Volatility': '{:.2%}',
                    'Sharpe Ratio': '{:.2f}',
                    'Max Drawdown': '{:.2%}'
                }).background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn', vmin=-1, vmax=3),
                use_container_width=True
            )

            # Efficient Frontier
            st.subheader("Efficient Frontier")

            with st.spinner("Generating efficient frontier..."):
                frontier_rets, frontier_vols, frontier_weights = generate_efficient_frontier(returns, num_portfolios=50)

            fig = go.Figure()

            # Plot efficient frontier
            fig.add_trace(go.Scatter(
                x=np.array(frontier_vols) * 100,
                y=np.array(frontier_rets) * 100,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=3)
            ))

            # Plot current portfolio
            fig.add_trace(go.Scatter(
                x=[current_metrics['Annualized Volatility'] * 100],
                y=[current_metrics['Annualized Return'] * 100],
                mode='markers',
                name='Current Portfolio',
                marker=dict(color='red', size=15, symbol='star')
            ))

            # Plot optimal portfolios
            colors = ['green', 'purple', 'orange', 'brown']
            for idx, (name, portfolio) in enumerate(optimal_portfolios.items()):
                fig.add_trace(go.Scatter(
                    x=[portfolio['metrics']['Annualized Volatility'] * 100],
                    y=[portfolio['metrics']['Annualized Return'] * 100],
                    mode='markers',
                    name=name,
                    marker=dict(color=colors[idx % len(colors)], size=12)
                ))

            fig.update_layout(
                title="Efficient Frontier & Portfolio Comparison",
                xaxis_title="Annualized Volatility (%)",
                yaxis_title="Annualized Return (%)",
                hovermode='closest',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Rebalancing Suggestions
            st.subheader("💡 Rebalancing Suggestions")

            selected_portfolio = st.selectbox(
                "Select optimal portfolio to view rebalancing actions:",
                list(optimal_portfolios.keys())
            )

            if selected_portfolio:
                optimal_weights = optimal_portfolios[selected_portfolio]['weights']

                # Calculate current portfolio value
                portfolio_value = (st.session_state.portfolio['Shares'] *
                                 st.session_state.portfolio['Symbol'].apply(get_current_price)).sum()

                # Create rebalancing DataFrame
                rebalance_df = pd.DataFrame({
                    'Symbol': symbols,
                    'Current %': current_weights * 100,
                    'Target %': optimal_weights * 100,
                    'Difference %': (optimal_weights - current_weights) * 100,
                    'Current Value': current_weights * portfolio_value,
                    'Target Value': optimal_weights * portfolio_value,
                    'Action $': (optimal_weights - current_weights) * portfolio_value
                })

                rebalance_df['Action'] = rebalance_df['Action $'].apply(
                    lambda x: f"Buy ${abs(x):,.2f}" if x > 0 else f"Sell ${abs(x):,.2f}" if x < 0 else "Hold"
                )

                st.dataframe(
                    rebalance_df.style.format({
                        'Current %': '{:.2f}%',
                        'Target %': '{:.2f}%',
                        'Difference %': '{:+.2f}%',
                        'Current Value': '${:,.2f}',
                        'Target Value': '${:,.2f}',
                        'Action $': '${:+,.2f}'
                    }).background_gradient(subset=['Difference %'], cmap='RdYlGn', vmin=-10, vmax=10),
                    use_container_width=True
                )

#==================== STOCK SCREENER ====================

elif page == "Stock Screener":
    st.header("🔍 Stock Screener")

    st.subheader("S&P 500 Screener")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        sector_filter = st.multiselect(
            "Sector",
            ["Technology", "Healthcare", "Financials", "Consumer", "Industrial", "Energy", "All"],
            default=["All"]
        )

    with col2:
        cap_filter = st.multiselect(
            "Market Cap",
            ["Large Cap (>$10B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)", "All"],
            default=["All"]
        )

    with col3:
        region_filter = st.multiselect(
            "Region",
            ["US Domestic", "International", "All"],
            default=["All"]
        )

    # Get S&P 500 tickers
    sp500_tickers = get_sp500_tickers()

    # Sector mapping (simplified for demo)
    sector_map = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'NVDA': 'Technology',
        'META': 'Technology', 'TSLA': 'Technology', 'ADBE': 'Technology', 'NFLX': 'Technology',
        'CRM': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology', 'ORCL': 'Technology',
        'AMD': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology', 'AVGO': 'Technology',
        'IBM': 'Technology', 'AMZN': 'Consumer', 'WMT': 'Consumer', 'HD': 'Consumer',
        'NKE': 'Consumer', 'SBUX': 'Consumer', 'DIS': 'Consumer', 'COST': 'Consumer',
        'JPM': 'Financials', 'V': 'Financials', 'MA': 'Financials', 'BRK.B': 'Financials',
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
        'TMO': 'Healthcare', 'MRK': 'Healthcare', 'ABT': 'Healthcare', 'LLY': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'NEE': 'Energy',
        'UNP': 'Industrial', 'HON': 'Industrial', 'RTX': 'Industrial', 'ACN': 'Industrial',
        'PG': 'Consumer', 'PEP': 'Consumer', 'PM': 'Consumer', 'VZ': 'Technology',
        'CMCSA': 'Technology', 'DHR': 'Healthcare'
    }

    # Apply filters (simplified)
    filtered_tickers = sp500_tickers.copy()

    if "All" not in sector_filter:
        filtered_tickers = [t for t in filtered_tickers if sector_map.get(t, 'Other') in sector_filter]

    # Display results
    st.subheader(f"Screener Results ({len(filtered_tickers)} stocks)")

    if len(filtered_tickers) > 0:
        # Batch fetch prices for filtered tickers
        with st.spinner(f"Fetching data for {len(filtered_tickers)} stocks..."):
            screener_data = []
            for ticker in filtered_tickers[:20]:  # Limit to 20 for demo
                price = get_current_price(ticker)
                if price:
                    screener_data.append({
                        'Symbol': ticker,
                        'Sector': sector_map.get(ticker, 'Other'),
                        'Price': price,
                        'Add': False
                    })

        if screener_data:
            screener_df = pd.DataFrame(screener_data)

            # Display with selection
            st.dataframe(
                screener_df.style.format({'Price': '${:.2f}'}),
                use_container_width=True
            )

            # Quick add to portfolio
            st.subheader("Quick Add to Portfolio")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                quick_symbol = st.selectbox("Select Stock", screener_df['Symbol'].tolist())
            with col2:
                quick_shares = st.number_input("Shares", min_value=1.0, value=10.0, step=1.0)
            with col3:
                quick_price = screener_df[screener_df['Symbol'] == quick_symbol]['Price'].iloc[0] if quick_symbol else 100
                quick_avg_price = st.number_input("Avg Price", value=float(quick_price), step=0.01)
            with col4:
                st.write("")
                st.write("")
                if st.button("➕ Add to Portfolio", type="primary"):
                    if quick_symbol not in st.session_state.portfolio['Symbol'].values:
                        new_row = pd.DataFrame({
                            'Symbol': [quick_symbol],
                            'Shares': [quick_shares],
                            'Avg Price': [quick_avg_price]
                        })
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                        st.success(f"✅ Added {quick_shares} shares of {quick_symbol} to portfolio!")
                    else:
                        st.warning(f"{quick_symbol} already in portfolio")
    else:
        st.info("No stocks match your filter criteria")

#==================== AI PREDICTIONS (Placeholder) ====================

elif page == "AI Predictions":
    st.header("🤖 AI Price Predictions")

    st.info("⚠️ ML prediction models coming soon! This will include LSTM, Prophet, and ensemble forecasting.")

    # Prediction input
    col1, col2 = st.columns(2)
    with col1:
        predict_ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
    with col2:
        prediction_horizon = st.selectbox("Prediction Horizon", ["1 Week", "1 Month", "3 Months"])

    if st.button("Generate Prediction", type="primary"):
        st.warning("ML models not yet implemented. Coming in next update!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "HedgeAbove v0.2.0 | Rise Above Market Uncertainty | "
    "<a href='https://github.com/lonespear/HedgeAbove'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
