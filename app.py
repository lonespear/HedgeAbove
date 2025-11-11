"""
HedgeAbove - AI-Powered Finance Analytics Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="HedgeAbove - Rise Above Market Uncertainty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">HedgeAbove</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Rise Above Market Uncertainty</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Portfolio Tracker", "Risk Analytics", "AI Predictions", "Options & Hedging"]
)

# Sidebar user info (placeholder for freemium)
st.sidebar.markdown("---")
st.sidebar.markdown("### Account")
st.sidebar.info("**Free Tier**\n\n1 Portfolio | 5 Predictions/day")
st.sidebar.button("Upgrade to Pro", type="primary", use_container_width=True)

# Home Page
if page == "Home":
    st.header("Welcome to HedgeAbove")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Portfolio Analytics")
        st.write("Track your investments across multiple portfolios with real-time insights.")

    with col2:
        st.markdown("### 🤖 AI Predictions")
        st.write("Machine learning models forecast price movements with confidence intervals.")

    with col3:
        st.markdown("### 🛡️ Risk Management")
        st.write("Advanced risk analytics and hedging strategies to protect your wealth.")

    st.markdown("---")

    # Feature showcase
    st.subheader("Core Features")

    features = {
        "Risk Analytics": ["Value at Risk (VaR)", "Correlation Analysis", "Sharpe Ratio", "Portfolio Optimization"],
        "AI/ML Predictions": ["LSTM Neural Networks", "Prophet Forecasting", "Ensemble Models", "Backtesting"],
        "Options Tools": ["Strategy Builder", "Greeks Calculator", "P&L Diagrams", "Break-even Analysis"],
        "Hedging Strategies": ["Portfolio Hedging", "Position Sizing", "Hedge Effectiveness", "Cost-Benefit Analysis"]
    }

    col1, col2 = st.columns(2)

    with col1:
        for feature, items in list(features.items())[:2]:
            with st.expander(f"**{feature}**"):
                for item in items:
                    st.write(f"- {item}")

    with col2:
        for feature, items in list(features.items())[2:]:
            with st.expander(f"**{feature}**"):
                for item in items:
                    st.write(f"- {item}")

    st.markdown("---")
    st.info("💡 **Getting Started:** Select 'Portfolio Tracker' from the sidebar to add your first portfolio!")

# Portfolio Tracker Page
elif page == "Portfolio Tracker":
    st.header("📊 Portfolio Tracker")

    # Sample portfolio data (in production, this would come from user input/database)
    st.subheader("Your Portfolio")

    # Portfolio input section
    with st.expander("➕ Add New Position", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Ticker Symbol", placeholder="AAPL")
        with col2:
            shares = st.number_input("Shares", min_value=0.0, step=1.0)
        with col3:
            avg_price = st.number_input("Avg Price", min_value=0.0, step=0.01)

        if st.button("Add Position"):
            st.success(f"Added {shares} shares of {ticker} at ${avg_price}")

    # Sample portfolio data
    portfolio_data = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        'Shares': [50, 30, 20, 15, 25],
        'Avg Price': [150.00, 300.00, 2800.00, 250.00, 450.00],
        'Current Price': [175.50, 380.25, 2950.00, 245.75, 495.30],
        'Value': [8775.00, 11407.50, 59000.00, 3686.25, 12382.50]
    })

    portfolio_data['Cost Basis'] = portfolio_data['Shares'] * portfolio_data['Avg Price']
    portfolio_data['P&L'] = portfolio_data['Value'] - portfolio_data['Cost Basis']
    portfolio_data['P&L %'] = ((portfolio_data['Current Price'] - portfolio_data['Avg Price']) / portfolio_data['Avg Price'] * 100).round(2)

    st.dataframe(
        portfolio_data.style.format({
            'Avg Price': '${:.2f}',
            'Current Price': '${:.2f}',
            'Value': '${:,.2f}',
            'Cost Basis': '${:,.2f}',
            'P&L': '${:,.2f}',
            'P&L %': '{:.2f}%'
        }),
        use_container_width=True
    )

    # Portfolio metrics
    st.subheader("Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)

    total_value = portfolio_data['Value'].sum()
    total_cost = portfolio_data['Cost Basis'].sum()
    total_pl = portfolio_data['P&L'].sum()
    total_pl_pct = (total_pl / total_cost * 100)

    col1.metric("Total Value", f"${total_value:,.2f}")
    col2.metric("Total Cost", f"${total_cost:,.2f}")
    col3.metric("P&L", f"${total_pl:,.2f}", f"{total_pl_pct:.2f}%")
    col4.metric("Positions", len(portfolio_data))

    # Asset allocation chart
    st.subheader("Asset Allocation")
    fig = px.pie(
        portfolio_data,
        values='Value',
        names='Symbol',
        title='Portfolio Allocation by Value',
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

# Risk Analytics Page
elif page == "Risk Analytics":
    st.header("🛡️ Risk Analytics")

    st.info("⚠️ This is a demo interface. Full risk calculations will be implemented with real market data.")

    # Risk metrics
    st.subheader("Portfolio Risk Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Portfolio Beta", "1.15", "0.05")
    col2.metric("Sharpe Ratio", "1.87", "0.12")
    col3.metric("Max Drawdown", "-18.5%", "-2.3%")

    # VaR Calculator
    st.subheader("Value at Risk (VaR)")

    col1, col2 = st.columns(2)
    with col1:
        confidence_level = st.slider("Confidence Level", 90, 99, 95)
        time_horizon = st.selectbox("Time Horizon", ["1 Day", "1 Week", "1 Month"])

    with col2:
        st.metric("VaR (Historical)", "$2,450", help=f"{confidence_level}% confidence, {time_horizon}")
        st.metric("VaR (Monte Carlo)", "$2,680", help=f"{confidence_level}% confidence, {time_horizon}")

    # Correlation matrix
    st.subheader("Correlation Matrix")

    # Generate sample correlation data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    correlation_matrix = pd.DataFrame(
        np.random.uniform(0.3, 0.9, (5, 5)),
        index=symbols,
        columns=symbols
    )
    np.fill_diagonal(correlation_matrix.values, 1.0)

    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdYlGn',
        title='Asset Correlation Matrix'
    )
    st.plotly_chart(fig, use_container_width=True)

# AI Predictions Page
elif page == "AI Predictions":
    st.header("🤖 AI Price Predictions")

    st.info("⚠️ This is a demo interface. ML models will be trained on real market data in production.")

    # Prediction input
    col1, col2 = st.columns(2)
    with col1:
        predict_ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
    with col2:
        prediction_horizon = st.selectbox("Prediction Horizon", ["1 Week", "1 Month", "3 Months", "6 Months"])

    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Running ML models..."):
            # Simulate prediction generation
            st.success(f"✅ Prediction generated for {predict_ticker}")

            # Generate sample prediction data
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            current_price = 175.50

            # Simulated predictions
            predictions = current_price + np.cumsum(np.random.randn(30) * 2)
            upper_bound = predictions + np.random.uniform(5, 10, 30)
            lower_bound = predictions - np.random.uniform(5, 10, 30)

            # Plot predictions
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=dates, y=predictions,
                mode='lines',
                name='Predicted Price',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=dates, y=upper_bound,
                mode='lines',
                name='Upper Bound (95% CI)',
                line=dict(width=0),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=dates, y=lower_bound,
                mode='lines',
                name='Lower Bound (95% CI)',
                fill='tonexty',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(width=0),
                showlegend=True
            ))

            fig.update_layout(
                title=f'{predict_ticker} Price Prediction - {prediction_horizon}',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Model confidence
            st.subheader("Model Insights")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Change", "+8.5%", "Bullish")
            col2.metric("Model Confidence", "78%", "High")
            col3.metric("Expected Range", "$168-$192")

            # Model details
            with st.expander("📊 Model Details"):
                st.write("**Ensemble Model Components:**")
                st.write("- LSTM Neural Network (40% weight)")
                st.write("- Facebook Prophet (35% weight)")
                st.write("- ARIMA (25% weight)")
                st.write("\n**Training Data:** 5 years of historical prices")
                st.write("**Last Updated:** 2 hours ago")

# Options & Hedging Page
elif page == "Options & Hedging":
    st.header("📐 Options & Hedging Strategies")

    tab1, tab2 = st.tabs(["Options Calculator", "Hedging Strategies"])

    with tab1:
        st.subheader("Options Profit/Loss Calculator")

        col1, col2, col3 = st.columns(3)
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            position = st.selectbox("Position", ["Long", "Short"])
        with col2:
            strike_price = st.number_input("Strike Price", value=180.0, step=1.0)
            premium = st.number_input("Premium", value=5.0, step=0.1)
        with col3:
            current_price = st.number_input("Current Stock Price", value=175.0, step=1.0)
            contracts = st.number_input("Contracts", value=1, step=1)

        if st.button("Calculate P&L"):
            # Generate P&L diagram
            stock_prices = np.linspace(strike_price * 0.7, strike_price * 1.3, 100)

            if option_type == "Call":
                payoff = np.maximum(stock_prices - strike_price, 0) - premium
            else:  # Put
                payoff = np.maximum(strike_price - stock_prices, 0) - premium

            if position == "Short":
                payoff = -payoff

            payoff *= contracts * 100  # Options contracts are for 100 shares

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_prices,
                y=payoff,
                mode='lines',
                fill='tozeroy',
                name='P&L',
                line=dict(color='blue', width=2)
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=strike_price, line_dash="dash", line_color="red", annotation_text="Strike")

            fig.update_layout(
                title=f'{position} {option_type} P&L Diagram',
                xaxis_title='Stock Price at Expiration ($)',
                yaxis_title='Profit/Loss ($)',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Break-even calculation
            if option_type == "Call":
                break_even = strike_price + premium
            else:
                break_even = strike_price - premium

            col1, col2, col3 = st.columns(3)
            col1.metric("Break-even Price", f"${break_even:.2f}")
            col2.metric("Max Profit", "Unlimited" if position == "Long" and option_type == "Call" else f"${abs(payoff.max()):,.2f}")
            col3.metric("Max Loss", f"${abs(payoff.min()):,.2f}")

    with tab2:
        st.subheader("Portfolio Hedging Strategies")

        st.write("**AI-Recommended Hedges for Your Portfolio:**")

        hedge_strategies = pd.DataFrame({
            'Strategy': ['Protective Put', 'Collar', 'Index Put', 'Short Futures'],
            'Cost': ['$2,450', '$890', '$1,750', '$0 (margin)'],
            'Protection': ['95%', '80%', '85%', '90%'],
            'Score': [8.5, 9.2, 7.8, 6.5]
        })

        for idx, row in hedge_strategies.iterrows():
            with st.expander(f"**{row['Strategy']}** - Score: {row['Score']}/10"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Cost", row['Cost'])
                col2.metric("Protection Level", row['Protection'])
                col3.metric("Effectiveness", f"{row['Score']}/10")

                st.write(f"**Description:** Simulated hedge strategy details would appear here.")
                st.button(f"Implement {row['Strategy']}", key=f"hedge_{idx}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "HedgeAbove v0.1.0 | Rise Above Market Uncertainty | "
    "<a href='#'>Documentation</a> | <a href='#'>Support</a>"
    "</div>",
    unsafe_allow_html=True
)
