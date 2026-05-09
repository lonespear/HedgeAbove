"""
Market data fetching via yfinance.
All functions use Streamlit caching for performance.
"""

import streamlit as st
import pandas as pd
import yfinance as yf


@st.cache_data(ttl=300)
def get_current_price(symbol):
    """Fetch current price from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_historical_data(symbols, period='1y'):
    """Fetch historical price data for one or more symbols."""
    try:
        data = yf.download(symbols, period=period, progress=False)
        if len(symbols) == 1:
            return data[['Close']].rename(columns={'Close': symbols[0]})
        return data['Close']
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_stock_info(symbol):
    """Fetch comprehensive stock information with all fundamental metrics."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='5y')

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]

        stock_data = {
            # Basic Info
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),

            # Price Data
            'current_price': current_price,
            'previous_close': info.get('previousClose', current_price),
            'open': info.get('open', current_price),
            'day_high': info.get('dayHigh', current_price),
            'day_low': info.get('dayLow', current_price),
            'volume': info.get('volume', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', current_price),
            '52_week_low': info.get('fiftyTwoWeekLow', current_price),

            # Valuation Metrics
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            'price_to_book': info.get('priceToBook', None),
            'price_to_sales': info.get('priceToSalesTrailing12Months', None),
            'ev_to_revenue': info.get('enterpriseToRevenue', None),
            'ev_to_ebitda': info.get('enterpriseToEbitda', None),

            # Per Share Metrics
            'eps_ttm': info.get('trailingEps', None),
            'eps_forward': info.get('forwardEps', None),
            'book_value_per_share': info.get('bookValue', None),
            'revenue_per_share': info.get('revenuePerShare', None),
            'free_cash_flow_per_share': (
                info.get('freeCashflow', 0) / info.get('sharesOutstanding', 1)
                if info.get('sharesOutstanding') else None
            ),

            # Growth Metrics
            'earnings_growth': info.get('earningsGrowth', None),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None),
            'revenue_growth': info.get('revenueGrowth', None),

            # Profitability Metrics
            'profit_margin': info.get('profitMargins', None),
            'operating_margin': info.get('operatingMargins', None),
            'gross_margin': info.get('grossMargins', None),
            'ebitda_margin': info.get('ebitdaMargins', None),
            'roe': info.get('returnOnEquity', None),
            'roa': info.get('returnOnAssets', None),
            'roic': info.get('returnOnCapital', None),

            # Dividend Metrics
            'dividend_yield': info.get('dividendYield', 0),
            'dividend_rate': info.get('dividendRate', 0),
            'payout_ratio': info.get('payoutRatio', None),
            'five_year_avg_div_yield': info.get('fiveYearAvgDividendYield', None),

            # Financial Health
            'current_ratio': info.get('currentRatio', None),
            'quick_ratio': info.get('quickRatio', None),
            'debt_to_equity': info.get('debtToEquity', None),
            'total_debt': info.get('totalDebt', 0),
            'total_cash': info.get('totalCash', 0),
            'free_cash_flow': info.get('freeCashflow', 0),
            'operating_cash_flow': info.get('operatingCashflow', 0),

            # Analyst Estimates
            'target_mean_price': info.get('targetMeanPrice', None),
            'target_high_price': info.get('targetHighPrice', None),
            'target_low_price': info.get('targetLowPrice', None),
            'recommendation': info.get('recommendationKey', 'N/A'),
            'number_of_analysts': info.get('numberOfAnalystOpinions', 0),

            # Additional
            'beta': info.get('beta', None),
            'shares_outstanding': info.get('sharesOutstanding', 0),
            'float_shares': info.get('floatShares', 0),
            'shares_short': info.get('sharesShort', 0),
            'short_ratio': info.get('shortRatio', None),
            'short_percent': info.get('shortPercentOfFloat', None),
        }
        return stock_data
    except Exception:
        return None
