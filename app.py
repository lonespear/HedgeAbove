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
    """Get comprehensive S&P 500 constituents with full sector coverage (~500+ tickers)"""
    return [
        # Technology - Mega Cap (40 stocks)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
        'ADBE', 'NFLX', 'CRM', 'CSCO', 'INTC', 'AMD', 'TXN', 'QCOM', 'IBM', 'NOW',
        'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',
        'PANW', 'ADSK', 'ANSS', 'WDAY', 'TEAM', 'DDOG', 'CRWD', 'ZS', 'SNOW', 'NET',

        # Technology - Software & Cloud (30 stocks)
        'MSFT', 'ORCL', 'SAP', 'SHOP', 'SQ', 'TWLO', 'DOCU', 'ZM', 'OKTA', 'SPLK',
        'VEEV', 'RNG', 'HUBS', 'ZI', 'BILL', 'MNDY', 'PATH', 'GTLB', 'S', 'ESTC',
        'MDB', 'CFLT', 'DT', 'DOCN', 'FROG', 'PD', 'NCNO', 'ASAN', 'PCOR', 'BRZE',

        # Technology - Semiconductors (25 stocks)
        'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'ADI', 'MCHP', 'KLAC', 'LRCX',
        'AMAT', 'MU', 'NXPI', 'MRVL', 'ON', 'MPWR', 'SWKS', 'QRVO', 'ENTG', 'ALGM',
        'WOLF', 'SLAB', 'CRUS', 'SITM', 'LSCC',

        # Healthcare - Pharma & Biotech (50 stocks)
        'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'REGN', 'VRTX', 'HUM', 'ISRG', 'ZTS',
        'BIIB', 'MRNA', 'ILMN', 'IQV', 'BSX', 'MDT', 'SYK', 'EW', 'IDXX', 'HCA',
        'A', 'ALGN', 'ALNY', 'BAX', 'BDX', 'BIO', 'CNC', 'CTLT', 'DGX', 'DVA',
        'EXAS', 'GEHC', 'HOLX', 'HSIC', 'INCY', 'LH', 'MCK', 'MOH', 'PODD', 'RMD',

        # Financials - Banks (40 stocks)
        'JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SCHW', 'AXP', 'SPGI',
        'CB', 'MMC', 'PGR', 'TFC', 'USB', 'PNC', 'COF', 'BK', 'AIG', 'MET',
        'CME', 'ICE', 'MCO', 'AON', 'TRV', 'ALL', 'AFL', 'PRU', 'HIG', 'CINF',
        'BRO', 'L', 'GL', 'WRB', 'RJF', 'NTRS', 'CFG', 'HBAN', 'RF', 'KEY',

        # Financials - Insurance & Asset Management (20 stocks)
        'BRK.B', 'BLK', 'TROW', 'BEN', 'IVZ', 'STT', 'AMG', 'SEIC', 'EVR', 'PFG',
        'FNF', 'FAF', 'JKHY', 'CBOE', 'NDAQ', 'MKTX', 'MSCI', 'FDS', 'TW', 'VIRT',

        # Consumer Discretionary - Retail (40 stocks)
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG',
        'MAR', 'GM', 'F', 'ABNB', 'CMG', 'YUM', 'DRI', 'ROST', 'DG', 'ULTA',
        'EBAY', 'ETSY', 'W', 'BBY', 'DKS', 'FIVE', 'OLLI', 'BJ', 'BBWI', 'AEO',
        'ANF', 'BOOT', 'BWA', 'CRI', 'DDS', 'FL', 'GES', 'GPS', 'KSS', 'M',

        # Consumer Discretionary - Auto & Leisure (25 stocks)
        'TSLA', 'GM', 'F', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'HMC', 'TM',
        'RACE', 'STLA', 'CCL', 'RCL', 'NCLH', 'LVS', 'WYNN', 'MGM', 'CZR', 'PENN',
        'DKNG', 'FLUT', 'BALY', 'RSI', 'LYV',

        # Consumer Staples (35 stocks)
        'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'CL', 'MDLZ', 'KMB',
        'GIS', 'KHC', 'TSN', 'HSY', 'K', 'CLX', 'SJM', 'CPB', 'CAG', 'HRL',
        'MKC', 'CHD', 'TAP', 'STZ', 'BF.B', 'SAM', 'KDP', 'MNST', 'CELH', 'KR',
        'SYY', 'COKE', 'FLO', 'INGR', 'POST',

        # Energy (35 stocks)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
        'WMB', 'KMI', 'BKR', 'HES', 'DVN', 'FANG', 'MRO', 'APA', 'CTRA', 'OVV',
        'NOV', 'FTI', 'CHX', 'RIG', 'VAL', 'PR', 'EQT', 'AR', 'MTDR', 'SM',
        'MGY', 'CNX', 'RRC', 'CIVI', 'CLB',

        # Industrials (50 stocks)
        'UNP', 'HON', 'RTX', 'UPS', 'CAT', 'DE', 'BA', 'LMT', 'GE', 'MMM',
        'FDX', 'NSC', 'EMR', 'ETN', 'ITW', 'PH', 'WM', 'CSX', 'NOC', 'GD',
        'PCAR', 'JCI', 'CARR', 'OTIS', 'TT', 'IR', 'FAST', 'ODFL', 'CHRW', 'JBHT',
        'EXPD', 'XPO', 'HUBG', 'GWW', 'WCC', 'DY', 'ALLE', 'BLDR', 'FBIN', 'VMI',
        'MLM', 'GNRC', 'AIT', 'AAON', 'ACM', 'ACA', 'AGCO', 'ALK', 'ARCB', 'B',

        # Communication Services (30 stocks)
        'META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'EA',
        'TTWO', 'RBLX', 'U', 'PINS', 'SNAP', 'SPOT', 'MTCH', 'BMBL', 'YELP', 'ZG',
        'ROKU', 'PARA', 'WBD', 'FOXA', 'FOX', 'NWSA', 'NWS', 'NYT', 'OMC', 'IPG',

        # Utilities (25 stocks)
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ES', 'ED',
        'PEG', 'EIX', 'WEC', 'AWK', 'DTE', 'PPL', 'FE', 'CMS', 'CNP', 'ATO',
        'NI', 'LNT', 'EVRG', 'PNW', 'OGE',

        # Real Estate - REITs (35 stocks)
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'VTR', 'SBAC', 'INVH', 'ARE', 'MAA', 'DOC', 'UDR', 'ESS', 'KIM',
        'REG', 'FRT', 'BXP', 'VNO', 'SLG', 'HST', 'RHP', 'CPT', 'ELS', 'AMH',
        'CUBE', 'REXR', 'FR', 'STAG', 'TRNO',

        # Materials (30 stocks)
        'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'VMC',
        'MLM', 'ALB', 'CTVA', 'EMN', 'MOS', 'CE', 'FMC', 'IFF', 'PKG', 'AMCR',
        'IP', 'SEE', 'AVY', 'BALL', 'CCK', 'NEM', 'GOLD', 'WPM', 'FNV', 'SCCO',

        # Payment/Fintech (20 stocks)
        'V', 'MA', 'PYPL', 'ADP', 'FIS', 'FISV', 'GPN', 'SQ', 'COIN', 'SOFI',
        'AFRM', 'UPST', 'LC', 'NU', 'HOOD', 'MELI', 'PAGS', 'STNE', 'PAYX', 'FLYW',

        # Aerospace & Defense (15 stocks)
        'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'HWM', 'TXT', 'HII',
        'AVAV', 'KTOS', 'AJRD', 'CW', 'SPR',

        # Emerging Growth & Innovation (20 stocks)
        'PLTR', 'IONQ', 'RKLB', 'SPCE', 'OPEN', 'DASH', 'UBER', 'LYFT', 'CVNA', 'CHWY',
        'CHEWY', 'W', 'FVRR', 'UPWK', 'ZI', 'DOCN', 'APPS', 'BIGC', 'SHOP', 'MELI'
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

#==================== ARIMA FUNCTIONS ====================

@st.cache_data(ttl=3600)
def fit_arima_model(data, auto=True, order=(1,1,1)):
    """Fit ARIMA model with auto parameter selection"""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        if auto:
            # Try pmdarima if available, otherwise use simple grid search
            try:
                from pmdarima import auto_arima
                # Auto ARIMA with seasonal=False for speed
                model = auto_arima(data, seasonal=False, stepwise=True,
                                 suppress_warnings=True, error_action='ignore',
                                 max_p=5, max_d=2, max_q=5, trace=False)
                fitted_model = model
                best_order = model.order
                aic = model.aic()
                bic = model.bic()
            except ImportError:
                # Fallback: simple grid search with statsmodels
                best_aic = np.inf
                best_order = (1, 1, 1)
                best_model = None

                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                temp_model = ARIMA(data, order=(p, d, q))
                                temp_fit = temp_model.fit()
                                if temp_fit.aic < best_aic:
                                    best_aic = temp_fit.aic
                                    best_order = (p, d, q)
                                    best_model = temp_fit
                            except:
                                continue

                fitted_model = best_model
                aic = best_model.aic
                bic = best_model.bic
        else:
            # Manual ARIMA with specified order
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            best_order = order
            aic = fitted_model.aic
            bic = fitted_model.bic

        return {
            'model': fitted_model,
            'order': best_order,
            'aic': aic,
            'bic': bic,
            'residuals': fitted_model.resid
        }
    except Exception as e:
        st.error(f"ARIMA fitting error: {str(e)}")
        return None

def forecast_arima(fitted_model, steps=30, alpha=0.05):
    """Generate ARIMA forecasts with confidence intervals"""
    try:
        # Check if it's a pmdarima model or statsmodels model
        if hasattr(fitted_model, 'predict') and hasattr(fitted_model, 'n_periods'):
            # pmdarima model
            forecast = fitted_model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
            if isinstance(forecast, tuple):
                predictions = forecast[0]
                conf_int = forecast[1]
            else:
                predictions = forecast
                conf_int = None
        else:
            # statsmodels ARIMA model
            forecast_result = fitted_model.get_forecast(steps=steps)
            predictions = forecast_result.predicted_mean
            conf_int_df = forecast_result.conf_int(alpha=alpha)
            conf_int = conf_int_df.values if conf_int_df is not None else None

        return {
            'forecast': predictions,
            'conf_int': conf_int,
            'lower': conf_int[:, 0] if conf_int is not None else None,
            'upper': conf_int[:, 1] if conf_int is not None else None
        }
    except Exception as e:
        st.error(f"ARIMA forecast error: {str(e)}")
        return None

def arima_diagnostics(residuals):
    """Generate ARIMA diagnostic statistics"""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy.stats import jarque_bera, normaltest

        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)

        # Normality tests
        jb_stat, jb_pvalue = jarque_bera(residuals)

        # Mean and std of residuals
        residual_mean = residuals.mean()
        residual_std = residuals.std()

        return {
            'ljung_box': lb_test,
            'jb_statistic': jb_stat,
            'jb_pvalue': jb_pvalue,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'is_normal': jb_pvalue > 0.05
        }
    except Exception as e:
        return {'error': str(e)}

#==================== ARCH/GARCH FUNCTIONS ====================

@st.cache_data(ttl=3600)
def fit_garch_model(returns, p=1, q=1, model_type='GARCH'):
    """Fit GARCH model to returns data"""
    try:
        from arch import arch_model

        # Remove mean from returns (assuming zero mean)
        returns_pct = returns * 100  # ARCH models work better with percentage returns

        # Fit GARCH model
        model = arch_model(returns_pct, vol=model_type, p=p, q=q)
        fitted_model = model.fit(disp='off', show_warning=False)

        return {
            'model': fitted_model,
            'params': fitted_model.params,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'conditional_volatility': fitted_model.conditional_volatility,
            'residuals': fitted_model.resid
        }
    except Exception as e:
        st.error(f"GARCH fitting error: {str(e)}")
        return None

def forecast_volatility(fitted_garch, horizon=30):
    """Forecast volatility using fitted GARCH model"""
    try:
        # Generate volatility forecast
        forecast = fitted_garch.forecast(horizon=horizon)

        # Extract variance forecasts and convert to volatility
        variance_forecast = forecast.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast)

        return {
            'volatility': volatility_forecast,
            'variance': variance_forecast,
            'horizon': horizon
        }
    except Exception as e:
        st.error(f"Volatility forecast error: {str(e)}")
        return None

def garch_var(returns, fitted_garch, confidence=0.95):
    """Calculate VaR using GARCH volatility forecast"""
    try:
        from scipy.stats import norm

        # Get 1-day ahead volatility forecast
        forecast = fitted_garch.forecast(horizon=1)
        volatility = np.sqrt(forecast.variance.values[-1, 0]) / 100  # Convert back from percentage

        # Calculate VaR assuming normal distribution
        z_score = norm.ppf(1 - confidence)
        var = z_score * volatility

        return var
    except Exception as e:
        return None

def extract_volatility_regimes(conditional_vol, threshold_percentile=75):
    """Identify high and low volatility regimes"""
    try:
        threshold = np.percentile(conditional_vol, threshold_percentile)

        high_vol_periods = conditional_vol > threshold
        low_vol_periods = conditional_vol <= np.percentile(conditional_vol, 25)

        return {
            'threshold': threshold,
            'high_vol_periods': high_vol_periods,
            'low_vol_periods': low_vol_periods,
            'high_vol_mean': conditional_vol[high_vol_periods].mean(),
            'low_vol_mean': conditional_vol[low_vol_periods].mean()
        }
    except Exception as e:
        return None

#==================== COPULA FUNCTIONS ====================

def fit_copula(returns1, returns2, copula_type='gaussian'):
    """Fit copula to bivariate returns data"""
    try:
        from copulas.bivariate import Gaussian, Clayton, Gumbel, Frank
        from scipy.stats import t as student_t

        # Convert to uniform marginals using empirical CDF
        from scipy.stats import rankdata
        n = len(returns1)
        u1 = rankdata(returns1) / (n + 1)
        u2 = rankdata(returns2) / (n + 1)

        # Create dataframe for copulas library
        data = pd.DataFrame({'u1': u1, 'u2': u2})

        # Fit copula based on type
        if copula_type.lower() == 'gaussian':
            copula = Gaussian()
        elif copula_type.lower() == 'clayton':
            copula = Clayton()
        elif copula_type.lower() == 'gumbel':
            copula = Gumbel()
        elif copula_type.lower() == 'frank':
            copula = Frank()
        else:
            copula = Gaussian()

        copula.fit(data)

        return {
            'copula': copula,
            'type': copula_type,
            'u1': u1,
            'u2': u2,
            'params': copula.to_dict()
        }
    except Exception as e:
        st.error(f"Copula fitting error: {str(e)}")
        return None

def calculate_tail_dependence(returns1, returns2, copula_type='gaussian'):
    """Calculate upper and lower tail dependence coefficients"""
    try:
        from scipy.stats import spearmanr, kendalltau

        # Spearman's rho and Kendall's tau
        rho, _ = spearmanr(returns1, returns2)
        tau, _ = kendalltau(returns1, returns2)

        # Theoretical tail dependence for common copulas
        if copula_type.lower() == 'gaussian':
            upper_tail = 0  # Gaussian has no tail dependence
            lower_tail = 0
        elif copula_type.lower() == 't':
            # For t-copula, both tails have dependence (symmetric)
            # This is a simplified approximation
            upper_tail = tau  # Placeholder
            lower_tail = tau
        elif copula_type.lower() == 'clayton':
            # Clayton has lower tail dependence
            theta = 2 * tau / (1 - tau) if tau < 1 else 1
            lower_tail = 2 ** (-1/theta) if theta > 0 else 0
            upper_tail = 0
        elif copula_type.lower() == 'gumbel':
            # Gumbel has upper tail dependence
            theta = 1 / (1 - tau) if tau < 1 else 1
            upper_tail = 2 - 2 ** (1/theta)
            lower_tail = 0
        else:
            upper_tail = 0
            lower_tail = 0

        return {
            'spearman_rho': rho,
            'kendall_tau': tau,
            'upper_tail': upper_tail,
            'lower_tail': lower_tail
        }
    except Exception as e:
        return {'error': str(e)}

def copula_var(portfolio_returns, returns_matrix, copula_type='gaussian', confidence=0.95, simulations=10000):
    """Calculate portfolio VaR using copula simulation"""
    try:
        from copulas.multivariate import GaussianMultivariate

        # Fit multivariate copula
        copula = GaussianMultivariate()
        copula.fit(returns_matrix)

        # Generate scenarios
        samples = copula.sample(simulations)

        # Calculate portfolio returns for each scenario
        weights = np.ones(returns_matrix.shape[1]) / returns_matrix.shape[1]  # Equal weights
        simulated_returns = (samples * weights).sum(axis=1)

        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence) * 100)

        return var
    except Exception as e:
        return None

def simulate_copula(copula, n_samples=1000):
    """Generate samples from fitted copula"""
    try:
        samples = copula.sample(n_samples)
        return samples
    except Exception as e:
        return None

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

#==================== SIDEBAR ====================

# Logo and branding
st.sidebar.image("hedge_above_logo.png", use_container_width=True)
st.sidebar.markdown(
    "<p style='text-align: center; font-family: monospace; color: #888; font-size: 0.9rem;'>Rise Above the Status Quo</p>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Stock Screener", "Portfolio Builder", "Risk Analytics", "Portfolio Optimization", "AI Predictions"]
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
    st.success("💡 **Get Started:** Select 'Stock Screener' to find stocks, then build your portfolio!")

#==================== STOCK SCREENER ====================

elif page == "Stock Screener":
    st.header("🔍 Stock Screener")

    st.subheader("S&P 500 Universe (~200+ stocks)")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        sector_filter = st.multiselect(
            "Sector",
            ["Technology", "Healthcare", "Financials", "Consumer Discretionary", "Consumer Staples",
             "Energy", "Industrials", "Communication Services", "Utilities", "Real Estate",
             "Materials", "All"],
            default=["All"]
        )

    with col2:
        cap_filter = st.multiselect(
            "Market Cap",
            ["Mega Cap (>$200B)", "Large Cap ($10B-$200B)", "Mid Cap ($2B-$10B)", "All"],
            default=["All"]
        )

    with col3:
        results_limit = st.slider("Max Results to Display", 10, 100, 30, step=10)

    # Get tickers
    all_tickers = get_sp500_tickers()

    # Apply sector filter (will use real sector data from yfinance)
    st.markdown("---")

    if len(all_tickers) > 0:
        # Fetch data for tickers
        with st.spinner(f"Fetching data for up to {results_limit} stocks..."):
            screener_data = []
            for ticker in all_tickers[:results_limit]:  # Limit API calls
                stock_info = get_stock_info(ticker)
                if stock_info:
                    # Apply filters
                    if "All" not in sector_filter:
                        if stock_info['sector'] not in sector_filter:
                            continue

                    screener_data.append({
                        'Symbol': ticker,
                        'Company': stock_info['name'],
                        'Sector': stock_info['sector'],
                        'Price': stock_info['current_price'],
                        'Market Cap': stock_info['market_cap'],
                        'P/E': stock_info['pe_ratio'] if stock_info['pe_ratio'] else None,
                        'Div Yield %': stock_info['dividend_yield'] * 100 if stock_info['dividend_yield'] else 0
                    })

        if screener_data:
            screener_df = pd.DataFrame(screener_data)

            # Display count by sector
            st.subheader(f"📊 Found {len(screener_df)} stocks")

            sector_counts = screener_df['Sector'].value_counts()
            col1, col2 = st.columns([2, 1])

            with col1:
                # Main screener table
                st.dataframe(
                    screener_df.style.format({
                        'Price': '${:.2f}',
                        'Market Cap': lambda x: f"${x/1e9:.2f}B" if x > 0 else "N/A",
                        'P/E': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'Div Yield %': '{:.2f}%'
                    }),
                    use_container_width=True,
                    height=600
                )

            with col2:
                st.write("**Sector Distribution:**")
                for sector, count in sector_counts.items():
                    st.write(f"• {sector}: {count}")

            # Quick add to portfolio
            st.markdown("---")
            st.subheader("➕ Quick Add to Portfolio")
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

            if st.button("➕ Add to Portfolio", type="primary", use_container_width=True, key="screener_add"):
                if quick_symbol not in st.session_state.portfolio['Symbol'].values:
                    new_row = pd.DataFrame({
                        'Symbol': [quick_symbol],
                        'Shares': [quick_shares],
                        'Avg Price': [quick_avg_price]
                    })
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    st.success(f"✅ Added {quick_shares} shares of {quick_symbol} ({selected_company}) at ${quick_avg_price:.2f}")
                    st.balloons()
                else:
                    st.warning(f"❌ {quick_symbol} already in portfolio. Use Portfolio Builder to edit it.")
        else:
            st.info("No stocks match your filter criteria. Try adjusting filters or increasing results limit.")
    else:
        st.error("Could not load stock list")

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

            # Copula Analysis
            st.subheader("📊 Tail Dependence & Copula Analysis")

            if len(symbols) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    asset1 = st.selectbox("Asset 1", symbols, key='copula_asset1')
                with col2:
                    asset2 = st.selectbox("Asset 2", [s for s in symbols if s != asset1], key='copula_asset2')

                copula_type = st.selectbox(
                    "Copula Type",
                    ["Gaussian", "t-Copula", "Clayton", "Gumbel"],
                    help="Select copula family for dependence modeling"
                )

                if st.button("Analyze Tail Dependence"):
                    with st.spinner("Fitting copula model..."):
                        returns1 = returns[asset1].dropna()
                        returns2 = returns[asset2].dropna()

                        # Align the two return series
                        common_index = returns1.index.intersection(returns2.index)
                        returns1 = returns1.loc[common_index]
                        returns2 = returns2.loc[common_index]

                        # Calculate tail dependence
                        tail_dep = calculate_tail_dependence(returns1, returns2, copula_type.lower().replace('-', ''))

                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Spearman's ρ", f"{tail_dep['spearman_rho']:.3f}")
                        with col2:
                            st.metric("Kendall's τ", f"{tail_dep['kendall_tau']:.3f}")
                        with col3:
                            st.metric("Upper Tail λU", f"{tail_dep['upper_tail']:.3f}")
                        with col4:
                            st.metric("Lower Tail λL", f"{tail_dep['lower_tail']:.3f}")

                        st.info(f"""
                        **Interpretation:**
                        - **Spearman's ρ** and **Kendall's τ**: Rank correlations (0 = independent, ±1 = perfect dependence)
                        - **Upper Tail Dependence (λU)**: Probability of joint extreme positive returns
                        - **Lower Tail Dependence (λL)**: Probability of joint extreme negative returns (crashes)
                        - **{copula_type}**: {
                            'No tail dependence' if copula_type == 'Gaussian' else
                            'Symmetric tail dependence' if 't' in copula_type else
                            'Lower tail dependence (crash risk)' if copula_type == 'Clayton' else
                            'Upper tail dependence (boom risk)'
                        }
                        """)

                        # Scatter plot with copula fit
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=returns1,
                            y=returns2,
                            mode='markers',
                            marker=dict(size=5, opacity=0.6, color='blue'),
                            name='Returns'
                        ))
                        fig.update_layout(
                            title=f"{asset1} vs {asset2} Returns with {copula_type}",
                            xaxis_title=f"{asset1} Returns",
                            yaxis_title=f"{asset2} Returns",
                            hovermode='closest'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("Add at least 2 assets to analyze tail dependence")

            st.markdown("---")

            # GARCH Volatility Forecasting
            st.subheader("🔥 GARCH Volatility Forecasting & Regimes")

            if len(portfolio_returns) > 100:  # Need enough data for GARCH
                col1, col2 = st.columns(2)

                with col1:
                    forecast_horizon = st.slider("Forecast Horizon (days)", 5, 60, 30)
                with col2:
                    garch_p = st.selectbox("GARCH p (lag)", [1, 2], index=0)
                    garch_q = st.selectbox("GARCH q (lag)", [1, 2], index=0)

                if st.button("Run GARCH Analysis"):
                    with st.spinner("Fitting GARCH model..."):
                        # Fit GARCH model to portfolio returns
                        garch_result = fit_garch_model(portfolio_returns, p=garch_p, q=garch_q)

                        if garch_result:
                            # Display model statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("AIC", f"{garch_result['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{garch_result['bic']:.2f}")
                            with col3:
                                current_vol = garch_result['conditional_volatility'].iloc[-1] / 100
                                st.metric("Current Volatility", f"{current_vol*np.sqrt(252)*100:.2f}%")

                            # Forecast volatility
                            vol_forecast = forecast_volatility(garch_result['model'], horizon=forecast_horizon)

                            if vol_forecast:
                                # Plot conditional volatility and forecast
                                fig = go.Figure()

                                # Historical conditional volatility
                                fig.add_trace(go.Scatter(
                                    x=portfolio_returns.index,
                                    y=garch_result['conditional_volatility'] / 100 * np.sqrt(252) * 100,
                                    mode='lines',
                                    name='Conditional Volatility',
                                    line=dict(color='orange', width=2)
                                ))

                                # Forecast
                                forecast_dates = pd.date_range(
                                    start=portfolio_returns.index[-1],
                                    periods=forecast_horizon + 1,
                                    freq='D'
                                )[1:]

                                fig.add_trace(go.Scatter(
                                    x=forecast_dates,
                                    y=vol_forecast['volatility'] * np.sqrt(252),
                                    mode='lines',
                                    name='Volatility Forecast',
                                    line=dict(color='red', width=2, dash='dash')
                                ))

                                fig.update_layout(
                                    title=f"GARCH({garch_p},{garch_q}) Volatility: Historical & Forecast",
                                    xaxis_title="Date",
                                    yaxis_title="Annualized Volatility (%)",
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Volatility regimes
                                regimes = extract_volatility_regimes(garch_result['conditional_volatility'])

                                if regimes:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("High Vol Regime Avg", f"{regimes['high_vol_mean']/100*np.sqrt(252)*100:.2f}%")
                                    with col2:
                                        st.metric("Low Vol Regime Avg", f"{regimes['low_vol_mean']/100*np.sqrt(252)*100:.2f}%")

                                # GARCH-based VaR
                                garch_var_95 = garch_var(portfolio_returns, garch_result['model'], confidence=0.95)
                                if garch_var_95:
                                    st.info(f"**GARCH-Based 1-Day VaR (95%):** {garch_var_95:.4f} ({garch_var_95*100:.2f}%)")
            else:
                st.info("Add more portfolio history (>100 days) for GARCH analysis")

            st.markdown("---")

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

#==================== AI PREDICTIONS ====================

elif page == "AI Predictions":
    st.header("🤖 Time Series Forecasting & Volatility Prediction")

    # Model selection tabs
    tab1, tab2, tab3 = st.tabs(["📈 ARIMA Price Forecasts", "🔥 GARCH Volatility", "🔮 Combined Models"])

    #---------- TAB 1: ARIMA ----------
    with tab1:
        st.subheader("ARIMA Price & Returns Forecasting")

        col1, col2, col3 = st.columns(3)
        with col1:
            arima_ticker = st.text_input("Ticker Symbol", value="AAPL", key='arima_ticker')
        with col2:
            forecast_days = st.slider("Forecast Horizon (days)", 5, 90, 30, key='arima_days')
        with col3:
            forecast_type = st.selectbox("Forecast Type", ["Price", "Returns"], key='arima_type')

        auto_arima = st.checkbox("Auto ARIMA (find best p,d,q)", value=True, key='auto_arima')

        if not auto_arima:
            col1, col2, col3 = st.columns(3)
            with col1:
                p_order = st.number_input("p (AR order)", 0, 5, 1, key='arima_p')
            with col2:
                d_order = st.number_input("d (differencing)", 0, 2, 1, key='arima_d')
            with col3:
                q_order = st.number_input("q (MA order)", 0, 5, 1, key='arima_q')
        else:
            p_order, d_order, q_order = 1, 1, 1

        if st.button("Generate ARIMA Forecast", type="primary", key='run_arima'):
            with st.spinner(f"Fetching data and fitting ARIMA model for {arima_ticker}..."):
                # Fetch historical data
                hist_data = get_historical_data([arima_ticker], period='2y')

                if not hist_data.empty:
                    prices = hist_data[arima_ticker].dropna()

                    if forecast_type == "Price":
                        data_to_model = prices
                    else:  # Returns
                        data_to_model = prices.pct_change().dropna()

                    # Fit ARIMA
                    arima_result = fit_arima_model(data_to_model, auto=auto_arima, order=(p_order, d_order, q_order))

                    if arima_result:
                        # Display model info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model Order (p,d,q)", f"{arima_result['order']}")
                        with col2:
                            st.metric("AIC", f"{arima_result['aic']:.2f}")
                        with col3:
                            st.metric("BIC", f"{arima_result['bic']:.2f}")

                        # Generate forecast
                        forecast_result = forecast_arima(arima_result['model'], steps=forecast_days, alpha=0.05)

                        if forecast_result:
                            # Create forecast dates
                            last_date = data_to_model.index[-1]
                            forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='D')[1:]

                            # Convert returns to prices if needed
                            if forecast_type == "Returns":
                                # For returns forecast, show as percentage
                                forecast_values = forecast_result['forecast'] * 100
                                lower_bound = forecast_result['lower'] * 100 if forecast_result['lower'] is not None else None
                                upper_bound = forecast_result['upper'] * 100 if forecast_result['upper'] is not None else None
                                y_label = "Returns (%)"
                            else:
                                # Price forecast
                                forecast_values = forecast_result['forecast']
                                lower_bound = forecast_result['lower']
                                upper_bound = forecast_result['upper']
                                y_label = "Price ($)"

                            # Plot
                            fig = go.Figure()

                            # Historical data (last 90 days)
                            historical_plot = data_to_model.tail(90)
                            if forecast_type == "Returns":
                                historical_plot = historical_plot * 100

                            fig.add_trace(go.Scatter(
                                x=historical_plot.index,
                                y=historical_plot.values,
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue', width=2)
                            ))

                            # Forecast
                            fig.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=forecast_values,
                                mode='lines',
                                name='Forecast',
                                line=dict(color='red', width=2, dash='dash')
                            ))

                            # Confidence interval
                            if lower_bound is not None and upper_bound is not None:
                                fig.add_trace(go.Scatter(
                                    x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                                    y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255,0,0,0.1)',
                                    line=dict(color='rgba(255,0,0,0)'),
                                    name='95% Confidence Interval',
                                    showlegend=True
                                ))

                            fig.update_layout(
                                title=f"{arima_ticker} {forecast_type} Forecast (ARIMA{arima_result['order']})",
                                xaxis_title="Date",
                                yaxis_title=y_label,
                                hovermode='x unified',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Diagnostics
                            with st.expander("📊 Model Diagnostics"):
                                diagnostics = arima_diagnostics(arima_result['residuals'])

                                if 'error' not in diagnostics:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Residual Mean", f"{diagnostics['residual_mean']:.6f}")
                                        st.metric("Residual Std", f"{diagnostics['residual_std']:.6f}")
                                    with col2:
                                        st.metric("Jarque-Bera Statistic", f"{diagnostics['jb_statistic']:.2f}")
                                        st.metric("JB p-value", f"{diagnostics['jb_pvalue']:.4f}")

                                    if diagnostics['is_normal']:
                                        st.success("✅ Residuals appear normally distributed (JB test)")
                                    else:
                                        st.warning("⚠️ Residuals may not be normally distributed")

                                    # Ljung-Box test
                                    st.write("**Ljung-Box Test for Autocorrelation:**")
                                    st.dataframe(diagnostics['ljung_box'], use_container_width=True)
                                else:
                                    st.error(f"Diagnostic error: {diagnostics['error']}")
                else:
                    st.error(f"Could not fetch data for {arima_ticker}")

    #---------- TAB 2: GARCH ----------
    with tab2:
        st.subheader("GARCH Volatility Modeling & Forecasting")

        col1, col2, col3 = st.columns(3)
        with col1:
            garch_ticker = st.text_input("Ticker Symbol", value="AAPL", key='garch_ticker')
        with col2:
            garch_horizon = st.slider("Forecast Horizon (days)", 5, 90, 30, key='garch_horizon')
        with col3:
            model_type = st.selectbox("Model Type", ["GARCH", "EGARCH", "GJR-GARCH"], key='garch_model_type')

        col1, col2 = st.columns(2)
        with col1:
            garch_p_param = st.selectbox("GARCH p (lag)", [1, 2, 3], index=0, key='garch_p_param')
        with col2:
            garch_q_param = st.selectbox("GARCH q (lag)", [1, 2, 3], index=0, key='garch_q_param')

        if st.button("Run GARCH Forecast", type="primary", key='run_garch'):
            with st.spinner(f"Fitting GARCH model for {garch_ticker}..."):
                # Fetch data
                hist_data = get_historical_data([garch_ticker], period='2y')

                if not hist_data.empty:
                    prices = hist_data[garch_ticker].dropna()
                    returns = prices.pct_change().dropna()

                    # Fit GARCH
                    garch_result = fit_garch_model(returns, p=garch_p_param, q=garch_q_param, model_type=model_type)

                    if garch_result:
                        # Model stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", f"{model_type}({garch_p_param},{garch_q_param})")
                        with col2:
                            st.metric("AIC", f"{garch_result['aic']:.2f}")
                        with col3:
                            st.metric("BIC", f"{garch_result['bic']:.2f}")

                        # Forecast volatility
                        vol_forecast = forecast_volatility(garch_result['model'], horizon=garch_horizon)

                        if vol_forecast:
                            # Plot
                            fig = go.Figure()

                            # Historical conditional volatility (last 180 days)
                            hist_vol = garch_result['conditional_volatility'].tail(180) / 100 * np.sqrt(252) * 100
                            fig.add_trace(go.Scatter(
                                x=returns.index[-len(hist_vol):],
                                y=hist_vol.values,
                                mode='lines',
                                name='Historical Volatility',
                                line=dict(color='orange', width=2)
                            ))

                            # Forecast
                            forecast_dates = pd.date_range(
                                start=returns.index[-1],
                                periods=garch_horizon+1,
                                freq='D'
                            )[1:]

                            fig.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=vol_forecast['volatility'] * np.sqrt(252) * 100,
                                mode='lines',
                                name='Volatility Forecast',
                                line=dict(color='red', width=2, dash='dash')
                            ))

                            fig.update_layout(
                                title=f"{garch_ticker} Volatility Forecast ({model_type}({garch_p_param},{garch_q_param}))",
                                xaxis_title="Date",
                                yaxis_title="Annualized Volatility (%)",
                                hovermode='x unified',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Current vs forecast volatility
                            current_vol = garch_result['conditional_volatility'].iloc[-1] / 100 * np.sqrt(252) * 100
                            forecast_vol_avg = vol_forecast['volatility'].mean() * np.sqrt(252) * 100

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Volatility", f"{current_vol:.2f}%")
                            with col2:
                                st.metric("Avg Forecast Volatility", f"{forecast_vol_avg:.2f}%")
                            with col3:
                                change = forecast_vol_avg - current_vol
                                st.metric("Expected Change", f"{change:+.2f}%")

                            # Use case examples
                            with st.expander("💡 Use Cases for GARCH Volatility"):
                                st.write("""
                                **Options Pricing**: Use forecasted volatility as input for Black-Scholes model

                                **Risk Management**: Adjust position sizes based on volatility regime

                                **VaR Calculation**: GARCH-based VaR captures time-varying volatility

                                **Trading Signals**: High volatility = higher option premiums, tighter stops
                                """)
                else:
                    st.error(f"Could not fetch data for {garch_ticker}")

    #---------- TAB 3: COMBINED MODELS ----------
    with tab3:
        st.subheader("Combined ARIMA-GARCH Forecasting")

        st.info("""
        **Combined Model Approach:**
        1. **ARIMA** models the conditional mean (price trend)
        2. **GARCH** models the conditional variance (volatility clustering)
        3. Together they provide complete price distribution forecasts
        """)

        col1, col2 = st.columns(2)
        with col1:
            combined_ticker = st.text_input("Ticker Symbol", value="AAPL", key='combined_ticker')
        with col2:
            combined_horizon = st.slider("Forecast Horizon (days)", 5, 60, 30, key='combined_horizon')

        if st.button("Run Combined Forecast", type="primary", key='run_combined'):
            with st.spinner(f"Fitting combined ARIMA-GARCH model for {combined_ticker}..."):
                # Fetch data
                hist_data = get_historical_data([combined_ticker], period='2y')

                if not hist_data.empty:
                    prices = hist_data[combined_ticker].dropna()
                    returns = prices.pct_change().dropna()

                    # Fit ARIMA to returns
                    arima_result = fit_arima_model(returns, auto=True)

                    # Fit GARCH to residuals or returns
                    garch_result = fit_garch_model(returns, p=1, q=1)

                    if arima_result and garch_result:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**ARIMA Component:**")
                            st.metric("Order", f"{arima_result['order']}")
                            st.metric("AIC", f"{arima_result['aic']:.2f}")
                        with col2:
                            st.write("**GARCH Component:**")
                            st.metric("Order", f"(1,1)")
                            st.metric("AIC", f"{garch_result['aic']:.2f}")

                        # Generate forecasts
                        arima_forecast = forecast_arima(arima_result['model'], steps=combined_horizon)
                        garch_forecast = forecast_volatility(garch_result['model'], horizon=combined_horizon)

                        if arima_forecast and garch_forecast:
                            # Monte Carlo simulation with ARIMA mean and GARCH volatility
                            n_simulations = 1000
                            forecast_dates = pd.date_range(
                                start=returns.index[-1],
                                periods=combined_horizon+1,
                                freq='D'
                            )[1:]

                            simulated_prices = np.zeros((n_simulations, combined_horizon))
                            last_price = prices.iloc[-1]

                            for sim in range(n_simulations):
                                price_path = [last_price]
                                for i in range(combined_horizon):
                                    # Sample return from ARIMA forecast + GARCH volatility
                                    mean_return = arima_forecast['forecast'][i]
                                    vol = garch_forecast['volatility'][i] / 100  # Convert to decimal
                                    simulated_return = np.random.normal(mean_return, vol)
                                    next_price = price_path[-1] * (1 + simulated_return)
                                    price_path.append(next_price)
                                    simulated_prices[sim, i] = next_price

                            # Calculate percentiles
                            median_forecast = np.median(simulated_prices, axis=0)
                            lower_5 = np.percentile(simulated_prices, 5, axis=0)
                            upper_95 = np.percentile(simulated_prices, 95, axis=0)
                            lower_25 = np.percentile(simulated_prices, 25, axis=0)
                            upper_75 = np.percentile(simulated_prices, 75, axis=0)

                            # Plot
                            fig = go.Figure()

                            # Historical prices
                            fig.add_trace(go.Scatter(
                                x=prices.tail(90).index,
                                y=prices.tail(90).values,
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue', width=2)
                            ))

                            # Median forecast
                            fig.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=median_forecast,
                                mode='lines',
                                name='Median Forecast',
                                line=dict(color='red', width=2)
                            ))

                            # 90% confidence interval
                            fig.add_trace(go.Scatter(
                                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                                y=upper_95.tolist() + lower_5.tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.1)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name='90% Confidence',
                                showlegend=True
                            ))

                            # 50% confidence interval
                            fig.add_trace(go.Scatter(
                                x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                                y=upper_75.tolist() + lower_25.tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,0,0,0)'),
                                name='50% Confidence',
                                showlegend=True
                            ))

                            fig.update_layout(
                                title=f"{combined_ticker} Combined ARIMA-GARCH Price Forecast",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                hovermode='x unified',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Summary stats
                            final_median = median_forecast[-1]
                            final_lower = lower_5[-1]
                            final_upper = upper_95[-1]

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Price", f"${last_price:.2f}")
                            with col2:
                                st.metric(f"{combined_horizon}-Day Median", f"${final_median:.2f}")
                            with col3:
                                expected_return = (final_median - last_price) / last_price * 100
                                st.metric("Expected Return", f"{expected_return:+.2f}%")
                            with col4:
                                st.metric("90% Range", f"${final_lower:.2f} - ${final_upper:.2f}")
                else:
                    st.error(f"Could not fetch data for {combined_ticker}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "HedgeAbove v0.2.0 | Rise Above Market Uncertainty | "
    "<a href='https://github.com/lonespear/HedgeAbove'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
