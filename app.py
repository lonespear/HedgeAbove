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
    """Fetch comprehensive stock information with all fundamental metrics"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='5y')  # Extended history

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
            'book_value_per_share': info.get('bookValue', None),  # BVPS
            'revenue_per_share': info.get('revenuePerShare', None),
            'free_cash_flow_per_share': info.get('freeCashflow', 0) / info.get('sharesOutstanding', 1) if info.get('sharesOutstanding') else None,

            # Growth Metrics
            'earnings_growth': info.get('earningsGrowth', None),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None),
            'revenue_growth': info.get('revenueGrowth', None),

            # Profitability Metrics
            'profit_margin': info.get('profitMargins', None),
            'operating_margin': info.get('operatingMargins', None),
            'gross_margin': info.get('grossMargins', None),
            'ebitda_margin': info.get('ebitdaMargins', None),
            'roe': info.get('returnOnEquity', None),  # ROE
            'roa': info.get('returnOnAssets', None),  # ROA
            'roic': info.get('returnOnCapital', None),  # ROIC

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
            'short_percent': info.get('shortPercentOfFloat', None)
        }
        return stock_data
    except Exception as e:
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
        'CHEWY', 'W', 'FVRR', 'UPWK', 'ZI', 'DOCN', 'APPS', 'BIGC', 'SHOP', 'MELI',

        # Small-Cap Technology (100 stocks)
        'SMCI', 'DELL', 'HPQ', 'HPE', 'NTAP', 'WDC', 'STX', 'PSTG', 'CVLT', 'DBX',
        'BOX', 'MIME', 'TENB', 'VRNS', 'QLYS', 'RPD', 'PLAN', 'BLKB', 'BRZE', 'RELY',
        'ASAN', 'TASK', 'WDAY', 'PAYC', 'PCTY', 'PYCR', 'EEFT', 'EVTC', 'GDDY', 'WIX',
        'JFROG', 'AZPN', 'QTWO', 'SMAR', 'ENV', 'APPN', 'CWAN', 'NEWR', 'SUMO', 'SAIC',
        'LDOS', 'CACI', 'BAH', 'KBR', 'VRSK', 'TRU', 'GMED', 'TDC', 'FTV', 'ZBRA',
        'SWI', 'NATI', 'ROP', 'KEYS', 'TER', 'COHR', 'II', 'NOVT', 'LITE', 'VIAV',
        'FORM', 'DIOD', 'MKSI', 'ONTO', 'UCTT', 'PLAB', 'AOSL', 'CRUS', 'CEVA', 'XLNX',
        'MXIM', 'RMBS', 'SYNA', 'CCMP', 'SMTC', 'MTSI', 'COHU', 'LSCC', 'AMBA', 'MRVL',
        'CRUS', 'ACLS', 'MLAB', 'PI', 'AMBA', 'MPWR', 'POWI', 'VICR', 'ENPH', 'SEDG',
        'GNRC', 'RUN', 'NOVA', 'BLDR', 'ATKR', 'APH', 'TEL', 'GLW', 'JBL', 'FN',

        # Small-Cap Healthcare (100 stocks)
        'TECH', 'PEN', 'WAT', 'MTD', 'DXCM', 'RVTY', 'IQV', 'CRL', 'MEDP', 'SOLV',
        'VTRS', 'CORT', 'CPRX', 'PRGO', 'TEVA', 'SUPN', 'TMDX', 'KRYS', 'NTRA', 'ADMA',
        'HALO', 'RARE', 'FOLD', 'ION', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'VERV', 'PRME',
        'BLUE', 'SAGE', 'NBIX', 'SRPT', 'BMRN', 'SGEN', 'JAZZ', 'UTHR', 'HZNP', 'ALKS',
        'ACAD', 'PTCT', 'RGNX', 'TBPH', 'ARVN', 'ARWR', 'MDGL', 'ITCI', 'KRTX', 'SAVA',
        'AGIO', 'APLS', 'DNLI', 'FATE', 'NRIX', 'VRTX', 'MRTX', 'NVCR', 'GLPG', 'IONS',
        'EXEL', 'BPMC', 'BGNE', 'LEGN', 'YMAB', 'IMMU', 'ESPR', 'PBYI', 'INSM', 'CLDX',
        'GOSS', 'RXRX', 'SDGR', 'ALLO', 'BCYC', 'MNOV', 'CDMO', 'VCEL', 'VCYT', 'QGEN',
        'NVST', 'LMNX', 'MYGN', 'NTRA', 'GKOS', 'SLP', 'NEOG', 'GTHX', 'KRYS', 'AVNS',
        'OMER', 'PCRX', 'ANIP', 'LBPH', 'AMRX', 'COLL', 'ETNB', 'SGMO', 'EDIT', 'PACB',

        # Small-Cap Financials (100 stocks)
        'EWBC', 'PACW', 'WAL', 'SIVB', 'SBNY', 'FRC', 'CMA', 'ZION', 'SNV', 'ONB',
        'UMBF', 'OZK', 'UBSI', 'HWC', 'ASB', 'FHN', 'BKU', 'FIBK', 'WSFS', 'TCBI',
        'CADE', 'SFNC', 'CASH', 'ABCB', 'VLY', 'PB', 'BANR', 'CATY', 'CBU', 'FFIN',
        'TBBK', 'SRCE', 'BPOP', 'FBK', 'FULT', 'INDB', 'WAFD', 'PFS', 'UCBI', 'BANF',
        'LKFN', 'BHLB', 'SFBS', 'HTLF', 'HOMB', 'CVBF', 'CBSH', 'BOKF', 'WTFC', 'ONB',
        'ENVA', 'BRKL', 'Ryan', 'BGC', 'VIRT', 'LPLA', 'SF', 'PIPR', 'APAM', 'HLNE',
        'LAZ', 'PJT', 'EVR', 'RYAN', 'JEF', 'MORN', 'MC', 'OMF', 'VRNT', 'CACC',
        'COOP', 'WRLD', 'TRUP', 'LMND', 'ROOT', 'MTTR', 'KNSL', 'BHF', 'ORI', 'RNR',
        'AFG', 'WTM', 'SAFT', 'KMPR', 'PLMR', 'THG', 'UFCS', 'NAVG', 'JRVR', 'IGIC',
        'AGII', 'STC', 'EIG', 'AMSF', 'PRA', 'HRTG', 'UFCS', 'HALL', 'TRIN', 'ANAT',

        # Small-Cap Consumer (100 stocks)
        'WING', 'BLMN', 'TXRH', 'EAT', 'CAKE', 'PLAY', 'DENN', 'RUTH', 'FWRG', 'BWLD',
        'SHAK', 'NDLS', 'PZZA', 'DAVE', 'BJRI', 'DIN', 'CBRL', 'BLMN', 'PNRA', 'BWLD',
        'SON', 'SONC', 'JACK', 'HAYW', 'ARKO', 'CASY', 'LAD', 'ABG', 'SAH', 'AN',
        'PAG', 'GPI', 'CRMT', 'MNRO', 'AAP', 'AZO', 'ORLY', 'BBWI', 'URBN', 'AEO',
        'PSMT', 'ANF', 'TLYS', 'CHS', 'GCO', 'HIBB', 'BGFV', 'CTRN', 'SHOO', 'WWW',
        'BOOT', 'DECK', 'CROX', 'VFC', 'RL', 'PVH', 'HBI', 'GIL', 'SCVL', 'GIII',
        'MOV', 'EXPR', 'ZUMZ', 'TLRD', 'JWN', 'M', 'KSS', 'DDS', 'CRI', 'BURL',
        'FIVE', 'DG', 'DLTR', 'BIG', 'OLLI', 'PRTY', 'CONN', 'BBBY', 'TCS', 'BEDS',
        'ASO', 'BGFV', 'HIBB', 'DKS', 'FL', 'SCVL', 'PIR', 'SIG', 'WGO', 'THO',
        'CWH', 'LCI', 'LCII', 'PATK', 'BC', 'POWL', 'REVG', 'SHYF', 'HCSG', 'CASY',

        # Small-Cap Energy (75 stocks)
        'PBF', 'DK', 'CIVI', 'CRC', 'CPE', 'WTI', 'OAS', 'AROC', 'VTLE', 'GPOR',
        'REI', 'TALO', 'CDEV', 'CLR', 'MTDR', 'FANG', 'PR', 'MUR', 'NBR', 'HP',
        'NINE', 'SWN', 'RRC', 'CNX', 'AR', 'CTRA', 'MGY', 'REPX', 'CRGY', 'GPRK',
        'LPI', 'GPRE', 'REX', 'VVV', 'AMPY', 'GRNT', 'ESTE', 'PTEN', 'PUMP', 'LBRT',
        'NEX', 'WTTR', 'TDW', 'WFRD', 'SDRL', 'VAL', 'TRGP', 'ENLC', 'PAGP', 'USAC',
        'GEL', 'DKL', 'EPD', 'ET', 'MMP', 'PAA', 'WES', 'AM', 'HESM', 'CEQP',
        'ENLC', 'NGL', 'SUN', 'SHLX', 'MPLX', 'PSX', 'NS', 'CQP', 'BP', 'E',
        'TOT', 'SHEL', 'ENB', 'TRP', 'CNQ',

        # Small-Cap Industrials (100 stocks)
        'JBHT', 'LSTR', 'KNX', 'SAIA', 'ARCB', 'WERN', 'ODFL', 'XPO', 'CVLG', 'YRCW',
        'GNK', 'SBLK', 'INSW', 'EGLE', 'SHIP', 'CMRE', 'EDRY', 'GOGL', 'NMM', 'SB',
        'TGH', 'MATX', 'KEX', 'HUBG', 'SNDR', 'FWRD', 'ECHO', 'MRTN', 'ULH', 'HTLD',
        'SNCY', 'RXO', 'GXO', 'JBLU', 'AAL', 'UAL', 'DAL', 'LUV', 'ALK', 'SAVE',
        'HA', 'SKYW', 'MESA', 'ATSG', 'AAWW', 'CYRX', 'ARCB', 'SNDR', 'JOBY', 'ACHR',
        'BLDE', 'LILM', 'EH', 'EVEX', 'EVTL', 'GEV', 'LEV', 'REE', 'WKHS', 'RIDE',
        'FSR', 'GOEV', 'ARVL', 'MULN', 'ELMS', 'OWLT', 'HYZN', 'NKLA', 'HYLN', 'GP',
        'RDN', 'MTG', 'ESNT', 'NMIH', 'HCI', 'UVE', 'NODK', 'PRA', 'UFCS', 'KNSL',
        'AIT', 'DY', 'WSO', 'MSM', 'RBC', 'TILE', 'FLS', 'BMI', 'CR', 'PRIM',
        'ATKR', 'AAON', 'AOS', 'AWI', 'AZEK', 'BCC', 'BECN', 'BLD', 'BXC', 'CSWI',

        # Small-Cap Materials (75 stocks)
        'MP', 'LAC', 'ALB', 'SQM', 'LTHM', 'PLL', 'SGML', 'LITM', 'NOVRF', 'CMP',
        'SMG', 'TUP', 'CATO', 'GFF', 'KOP', 'HWKN', 'FUL', 'HXL', 'SLVM', 'KWR',
        'NGVT', 'TROX', 'IOSP', 'NEU', 'SXT', 'CSTM', 'OMI', 'GEF', 'SON', 'SLGN',
        'MERC', 'RPM', 'AXTA', 'HUN', 'OLN', 'TSE', 'KRA', 'VVV', 'CBT', 'CC',
        'WLK', 'IOSP', 'PCT', 'ESNT', 'NGVT', 'BCPC', 'FUL', 'GRA', 'CENX', 'KALU',
        'ATI', 'ZEUS', 'HCC', 'HAYN', 'SYNL', 'MTUS', 'CRS', 'TMST', 'WOR', 'MTRN',
        'CMC', 'CLF', 'STLD', 'RS', 'X', 'MT', 'TX', 'ASTL', 'HEES', 'NEWP',
        'USLM', 'TGLS', 'HL', 'AG', 'CDE', 'EGO', 'PAAS', 'GPL', 'SVM', 'NGD',
        'AUY', 'SSRM', 'KGC', 'IAG', 'BTG', 'VALE', 'RIO', 'BHP', 'SCCO', 'FCX',
        'TECK', 'HBM', 'CMCL', 'VEDL', 'GLNCY',

        # Small-Cap Real Estate (75 stocks)
        'VRE', 'STWD', 'BXMT', 'AGNC', 'NLY', 'TWO', 'MITT', 'ARR', 'CIM', 'MFA',
        'NYMT', 'DX', 'PMT', 'EARN', 'IVR', 'RC', 'GPMT', 'ARI', 'TRTX', 'ORC',
        'LADR', 'AAIC', 'GPMT', 'LOAN', 'RC', 'BRMK', 'RWT', 'CHMI', 'EFC', 'WMC',
        'NRZ', 'RITM', 'AAMC', 'ABR', 'AJX', 'ACRE', 'MITT', 'KREF', 'TPVG', 'OXSQ',
        'INN', 'PEB', 'RLJ', 'SHO', 'PK', 'AHT', 'APLE', 'CHH', 'XHR', 'DRH',
        'CLDT', 'FCPT', 'GTY', 'JBGS', 'KRC', 'CUZ', 'DEI', 'HIW', 'SLG', 'BXP',
        'PGRE', 'PDM', 'CLI', 'VNO', 'ESRT', 'NYC', 'SVC', 'ALEX', 'BDN', 'CIO',
        'GOOD', 'RMAX', 'OPI', 'SRC', 'AAT', 'ADC', 'AKR', 'BFS', 'BRX', 'CDR',
        'CTRE', 'ELME', 'EPRT', 'LAND', 'LXP', 'NNN', 'NTST', 'OUT', 'ROIC', 'RPT',
        'SITC', 'UE', 'UMH', 'VRE', 'WPC',

        # Small-Cap Utilities (50 stocks)
        'AVA', 'AGR', 'ALE', 'AQN', 'ARTNA', 'BKH', 'CPK', 'CWEN', 'CWEN.A', 'NWE',
        'NWN', 'MDU', 'MGE', 'MSEX', 'OTTR', 'PNM', 'POR', 'SJW', 'SR', 'SWX',
        'UTL', 'YORW', 'BIP', 'NEP', 'AY', 'AWR', 'CDZI', 'CWCO', 'ELPC', 'GNE',
        'NOVA', 'NWE', 'NWN', 'OTTR', 'PNM', 'POR', 'SJW', 'SR', 'SWX', 'UTL',
        'UGI', 'NFE', 'CWEN', 'TAC', 'DUK', 'FTS', 'BEPC', 'AEP', 'CEG', 'VST',

        # Small-Cap Communication & Media (50 stocks)
        'SSTK', 'TGNA', 'LEE', 'NXST', 'GTN', 'SCHL', 'MSGS', 'FUBO', 'SATS', 'EVER',
        'GOGO', 'IRDM', 'GILT', 'VSAT', 'IRDM', 'CMCSA', 'CHTR', 'CABO', 'LBRDA', 'LBRDK',
        'LILA', 'LILAK', 'SIRI', 'LSXMA', 'LSXMB', 'LSXMK', 'GSAT', 'ASTS', 'SPCE', 'RKLB',
        'MAXN', 'PUBM', 'MGNI', 'TTD', 'APPS', 'BIGC', 'CRTO', 'NCMI', 'IMAX', 'CNK',
        'RGC', 'MSGN', 'WMG', 'SPOT', 'BMBL', 'MTCH', 'IAC', 'ANGI', 'CARS', 'CVNA',

        # Micro-Cap & Emerging Stocks (200 stocks)
        'AEHR', 'CLOV', 'GTLB', 'IOT', 'LUNR', 'PL', 'PTON', 'BROS', 'GRND', 'GPRO',
        'BYND', 'OUST', 'LAZR', 'LIDR', 'INVZ', 'MVIS', 'VLDR', 'AEYE', 'OLED', 'KOPN',
        'VUZI', 'WIMI', 'HIMX', 'GRMN', 'WOLF', 'SEDG', 'ENPH', 'RUN', 'ARRY', 'CSIQ',
        'DQ', 'FSLR', 'JKS', 'MAXN', 'NOVA', 'SOL', 'SPWR', 'SHLS', 'VVPR', 'AMPS',
        'BE', 'CLNE', 'FCEL', 'GEVO', 'AMTX', 'BLDP', 'PLUG', 'HYSR', 'AMRC', 'FLNC',
        'QS', 'SES', 'ABML', 'CBAT', 'POLA', 'EOSE', 'FREYR', 'ENVX', 'PCVX', 'STEM',
        'BLNK', 'CHPT', 'EVGo', 'DCFC', 'VLTA', 'WBX', 'ALPP', 'AYRO', 'NWTN', 'SOLO',
        'LEV', 'GOEV', 'ARVL', 'FSR', 'RIDE', 'WKHS', 'GEV', 'ACTC', 'NGAC', 'CCIV',
        'PSNY', 'REE', 'INDI', 'ELMS', 'HYZN', 'PTRA', 'MPAA', 'EMBK', 'XL', 'PROTERRA',
        'BIRD', 'HIMS', 'BROS', 'CANO', 'DNA', 'NTLA', 'BEAM', 'CRSP', 'EDIT', 'VERV',
        'MASS', 'VCYT', 'FATE', 'BLUE', 'QURE', 'SGMO', 'CRIS', 'VKTX', 'VERU', 'NRIX',
        'ALLO', 'ABUS', 'ADAP', 'ADMA', 'ADPT', 'ADTX', 'ADVM', 'AGLE', 'AGRX', 'AIMD',
        'AKBA', 'AKRO', 'ALDX', 'ALEC', 'ALIM', 'ALLO', 'ALNY', 'ALVR', 'AMED', 'AMGN',
        'AMPH', 'ANGO', 'ANIP', 'ANPC', 'ANTE', 'APDN', 'APTO', 'APYX', 'ARDX', 'ARDS',
        'ARQT', 'ARWR', 'ASLN', 'ASND', 'ASRT', 'ASXC', 'ATNF', 'ATOS', 'ATRA', 'ATNM',
        'AVDL', 'AVEO', 'AVGR', 'AVIR', 'AVRO', 'AVXL', 'AXGN', 'AXLA', 'AXNX', 'AXSM',
        'AYTU', 'BCAB', 'BCDA', 'BCEL', 'BCLI', 'BCRX', 'BDSX', 'BDTX', 'BEAT', 'BFRI',
        'BHTG', 'BIOL', 'BIOX', 'BLRX', 'BMEA', 'BMRA', 'BNGO', 'BOLD', 'BPTH', 'BRTX',
        'BSGM', 'BSQR', 'BTAI', 'BVXV', 'BYSI', 'BZUN', 'CAPR', 'CARA', 'CARV', 'CATB',
        'CBAY', 'CBIO', 'CBRX', 'CCCC', 'CCXI', 'CDAK', 'CDMO', 'CDNA', 'CDTX', 'CDXC',
        'CDXS', 'CEMI', 'CENT', 'CERE', 'CERS', 'CGEN', 'CGEM', 'CHEK', 'CHMA', 'CHRS',

        # Additional Russell 2000 - Small-Cap Tech (100 stocks)
        'AAOI', 'AAON', 'ABCL', 'ABEO', 'ACIA', 'ACMR', 'ACRS', 'ADTN', 'ADUS', 'AEIS',
        'AEYE', 'AFRM', 'AIRC', 'AKAM', 'ALKT', 'ALRM', 'ALTR', 'AMBA', 'AMED', 'AMKR',
        'AMSC', 'AMWD', 'ANGI', 'ANNX', 'AOSL', 'APPF', 'ARAY', 'ARCE', 'ARCT', 'ARLO',
        'ARVL', 'ASYS', 'ATEC', 'ATEX', 'ATOM', 'ATTO', 'AUDC', 'AVAV', 'AVNW', 'AXTI',
        'AZTA', 'BBAI', 'BBCP', 'BCOR', 'BCOV', 'BELFB', 'BGNE', 'BILL', 'BKKT', 'BLBD',
        'BLFS', 'BLNK', 'BMBL', 'BNFT', 'BNSO', 'BPMC', 'BPOP', 'BRID', 'BRKR', 'BRKS',
        'BTBT', 'BTCT', 'BWMX', 'BYFC', 'CALX', 'CAMP', 'CAMT', 'CARB', 'CART', 'CASS',
        'CCRN', 'CCSI', 'CDLX', 'CDNS', 'CDZI', 'CEVA', 'CGNT', 'CHRS', 'CIEN', 'CIFR',
        'CIGI', 'CGNX', 'CLBT', 'CLFD', 'CLIR', 'CLOU', 'CLPT', 'CLRO', 'CLVT', 'CLWT',
        'CMCT', 'CMPR', 'CNCE', 'CNMD', 'CNOB', 'CNSL', 'CNST', 'CNXC', 'CNXN', 'COGT',

        # Additional Russell 2000 - Small-Cap Healthcare (100 stocks)
        'CLDX', 'CMPS', 'CNTA', 'COCP', 'COHN', 'COHR', 'COLB', 'COLL', 'CORT', 'COYA',
        'CPRX', 'CRBP', 'CRDF', 'CRNX', 'CRSP', 'CRTX', 'CRVS', 'CRWS', 'CSTL', 'CTHR',
        'CTMX', 'CTIC', 'CTSO', 'CTRN', 'CTXR', 'CUTR', 'CVAC', 'CVCO', 'CVGW', 'CVIG',
        'CVLT', 'CVRX', 'CWST', 'CXDO', 'CYCN', 'CYRX', 'CYTK', 'CZNC', 'DADA', 'DAIO',
        'DAWN', 'DBRG', 'DCGO', 'DCOM', 'DENN', 'DERM', 'DFIN', 'DGII', 'DGLY', 'DIOD',
        'DJCO', 'DLHC', 'DMAC', 'DMRC', 'DNLI', 'DOGZ', 'DOMO', 'DORM', 'Doug', 'DRCT',
        'DRMA', 'DRRX', 'DSGX', 'DSWL', 'DTIL', 'DXPE', 'DXYN', 'DYAI', 'DYNC', 'EAGL',
        'EARN', 'EARS', 'EAST', 'EBIX', 'EBON', 'ECHO', 'ECOR', 'ECPG', 'EDAP', 'EDBL',
        'EDIT', 'EDRY', 'EDSA', 'EDUC', 'EEFT', 'EFOI', 'EFSC', 'EGAN', 'EGBN', 'EGLE',
        'EGRX', 'EH', 'EHTH', 'EIGR', 'EKSO', 'ELDN', 'ELIO', 'ELMD', 'ELSE', 'ELTK',

        # Additional Russell 2000 - Small-Cap Financials (75 stocks)
        'EMBC', 'EMCF', 'EME', 'EMKR', 'EML', 'EMMA', 'EMMS', 'EMP', 'ENCP', 'ENFN',
        'ENG', 'ENIA', 'ENJY', 'ENLV', 'ENOB', 'ENOV', 'ENR', 'ENSC', 'ENTA', 'ENTG',
        'ENVA', 'ENVB', 'ENVI', 'ENVX', 'ENZC', 'EOLS', 'EOSE', 'EPAC', 'EPAM', 'EPAY',
        'EPIX', 'EPRT', 'EQBK', 'EQIX', 'EQOS', 'EQRX', 'ERAS', 'ERES', 'ERIC', 'ERIE',
        'ERII', 'ERIEY', 'EROS', 'ESBK', 'ESCA', 'ESEA', 'ESGR', 'ESGRP', 'ESGV', 'ESLT',
        'ESMT', 'ESNT', 'ESOA', 'ESPN', 'ESPR', 'ESQ', 'ESSA', 'ESSC', 'ESTA', 'ESTC',
        'ESTY', 'ESXB', 'ETAO', 'ETD', 'ETNB', 'ETON', 'ETSY', 'ETTX', 'EUDA', 'EURN',
        'EUSA', 'EVCM', 'EVGN', 'EVGO', 'EVGR', 'EVLO', 'EVLV', 'EVOK', 'EVRG', 'EVRI',

        # Additional Russell 2000 - Small-Cap Consumer (75 stocks)
        'EVTV', 'EVTC', 'EWTX', 'EWZS', 'EXAS', 'EXEL', 'EXFY', 'EXLS', 'EXOD', 'EXPD',
        'EXPE', 'EXPI', 'EXPO', 'EXTR', 'EYE', 'EYEN', 'EYEG', 'EYES', 'EYESW', 'EYPT',
        'EZFL', 'EZGO', 'EZPW', 'FAAR', 'FAAS', 'FAF', 'FALC', 'FAMI', 'FANG', 'FANH',
        'FARM', 'FARO', 'FAST', 'FATBB', 'FATE', 'FATP', 'FBNC', 'FBIZ', 'FBLG', 'FBMS',
        'FBNK', 'FBRT', 'FBRX', 'FCAP', 'FCBC', 'FCBP', 'FCCO', 'FCCY', 'FCEL', 'FCFS',
        'FCNCA', 'FCPT', 'FDBC', 'FDMT', 'FDUS', 'FEIM', 'FELE', 'FELP', 'FENC', 'FEND',
        'FENG', 'FERG', 'FEXD', 'FFBC', 'FFBW', 'FFG', 'FFIC', 'FFIE', 'FFIN', 'FFIV',
        'FFNW', 'FFWM', 'FGBI', 'FGBIP', 'FGEN', 'FGF', 'FGFPP', 'FGI', 'FGIWW', 'FGMC',

        # Additional Russell 2000 - Small-Cap Industrials (75 stocks)
        'FHB', 'FHI', 'FHLT', 'FIAC', 'FIBK', 'FICO', 'FINV', 'FINW', 'FISI', 'FITB',
        'FITBI', 'FITBO', 'FITBP', 'FIVN', 'FIXD', 'FIXX', 'FIZZ', 'FKWL', 'FLAG', 'FLDM',
        'FLEX', 'FLFV', 'FLIC', 'FLKS', 'FLLC', 'FLLCU', 'FLME', 'FLMN', 'FLMNW', 'FLNG',
        'FLNT', 'FLUX', 'FLWS', 'FLXN', 'FLXS', 'FLYW', 'FMAO', 'FMBH', 'FMBI', 'FMC',
        'FMIV', 'FMNB', 'FN', 'FNA', 'FNCB', 'FNCH', 'FNHC', 'FNJN', 'FNKO', 'FNLC',
        'FNRN', 'FNVT', 'FNWB', 'FNWD', 'FOCS', 'FOE', 'FOLD', 'FOMX', 'FONE', 'FOR',
        'FORD', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FOX', 'FOXA', 'FOXF', 'FOXW', 'FPAC',
        'FPAY', 'FPAYU', 'FPAYW', 'FPF', 'FPH', 'FPI', 'FRAF', 'FRBA', 'FRBK', 'FREE',

        # Additional Russell 2000 - Small-Cap Energy & Materials (65 stocks)
        'FREQ', 'FRES', 'FRGE', 'FRGAP', 'FRGT', 'FRHC', 'FRLA', 'FRLAU', 'FRLAW', 'FRM',
        'FRME', 'FRMEP', 'FRO', 'FRPH', 'FRPT', 'FRSX', 'FRZA', 'FSBC', 'FSBW', 'FSCO',
        'FSD', 'FSEA', 'FSFG', 'FSK', 'FSLR', 'FSLY', 'FSM', 'FSNB', 'FSP', 'FSR',
        'FSS', 'FSTR', 'FSV', 'FTAA', 'FTAAU', 'FTAAW', 'FTAI', 'FTCH', 'FTCI', 'FTCV',
        'FTCVU', 'FTCVW', 'FTDR', 'FTEK', 'FTF', 'FTFT', 'FTHM', 'FTHY', 'FTI', 'FTK',
        'FTLF', 'FTNT', 'FTRE', 'FTRP', 'FTSH', 'FTV', 'FUBO', 'FUEL', 'FUL', 'FULC',
        'FULT', 'FULTP', 'FUN', 'FUNC', 'FUND', 'FURY', 'FUSB', 'FUTU', 'FVCB', 'FVE',

        # Additional Russell 2000 - Micro-Cap Diversified (125 stocks)
        'FWAC', 'FWBI', 'FWONA', 'FWONK', 'FWRD', 'FXCO', 'FXLV', 'FXNC', 'FYLD', 'GABC',
        'GAIA', 'GAIN', 'GAINL', 'GAINM', 'GAINN', 'GAINO', 'GALT', 'GAM', 'GAMB', 'GAMC',
        'GAME', 'GAN', 'GANX', 'GASS', 'GATE', 'GATEU', 'GATEW', 'GBAB', 'GBCI', 'GBDC',
        'GBIO', 'GBLI', 'GBLIL', 'GBLIZ', 'GBNK', 'GBNY', 'GBRGR', 'GBRGU', 'GBRGW', 'GBS',
        'GBT', 'GCBC', 'GCMG', 'GCMGW', 'GCO', 'GCOR', 'GCP', 'GCT', 'GCTK', 'GCV',
        'GDEV', 'GDEVW', 'GDHG', 'GDNR', 'GDNRW', 'GDO', 'GDOC', 'GDOT', 'GDRX', 'GDS',
        'GDST', 'GDTC', 'GDYN', 'GDYNW', 'GECC', 'GECCM', 'GECCN', 'GECCO', 'GEF', 'GEFA',
        'GEG', 'GEHC', 'GEL', 'GEN', 'GENC', 'GENE', 'GENI', 'GENK', 'GEOS', 'GERN',
        'GES', 'GESCO', 'GETD', 'GETY', 'GEVO', 'GFAI', 'GFF', 'GFGD', 'GFGF', 'GFI',
        'GFL', 'GFOR', 'GFVD', 'GFX', 'GGAA', 'GGAAU', 'GGAAW', 'GGAL', 'GGE', 'GGMC',
        'GGMCU', 'GGMCW', 'GGN', 'GGOOW', 'GGR', 'GGROW', 'GGZ', 'GH', 'GHC', 'GHIX',
        'GHIXU', 'GHIXW', 'GHL', 'GHLD', 'GHRS', 'GHSI', 'GIAC', 'GIACU', 'GIACW', 'GIB',
        'GIFI', 'GIFT', 'GIII', 'GIIX', 'GIIXU', 'GIIXW', 'GIL',

        # Final Russell 2000 Additions - Mixed Sectors (100 stocks)
        'GILD', 'GILT', 'GILTI', 'GIS', 'GKOS', 'GL', 'GLAD', 'GLADD', 'GLBE', 'GLBS',
        'GLBZ', 'GLDD', 'GLDI', 'GLG', 'GLHA', 'GLHAU', 'GLHAW', 'GLMD', 'GLNG', 'GLOP',
        'GLOV', 'GLP', 'GLPG', 'GLPI', 'GLRE', 'GLSI', 'GLST', 'GLT', 'GLTO', 'GLUE',
        'GLXG', 'GLYC', 'GM', 'GMAB', 'GMBL', 'GMBLW', 'GMDA', 'GME', 'GMED', 'GMGI',
        'GMLP', 'GMLPP', 'GMRE', 'GMS', 'GMTX', 'GNCA', 'GNFT', 'GNK', 'GNL', 'GNLX',
        'GNMA', 'GNPX', 'GNRC', 'GNSS', 'GNTA', 'GNTX', 'GNTY', 'GNUS', 'GO', 'GOCCU',
        'GOCCW', 'GODN', 'GOEV', 'GOEVW', 'GOF', 'GOGL', 'GOGO', 'GOL', 'GOLD', 'GOLF',
        'GOOS', 'GORO', 'GORV', 'GOSS', 'GOTU', 'GOVX', 'GOVXW', 'GP', 'GPAC', 'GPACU',
        'GPACW', 'GPC', 'GPI', 'GPJA', 'GPK', 'GPL', 'GPMT', 'GPN', 'GPOR', 'GPRE',
        'GPRK', 'GPRO', 'GPS', 'GPTX', 'GPX', 'GRAB', 'GRABW', 'GRAL', 'GRBK', 'GRC',

        # ==================== INTERNATIONAL STOCKS ====================

        # Europe - Technology & Semiconductors (40 stocks)
        'ASML', 'SAP', 'SHOP', 'SE', 'SPOT', 'NICE', 'CYBR', 'CHKP', 'WIX', 'MNDY',
        'STM', 'ERIC', 'NOK', 'ARM', 'INFN', 'LITE', 'SWKS', 'SMCI', 'LOGI', 'LSCC',
        'SGMS', 'SSNLF', 'IFNNY', 'NOKIA', 'EADSY', 'SIEGY', 'TKOMY', 'TSM', 'UMC', 'ASX',
        'HIMX', 'SPIL', 'OLED', 'AU', 'HTHT', 'VNET', 'KC', 'TIGR', 'FUTU', 'UP',

        # Europe - Financials & Banking (50 stocks)
        'BCS', 'DB', 'CS', 'UBS', 'BBVA', 'SAN', 'INGA', 'ING', 'BNP', 'AXA',
        'SCOR', 'AEGN', 'AEGON', 'NN', 'ASR', 'CRDI', 'UCG', 'ISP', 'BPSO', 'BAMI',
        'BAMI', 'CABK', 'RBS', 'LYG', 'HSBC', 'VOD', 'BT', 'TEF', 'TI', 'ORAN',
        'FTE', 'VIV', 'EQNR', 'NG', 'SHEL', 'BP', 'TTE', 'RDSB', 'RDS.B', 'TOT',
        'ENI', 'REP', 'REPYY', 'GALP', 'OMV', 'PKN', 'LONN', 'MOWI', 'BAKKA', 'DNO',

        # Europe - Consumer & Retail (50 stocks)
        'NVO', 'NOVO', 'NOVOB', 'AZN', 'GSK', 'SNY', 'BAYRY', 'RHHBY', 'BAYN', 'NVS',
        'ROG', 'NESN', 'ADDYY', 'OR', 'MC', 'RMS', 'KER', 'CFR', 'BURBY', 'IDEXY',
        'HENKY', 'HEN3', 'BEI', 'LRLCY', 'LRLCF', 'LVMUY', 'PRDSY', 'LWAY', 'WPP', 'PUBGY',
        'REXR', 'FP', 'EDF', 'EDP', 'IBE', 'ELE', 'ENGI', 'VIE', 'EOAN', 'RWE',
        'E.ON', 'STLAM', 'STOXX', 'NOKIA', 'FIAT', 'STLA', 'RACE', 'VOW3', 'BMW', 'DAI',

        # Asia-Pacific - China (60 stocks)
        'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'TCOM', 'BILI', 'IQ', 'TME', 'HUYA',
        'DOYU', 'MOMO', 'YY', 'JMIA', 'WB', 'VIPS', 'ATHM', 'BZUN', 'DADA', 'DDL',
        'DAO', 'BEST', 'TUYA', 'RLX', 'GOTU', 'EDU', 'TAL', 'GOTU', 'DUO', 'LAIX',
        'TWOU', 'ZYXI', 'TEDU', 'YANG', 'YINN', 'KWEB', 'CQQQ', 'GXC', 'FXI', 'MCHI',
        'NIO', 'XPEV', 'LI', 'KNDI', 'NIU', 'SOLO', 'WKHS', 'IDEX', 'BLNK', 'SBE',
        'QS', 'GOEV', 'AYRO', 'KANDI', 'NKLA', 'RIDE', 'HYLN', 'CIIC', 'SBE', 'PSNY',

        # Asia-Pacific - India (40 stocks)
        'INFY', 'WIT', 'HDB', 'IBN', 'SIFY', 'REDF', 'REDY', 'TTM', 'VEDL', 'WNS',
        'YTRA', 'RDY', 'ICICI', 'AXISB', 'SBIN', 'HDFCB', 'KOTAKB', 'YESBK', 'PNB', 'BOB',
        'CANBK', 'IDBI', 'UNBNK', 'INDBNK', 'FED', 'DHFL', 'ING', 'PIN', 'ITC', 'HIND',
        'TATA', 'RLNC', 'BHARAT', 'ONGC', 'COAL', 'NTPC', 'POWERGD', 'IOC', 'BPCL', 'HPCL',

        # Asia-Pacific - South Korea (35 stocks)
        'TSM', 'SSNLF', 'LPL', 'SKM', 'KB', 'SHG', 'HMC', 'TM', 'PCRFY', 'HYMTF',
        'SMSN', 'LG', 'LGIH', 'LGLG', 'LGEL', 'LGCL', 'LGLD', 'HYSN', 'KIMTF', 'SSNGY',
        'KEP', 'PKX', 'SSL', 'SPOT', 'HYUD', 'KIA', 'VLKAF', 'SMAWF', 'POAHY', 'NCTY',
        'NAVER', 'KAKAO', 'COUPN', 'TCEHY', 'BABA',

        # Asia-Pacific - Japan (50 stocks)
        'SONY', 'TM', 'HMC', 'NSANY', 'NTDOY', 'FUJIY', 'HTHIY', 'SNEJF', 'MSBHF', 'MITSY',
        'MITSF', 'SMFG', 'MTU', 'MFG', 'MUFG', 'NMR', 'KB', 'SMFNF', 'MITSUBISHI', 'ITOCHU',
        'MARUY', 'SOMMY', 'CANNY', 'KDDIY', 'TOELY', 'FANUY', 'PCRFY', 'FUJHD', 'RICOY', 'SEKEY',
        'SHKLY', 'DNZOY', 'SZKMY', 'KAISY', 'AJINY', 'OLIMP', 'NPSNY', 'SHCAY', 'TAKAY', 'SYIEY',
        'AIQUY', 'DSKYY', 'OTSKY', 'SXRCY', 'RCRUY', 'KUBTY', 'YAMCY', 'MZDAY', 'DPSGY', 'KGFHY',

        # Latin America (40 stocks)
        'VALE', 'PBR', 'ITUB', 'BBD', 'ABEV', 'SBS', 'BSAC', 'BVN', 'GGAL', 'YPF',
        'TEO', 'TX', 'CIG', 'PAM', 'SID', 'CBD', 'ERJ', 'GOL', 'CIB', 'BSBR',
        'EBR', 'ELET', 'VIV', 'AMX', 'TV', 'TSU', 'TIMB', 'FMX', 'KOF', 'AC',
        'ASUR', 'GAP', 'OMA', 'PAC', 'VLRS', 'SU', 'QIWI', 'MAIL', 'YNDX', 'OZON',

        # Middle East & Africa (25 stocks)
        'TEVA', 'CHKP', 'CYBR', 'NICE', 'WIX', 'MNDY', 'FVRR', 'LMND', 'GLBE', 'MGIC',
        'TIGO', 'MTN', 'SBSA', 'JSE', 'IMPUY', 'ANGPY', 'SBSW', 'HGTY', 'GOLD', 'AU',
        'GFI', 'HMY', 'RGLD', 'NG', 'DRIP',

        # Canada (40 stocks)
        'SHOP', 'TD', 'RY', 'BNS', 'BMO', 'CM', 'ENB', 'CNQ', 'TRP', 'SU',
        'CNR', 'CP', 'ABX', 'GOLD', 'NEM', 'AEM', 'FNV', 'WPM', 'PAAS', 'EGO',
        'BB', 'LSPD', 'REAL', 'WELL', 'DOC', 'FOOD', 'QSR', 'RBI', 'MGA', 'ATD',
        'WCN', 'BEP', 'BEPC', 'AQN', 'HASI', 'NPI', 'BLX', 'BAM', 'BIP', 'BIPC',

        # Australia & New Zealand (30 stocks)
        'BHP', 'RIO', 'WES', 'CSL', 'CBA', 'NAB', 'ANZ', 'WBC', 'MQG', 'TLS',
        'WOW', 'WPL', 'STO', 'ORG', 'S32', 'FMG', 'NCM', 'EVN', 'NST', 'SFR',
        'RMD', 'COH', 'REA', 'SEK', 'XRO', 'APT', 'WTC', 'A2M', 'TWE', 'ALU',

        # Emerging Markets - Southeast Asia (35 stocks)
        'GRAB', 'SEA', 'BABA', 'BEKE', 'DIDI', 'TME', 'BGNE', 'VIPS', 'ZTO', 'YMM',
        'GDS', 'HTHT', 'IQ', 'BIDU', 'KC', 'GOTU', 'TAL', 'EDU', 'BEKE', 'DOYU',
        'HUYA', 'YY', 'MOMO', 'WB', 'BEST', 'TUYA', 'DAO', 'DADA', 'DDL', 'LU',
        'RLX', 'MOGU', 'TIGR', 'FUTU', 'UP',

        # Global ADRs - Telecommunications (25 stocks)
        'VOD', 'TEF', 'TI', 'VIV', 'ORAN', 'FTE', 'AMX', 'CHT', 'CHL', 'TU',
        'SKM', 'DCM', 'PHI', 'TEO', 'TIM', 'VIP', 'VIVHY', 'DTEGY', 'TMOBY', 'NCDX',
        'TELF', 'TLSNF', 'TLSN', 'TLSYY', 'BCE',

        # Global ADRs - Energy & Utilities (35 stocks)
        'EQNR', 'E', 'SHEL', 'BP', 'TTE', 'TOT', 'SU', 'CNQ', 'IMO', 'CVE',
        'ENB', 'TRP', 'PBA', 'EC', 'ENIC', 'EOAN', 'RWE', 'IBE', 'ELE', 'EDP',
        'ENGI', 'FP', 'NG', 'VEDL', 'SCCO', 'FCX', 'TECK', 'HBM', 'GLEN', 'AAL',
        'VALE', 'RIO', 'BHP', 'SSRM', 'PAAS',

        # Global ADRs - Industrials & Conglomerates (30 stocks)
        'UL', 'ULVR', 'UN', 'DEO', 'DANOY', 'NSRGY', 'UNLRY', 'BUD', 'SAM', 'TAP',
        'STZ', 'HEINY', 'CCEP', 'KO', 'PEP', 'SBMRY', 'SAPMY', 'SDMRY', 'BASFY', 'BAYRY',
        'LNVGY', 'LNVGF', 'LIN', 'AIR', 'EADSY', 'BA', 'RTX', 'GD', 'LMT', 'NOC'
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

#==================== BLACK-SCHOLES & GREEKS ====================

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price

    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate (annual)
    sigma: Volatility (annual)
    option_type: 'call' or 'put'

    Returns:
    Option price
    """
    from scipy.stats import norm

    if T <= 0:
        # At expiration
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)

    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate (annual)
    sigma: Volatility (annual)
    option_type: 'call' or 'put'

    Returns:
    Dictionary with all Greeks
    """
    from scipy.stats import norm

    if T <= 0:
        # At expiration, Greeks are 0 except Delta
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0

        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (same for calls and puts)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% change

    # Theta
    if option_type == 'call':
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
    else:
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)

    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Divide by 100 for 1% change
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def implied_volatility(option_price, S, K, T, r, option_type='call', max_iterations=100, tolerance=1e-5):
    """
    Calculate implied volatility using Newton-Raphson method

    Parameters:
    option_price: Market price of the option
    S: Current stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate (annual)
    option_type: 'call' or 'put'

    Returns:
    Implied volatility (annual)
    """
    # Initial guess
    sigma = 0.3

    for i in range(max_iterations):
        # Calculate option price and vega with current sigma
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = calculate_greeks(S, K, T, r, sigma, option_type)['vega'] * 100  # Multiply back for calculation

        # Price difference
        diff = option_price - price

        # Check convergence
        if abs(diff) < tolerance:
            return sigma

        # Newton-Raphson update
        if vega != 0:
            sigma = sigma + diff / vega
        else:
            return None

        # Keep sigma positive and reasonable
        sigma = max(0.01, min(sigma, 5.0))

    return None  # Failed to converge

def option_payoff(S_range, K, premium, option_type='call', position='long'):
    """
    Calculate option payoff diagram

    Parameters:
    S_range: Array of stock prices
    K: Strike price
    premium: Option premium paid/received
    option_type: 'call' or 'put'
    position: 'long' or 'short'

    Returns:
    Array of payoffs
    """
    if option_type == 'call':
        intrinsic = np.maximum(S_range - K, 0)
    else:
        intrinsic = np.maximum(K - S_range, 0)

    if position == 'long':
        payoff = intrinsic - premium
    else:
        payoff = premium - intrinsic

    return payoff

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
    ["Home", "Stock Screener", "Portfolio Builder", "Risk Analytics", "Portfolio Optimization", "AI Predictions", "Options Pricing"]
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
    st.header("🔍 Comprehensive Stock Screener")
    st.caption("Screen 2,535+ global stocks with 50+ fundamental metrics")

    # Screening Presets
    st.subheader("📋 Screening Strategy")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎯 Value Stocks", use_container_width=True):
            st.session_state.screen_preset = "value"
    with col2:
        if st.button("📈 Growth Stocks", use_container_width=True):
            st.session_state.screen_preset = "growth"
    with col3:
        if st.button("💎 Quality Stocks", use_container_width=True):
            st.session_state.screen_preset = "quality"
    with col4:
        if st.button("💰 Dividend Stocks", use_container_width=True):
            st.session_state.screen_preset = "dividend"

    # Initialize preset
    if 'screen_preset' not in st.session_state:
        st.session_state.screen_preset = None

    st.markdown("---")

    # Filters Section
    with st.expander("🔧 Screening Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Basic Filters**")
            sector_filter = st.multiselect(
                "Sector",
                ["Technology", "Healthcare", "Financials", "Consumer Discretionary", "Consumer Staples",
                 "Energy", "Industrials", "Communication Services", "Utilities", "Real Estate",
                 "Materials", "All"],
                default=["All"]
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
        st.info("🎯 **Value Strategy**: Low P/E (<15), Low P/B (<3), High Dividend Yield (>2%)")
        pe_max = 15.0
        pb_max = 3.0
    elif st.session_state.screen_preset == "growth":
        st.info("📈 **Growth Strategy**: High Earnings Growth (>15%), High Revenue Growth (>10%), High ROE (>15%)")
        roe_min = 15.0
    elif st.session_state.screen_preset == "quality":
        st.info("💎 **Quality Strategy**: High ROIC (>15%), High Profit Margin (>10%), Low Debt/Equity (<50%)")
        roic_min = 15.0
        profit_margin_min = 10.0
    elif st.session_state.screen_preset == "dividend":
        st.info("💰 **Dividend Strategy**: High Dividend Yield (>3%), Payout Ratio <70%, Positive Cash Flow")

    st.markdown("---")

    # Get tickers
    all_tickers = get_sp500_tickers()

    if len(all_tickers) > 0:
        # Fetch data for tickers
        with st.spinner(f"📊 Fetching comprehensive data for up to {results_limit} stocks... (This may take 30-60 seconds)"):
            screener_data = []
            progress_bar = st.progress(0)

            for idx, ticker in enumerate(all_tickers[:results_limit]):
                stock_info = get_stock_info(ticker)
                if stock_info:
                    # Apply sector filter
                    if "All" not in sector_filter:
                        if stock_info['sector'] not in sector_filter:
                            continue

                    # Apply valuation filters
                    if stock_info['pe_ratio']:
                        if stock_info['pe_ratio'] < pe_min or stock_info['pe_ratio'] > pe_max:
                            continue

                    if stock_info['price_to_book']:
                        if stock_info['price_to_book'] < pb_min or stock_info['price_to_book'] > pb_max:
                            continue

                    # Apply profitability filters
                    if stock_info['roe']:
                        if stock_info['roe'] * 100 < roe_min:
                            continue

                    if stock_info['roic']:
                        if stock_info['roic'] * 100 < roic_min:
                            continue

                    if stock_info['profit_margin']:
                        if stock_info['profit_margin'] * 100 < profit_margin_min:
                            continue

                    # Build comprehensive screener row
                    screener_data.append({
                        # Basic Info
                        'Symbol': ticker,
                        'Company': stock_info['name'][:30],  # Truncate long names
                        'Sector': stock_info['sector'],
                        'Price': stock_info['current_price'],
                        'Mkt Cap': stock_info['market_cap'],

                        # Valuation Metrics
                        'P/E': stock_info['pe_ratio'],
                        'Fwd P/E': stock_info['forward_pe'],
                        'PEG': stock_info['peg_ratio'],
                        'P/B': stock_info['price_to_book'],
                        'P/S': stock_info['price_to_sales'],
                        'EV/EBITDA': stock_info['ev_to_ebitda'],

                        # Per Share Metrics
                        'BVPS': stock_info['book_value_per_share'],
                        'EPS (TTM)': stock_info['eps_ttm'],
                        'EPS (Fwd)': stock_info['eps_forward'],

                        # Growth Metrics
                        'EPS Growth': stock_info['earnings_growth'] * 100 if stock_info['earnings_growth'] else None,
                        'Rev Growth': stock_info['revenue_growth'] * 100 if stock_info['revenue_growth'] else None,

                        # Profitability Metrics
                        'ROE': stock_info['roe'] * 100 if stock_info['roe'] else None,
                        'ROA': stock_info['roa'] * 100 if stock_info['roa'] else None,
                        'ROIC': stock_info['roic'] * 100 if stock_info['roic'] else None,
                        'Profit Margin': stock_info['profit_margin'] * 100 if stock_info['profit_margin'] else None,
                        'Operating Margin': stock_info['operating_margin'] * 100 if stock_info['operating_margin'] else None,
                        'Gross Margin': stock_info['gross_margin'] * 100 if stock_info['gross_margin'] else None,

                        # Dividend Metrics
                        'Div Yield': stock_info['dividend_yield'] * 100 if stock_info['dividend_yield'] else 0,
                        'Payout Ratio': stock_info['payout_ratio'] * 100 if stock_info['payout_ratio'] else None,

                        # Financial Health
                        'Current Ratio': stock_info['current_ratio'],
                        'D/E Ratio': stock_info['debt_to_equity'],
                        'FCF': stock_info['free_cash_flow'],

                        # Analyst Data
                        'Target Price': stock_info['target_mean_price'],
                        'Upside': ((stock_info['target_mean_price'] / stock_info['current_price']) - 1) * 100 if stock_info['target_mean_price'] and stock_info['current_price'] else None,
                        'Rating': stock_info['recommendation'],

                        # Additional
                        'Beta': stock_info['beta']
                    })

                # Update progress
                progress_bar.progress((idx + 1) / min(results_limit, len(all_tickers)))

            progress_bar.empty()

        if screener_data:
            screener_df = pd.DataFrame(screener_data)

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Stocks Found", len(screener_df))
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

            # Tabbed view for different metric categories
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview",
                "💵 Valuation",
                "📈 Growth & Profitability",
                "💰 Dividends",
                "🏦 Financial Health",
                "🎯 Analyst Data"
            ])

            with tab1:
                st.subheader("Stock Overview")
                overview_df = screener_df[[
                    'Symbol', 'Company', 'Sector', 'Price', 'Mkt Cap',
                    'P/E', 'ROE', 'Div Yield', 'Beta'
                ]].copy()

                st.dataframe(
                    overview_df.style.format({
                        'Price': '${:.2f}',
                        'Mkt Cap': lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x > 0 else "N/A",
                        'P/E': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'ROE': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'Div Yield': lambda x: f"{x:.2f}%" if pd.notna(x) else "0.00%",
                        'Beta': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                    }),
                    use_container_width=True,
                    height=500
                )

            with tab2:
                st.subheader("Valuation Metrics")
                valuation_df = screener_df[[
                    'Symbol', 'Company', 'Price', 'P/E', 'Fwd P/E', 'PEG',
                    'P/B', 'P/S', 'EV/EBITDA', 'BVPS'
                ]].copy()

                st.dataframe(
                    valuation_df.style.format({
                        'Price': '${:.2f}',
                        'P/E': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'Fwd P/E': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'PEG': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'P/B': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'P/S': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'EV/EBITDA': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'BVPS': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                    }),
                    use_container_width=True,
                    height=500
                )

                st.caption("**BVPS**: Book Value Per Share | **P/E**: Price-to-Earnings | **PEG**: Price/Earnings to Growth | **P/B**: Price-to-Book | **EV/EBITDA**: Enterprise Value to EBITDA")

            with tab3:
                st.subheader("Growth & Profitability")
                growth_df = screener_df[[
                    'Symbol', 'Company', 'EPS (TTM)', 'EPS (Fwd)', 'EPS Growth', 'Rev Growth',
                    'ROE', 'ROA', 'ROIC', 'Profit Margin', 'Operating Margin', 'Gross Margin'
                ]].copy()

                st.dataframe(
                    growth_df.style.format({
                        'EPS (TTM)': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                        'EPS (Fwd)': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                        'EPS Growth': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'Rev Growth': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'ROE': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'ROA': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'ROIC': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'Profit Margin': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'Operating Margin': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'Gross Margin': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                    }),
                    use_container_width=True,
                    height=500
                )

                st.caption("**EPS**: Earnings Per Share | **ROE**: Return on Equity | **ROA**: Return on Assets | **ROIC**: Return on Invested Capital")

            with tab4:
                st.subheader("Dividend Metrics")
                dividend_df = screener_df[[
                    'Symbol', 'Company', 'Price', 'Div Yield', 'Payout Ratio',
                    'EPS (TTM)', 'FCF'
                ]].copy()

                st.dataframe(
                    dividend_df.style.format({
                        'Price': '${:.2f}',
                        'Div Yield': lambda x: f"{x:.2f}%" if pd.notna(x) else "0.00%",
                        'Payout Ratio': lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
                        'EPS (TTM)': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                        'FCF': lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x > 0 else "N/A"
                    }),
                    use_container_width=True,
                    height=500
                )

                st.caption("**FCF**: Free Cash Flow | **Payout Ratio**: Dividend / Earnings")

            with tab5:
                st.subheader("Financial Health")
                health_df = screener_df[[
                    'Symbol', 'Company', 'Current Ratio', 'D/E Ratio',
                    'FCF', 'Beta', 'Mkt Cap'
                ]].copy()

                st.dataframe(
                    health_df.style.format({
                        'Current Ratio': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'D/E Ratio': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'FCF': lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x > 0 else "N/A",
                        'Beta': lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
                        'Mkt Cap': lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x > 0 else "N/A"
                    }),
                    use_container_width=True,
                    height=500
                )

                st.caption("**Current Ratio**: Current Assets / Current Liabilities | **D/E**: Debt-to-Equity Ratio | **Beta**: Market volatility (1.0 = market)")

            with tab6:
                st.subheader("Analyst Targets & Recommendations")
                analyst_df = screener_df[[
                    'Symbol', 'Company', 'Price', 'Target Price', 'Upside', 'Rating'
                ]].copy()

                st.dataframe(
                    analyst_df.style.format({
                        'Price': '${:.2f}',
                        'Target Price': lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                        'Upside': lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
                    }),
                    use_container_width=True,
                    height=500
                )

                st.caption("**Upside**: Potential gain to analyst target price | **Rating**: buy, strong_buy, hold, sell, strong_sell")

            # Sector breakdown
            st.markdown("---")
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader("📊 Sector Distribution")
                sector_counts = screener_df['Sector'].value_counts()
                fig = px.bar(
                    x=sector_counts.values,
                    y=sector_counts.index,
                    orientation='h',
                    labels={'x': 'Count', 'y': 'Sector'},
                    title="Stocks by Sector"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("")
                st.write("")
                st.write("")
                for sector, count in sector_counts.items():
                    st.write(f"**{sector}**: {count}")

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

#==================== OPTIONS PRICING ====================

elif page == "Options Pricing":
    st.header("💹 Options Pricing & Greeks Calculator")
    st.caption("Black-Scholes pricing model with full Greeks analysis")

    # Tabs for different calculators
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Option Pricing",
        "📊 Options Strategy Builder",
        "📈 Greeks Surface",
        "🔮 Implied Volatility"
    ])

    #---------- TAB 1: SINGLE OPTION PRICING ----------
    with tab1:
        st.subheader("Black-Scholes Option Pricing")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Option Parameters")

            # Fetch current price option
            ticker_input = st.text_input("Stock Ticker (optional - for live price)", value="AAPL", key='bs_ticker')
            fetch_price = st.checkbox("Fetch current price", value=False, key='fetch_price')

            if fetch_price and ticker_input:
                with st.spinner(f"Fetching {ticker_input} data..."):
                    stock_info = get_stock_info(ticker_input)
                    if stock_info:
                        S_default = stock_info['current_price']
                        sigma_default = stock_info.get('beta', 0.3) * 0.20 if stock_info.get('beta') else 0.30
                        st.success(f"Current price: ${S_default:.2f}")
                    else:
                        S_default = 100.0
                        sigma_default = 0.30
                        st.warning("Could not fetch price, using default")
            else:
                S_default = 100.0
                sigma_default = 0.30

            S = st.number_input("Current Stock Price ($)", value=float(S_default), min_value=0.01, step=1.0, key='bs_S')
            K = st.number_input("Strike Price ($)", value=float(S_default), min_value=0.01, step=1.0, key='bs_K')
            T = st.number_input("Time to Expiration (days)", value=30, min_value=1, step=1, key='bs_T') / 365.0
            r = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1, key='bs_r') / 100
            sigma = st.number_input("Volatility (% annual)", value=sigma_default * 100, min_value=1.0, max_value=200.0, step=1.0, key='bs_sigma') / 100
            option_type = st.selectbox("Option Type", ["call", "put"], key='bs_type')

        with col2:
            st.markdown("#### Pricing Results")

            # Calculate option price
            option_price = black_scholes(S, K, T, r, sigma, option_type)

            # Calculate Greeks
            greeks = calculate_greeks(S, K, T, r, sigma, option_type)

            # Display price
            st.metric("Option Price", f"${option_price:.4f}", help="Black-Scholes theoretical price")

            # Calculate intrinsic and time value
            if option_type == 'call':
                intrinsic = max(S - K, 0)
            else:
                intrinsic = max(K - S, 0)
            time_value = option_price - intrinsic

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Intrinsic Value", f"${intrinsic:.4f}")
            with col_b:
                st.metric("Time Value", f"${time_value:.4f}")

            # Moneyness
            moneyness = S / K
            if moneyness > 1.02:
                money_status = "In-the-Money (ITM)" if option_type == 'call' else "Out-of-the-Money (OTM)"
            elif moneyness < 0.98:
                money_status = "Out-of-the-Money (OTM)" if option_type == 'call' else "In-the-Money (ITM)"
            else:
                money_status = "At-the-Money (ATM)"

            st.info(f"**Status:** {money_status} (S/K = {moneyness:.4f})")

        st.markdown("---")
        st.subheader("📐 The Greeks")
        st.caption("Sensitivity measures of option price to various factors")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Delta (Δ)",
                f"{greeks['delta']:.4f}",
                help="Change in option price for $1 change in stock price"
            )

        with col2:
            st.metric(
                "Gamma (Γ)",
                f"{greeks['gamma']:.4f}",
                help="Change in Delta for $1 change in stock price"
            )

        with col3:
            st.metric(
                "Theta (Θ)",
                f"{greeks['theta']:.4f}",
                help="Change in option price per day (time decay)"
            )

        with col4:
            st.metric(
                "Vega (ν)",
                f"{greeks['vega']:.4f}",
                help="Change in option price for 1% change in volatility"
            )

        with col5:
            st.metric(
                "Rho (ρ)",
                f"{greeks['rho']:.4f}",
                help="Change in option price for 1% change in interest rate"
            )

        # Greeks interpretation
        st.markdown("#### Greeks Interpretation")

        if greeks['delta'] >= 0:
            delta_interp = f"For every \\$1 increase in stock price, the option price increases by \\${greeks['delta']:.4f}"
        else:
            delta_interp = f"For every \\$1 increase in stock price, the option price decreases by \\${abs(greeks['delta']):.4f}"

        gamma_interp = f"Delta changes by {greeks['gamma']:.4f} for each \\$1 move in stock price"

        if greeks['theta'] >= 0:
            theta_interp = f"Option gains \\${greeks['theta']:.4f} in value per day"
        else:
            theta_interp = f"Option loses \\${abs(greeks['theta']):.4f} in value per day (time decay)"

        vega_interp = f"Option price changes by \\${greeks['vega']:.4f} for each 1% change in volatility"

        st.markdown(f"**Delta:** {delta_interp}")
        st.markdown(f"**Gamma:** {gamma_interp}")
        st.markdown(f"**Theta:** {theta_interp}")
        st.markdown(f"**Vega:** {vega_interp}")

        # Payoff diagram
        st.markdown("---")
        st.subheader("📊 Payoff Diagram at Expiration")

        # Create range of stock prices
        S_range = np.linspace(S * 0.7, S * 1.3, 100)

        # Calculate payoff (assuming option bought at theoretical price)
        payoff_long = option_payoff(S_range, K, option_price, option_type, 'long')
        payoff_short = option_payoff(S_range, K, option_price, option_type, 'short')

        # Plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=S_range,
            y=payoff_long,
            mode='lines',
            name=f'Long {option_type.capitalize()}',
            line=dict(color='green', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=S_range,
            y=payoff_short,
            mode='lines',
            name=f'Short {option_type.capitalize()}',
            line=dict(color='red', width=2)
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        # Add current price line
        fig.add_vline(x=S, line_dash="dot", line_color="blue", opacity=0.5, annotation_text="Current Price")

        # Add strike line
        fig.add_vline(x=K, line_dash="dot", line_color="orange", opacity=0.5, annotation_text="Strike")

        fig.update_layout(
            title=f"{option_type.capitalize()} Option Payoff at Expiration",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Break-even analysis
        if option_type == 'call':
            breakeven = K + option_price
            max_loss = option_price
            max_gain = "Unlimited"
        else:
            breakeven = K - option_price
            max_loss = option_price
            max_gain = f"${K - option_price:.2f}"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Break-Even Price", f"${breakeven:.2f}")
        with col2:
            st.metric("Max Loss (Long)", f"${max_loss:.2f}")
        with col3:
            st.metric("Max Gain (Long)", max_gain)

    #---------- TAB 2: OPTIONS STRATEGY BUILDER ----------
    with tab2:
        st.subheader("Multi-Leg Options Strategy Builder")
        st.info("🚧 Coming Soon: Build complex strategies like spreads, straddles, iron condors, and more!")

        # Preview of strategies
        st.markdown("### Popular Strategies")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Bull Call Spread")
            st.write("• Buy Call (lower strike)")
            st.write("• Sell Call (higher strike)")
            st.write("• Limited risk, limited reward")

        with col2:
            st.markdown("#### Straddle")
            st.write("• Buy Call (ATM)")
            st.write("• Buy Put (ATM)")
            st.write("• Profit from volatility")

        with col3:
            st.markdown("#### Iron Condor")
            st.write("• Sell Call & Put (ATM)")
            st.write("• Buy Call & Put (OTM)")
            st.write("• Profit from low volatility")

    #---------- TAB 3: GREEKS SURFACE ----------
    with tab3:
        st.subheader("Greeks Surface Visualization")

        greek_to_plot = st.selectbox(
            "Select Greek to Visualize",
            ["Delta", "Gamma", "Theta", "Vega"],
            key='greek_surface'
        )

        # Use parameters from tab1 or defaults
        S_surface = st.slider("Current Stock Price", 50.0, 200.0, 100.0, 5.0, key='surface_S')
        T_days = st.slider("Days to Expiration", 1, 180, 30, 1, key='surface_T')
        r_surface = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.5, key='surface_r') / 100
        option_type_surface = st.selectbox("Option Type", ["call", "put"], key='surface_type')

        if st.button("Generate Greeks Surface", type="primary", key='gen_surface'):
            with st.spinner("Generating 3D surface..."):
                # Create meshgrid
                strikes = np.linspace(S_surface * 0.7, S_surface * 1.3, 30)
                volatilities = np.linspace(0.1, 1.0, 30)
                K_grid, Sigma_grid = np.meshgrid(strikes, volatilities)

                # Calculate Greeks
                Greek_grid = np.zeros_like(K_grid)
                T_years = T_days / 365.0

                for i in range(len(volatilities)):
                    for j in range(len(strikes)):
                        greeks = calculate_greeks(
                            S_surface,
                            strikes[j],
                            T_years,
                            r_surface,
                            volatilities[i],
                            option_type_surface
                        )
                        Greek_grid[i, j] = greeks[greek_to_plot.lower()]

                # Plot 3D surface
                fig = go.Figure(data=[go.Surface(
                    x=K_grid,
                    y=Sigma_grid * 100,  # Convert to percentage
                    z=Greek_grid,
                    colorscale='Viridis'
                )])

                fig.update_layout(
                    title=f"{greek_to_plot} Surface for {option_type_surface.capitalize()} Option",
                    scene=dict(
                        xaxis_title="Strike Price ($)",
                        yaxis_title="Volatility (%)",
                        zaxis_title=greek_to_plot
                    ),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption(f"**Spot Price:** ${S_surface} | **Days to Expiration:** {T_days} | **Risk-Free Rate:** {r_surface*100:.1f}%")

    #---------- TAB 4: IMPLIED VOLATILITY ----------
    with tab4:
        st.subheader("Implied Volatility Calculator")
        st.caption("Calculate implied volatility from market option prices")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Input Parameters")

            S_iv = st.number_input("Current Stock Price ($)", value=100.0, min_value=0.01, step=1.0, key='iv_S')
            K_iv = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=1.0, key='iv_K')
            T_iv = st.number_input("Time to Expiration (days)", value=30, min_value=1, step=1, key='iv_T') / 365.0
            r_iv = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.1, key='iv_r') / 100
            option_price_iv = st.number_input("Market Option Price ($)", value=5.0, min_value=0.01, step=0.1, key='iv_price')
            option_type_iv = st.selectbox("Option Type", ["call", "put"], key='iv_type')

        with col2:
            st.markdown("#### Results")

            if st.button("Calculate Implied Volatility", type="primary", key='calc_iv'):
                with st.spinner("Calculating implied volatility..."):
                    iv = implied_volatility(option_price_iv, S_iv, K_iv, T_iv, r_iv, option_type_iv)

                    if iv is not None:
                        st.success("✅ Convergence achieved!")
                        st.metric("Implied Volatility", f"{iv * 100:.2f}%")

                        # Calculate option price with IV
                        calculated_price = black_scholes(S_iv, K_iv, T_iv, r_iv, iv, option_type_iv)
                        st.metric("Recalculated Price", f"${calculated_price:.4f}")
                        st.caption(f"Market Price: ${option_price_iv:.4f} | Difference: ${abs(calculated_price - option_price_iv):.4f}")

                        # Compare to historical volatility
                        st.markdown("---")
                        st.markdown("#### Implied vs Historical Volatility")
                        st.info("💡 High implied volatility suggests the market expects large price movements")

                        # Fetch historical volatility if ticker was used
                        if fetch_price and ticker_input and stock_info:
                            hist_data = get_historical_data([ticker_input], period='1y')
                            if not hist_data.empty:
                                returns = hist_data[ticker_input].pct_change().dropna()
                                hist_vol = returns.std() * np.sqrt(252)

                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Implied Volatility", f"{iv * 100:.2f}%")
                                with col_b:
                                    st.metric("Historical Volatility (1Y)", f"{hist_vol * 100:.2f}%")

                                vol_ratio = iv / hist_vol
                                if vol_ratio > 1.2:
                                    st.warning(f"⚠️ Options are expensive! IV is {((vol_ratio - 1) * 100):.1f}% higher than historical")
                                elif vol_ratio < 0.8:
                                    st.success(f"✅ Options are cheap! IV is {((1 - vol_ratio) * 100):.1f}% lower than historical")
                                else:
                                    st.info("ℹ️ IV is fairly priced relative to historical volatility")
                    else:
                        st.error("❌ Failed to converge. Check input parameters.")

        st.markdown("---")
        st.markdown("### Understanding Implied Volatility")
        st.write("""
        **Implied Volatility (IV)** is the market's forecast of a likely movement in a security's price.
        It's derived from the option's market price using the Black-Scholes model.

        - **High IV**: Market expects large price swings (options are expensive)
        - **Low IV**: Market expects small price movements (options are cheap)
        - **IV > Historical Vol**: Options may be overpriced
        - **IV < Historical Vol**: Options may be underpriced
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "HedgeAbove v0.2.0 | Rise Above Market Uncertainty | "
    "<a href='https://github.com/lonespear/HedgeAbove'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
