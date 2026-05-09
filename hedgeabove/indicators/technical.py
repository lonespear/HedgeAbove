"""Technical indicators: pure-pandas computations, no Streamlit dependency."""
import pandas as pd


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def add_indicators(df, rsi_period=14, ma_fast=50, ma_slow=200):
    """Append RSI, MACD(12,26,9), and SMA(fast/slow) columns via pandas-ta."""
    import pandas_ta as ta  # noqa: F401  (registers .ta accessor)
    df.ta.rsi(length=rsi_period, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=ma_fast, append=True)
    df.ta.sma(length=ma_slow, append=True)
    return df
