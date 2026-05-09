"""Technical indicators: pure-pandas computations, no Streamlit dependency."""
import re
import pandas as pd


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def add_indicators(df, rsi_period=14, ma_fast=50, ma_slow=200,
                   bb_length=20, bb_std=2.0, vol_ma=20, lookback_52w=252):
    """Append the standard indicator set used by the rule registry.

    Columns added (suffixes match pandas-ta defaults):
      RSI_{rsi_period}                                  RSI
      MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9        MACD line/signal/hist
      SMA_{ma_fast}, SMA_{ma_slow}                      moving averages
      BBL_{bb_length}_{bb_std}, BBM..., BBU..., BBB...  Bollinger bands +width
      VOL_MA_{vol_ma}                                   rolling avg volume
      HIGH_{lookback_52w}, LOW_{lookback_52w}           rolling 52w extremes (prior bars)
    """
    import pandas_ta as ta  # noqa: F401  (registers .ta accessor)
    df.ta.rsi(length=rsi_period, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=ma_fast, append=True)
    df.ta.sma(length=ma_slow, append=True)
    df.ta.bbands(length=bb_length, std=bb_std, append=True)

    # pandas-ta 0.4.x emits Bollinger column names with the std suffix
    # duplicated (e.g. ``BBL_20_2.0_2.0``); collapse to a single suffix so
    # rules don't have to know about the version-specific naming quirk.
    df.columns = [
        re.sub(r"^(BB[LMUBP])_(\d+)_(\d+\.\d+)_\3$", r"\1_\2_\3", c) for c in df.columns
    ]

    if "Volume" in df.columns:
        df[f"VOL_MA_{vol_ma}"] = df["Volume"].rolling(vol_ma).mean()

    # 52-week high/low computed on the *prior* `lookback_52w` bars so a
    # breakout rule can compare today's close against that window without
    # the window already containing today's value.
    df[f"HIGH_{lookback_52w}"] = df["High"].shift(1).rolling(lookback_52w).max()
    df[f"LOW_{lookback_52w}"] = df["Low"].shift(1).rolling(lookback_52w).min()
    return df
