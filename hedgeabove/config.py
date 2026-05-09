"""
Environment-driven configuration for HedgeAbove.
Reads from .env via python-dotenv, with defaults for development.
Importable from headless contexts (no Streamlit dependency).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
MA_FAST = int(os.getenv("MA_FAST", "50"))
MA_SLOW = int(os.getenv("MA_SLOW", "200"))
LOOKBACK = os.getenv("LOOKBACK", "1y")
