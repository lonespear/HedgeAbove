"""Telegram Bot API sender. No Streamlit dependency."""
import requests
from hedgeabove import config


def send(message):
    """Post a message via Telegram. Returns True if delivered.

    Falls back to printing the alert to stdout if no token is configured —
    keeps `scan.py --once` useful for local dry-runs without secrets.
    """
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        print(f"[ALERT — Telegram not configured]\n{message}")
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": message},
            timeout=10,
        )
        return r.ok
    except Exception as e:
        print(f"Telegram error: {e}")
        return False
