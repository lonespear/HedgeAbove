"""Headless scan entry point — for cron jobs (Pi deployment).

Cron example:
    5 16 * * 1-5 cd /home/pi/HedgeAbove && ./venv/bin/python scan.py

Configuration lives in the SQLite DB at hedgeabove/data/hedgeabove.db.
First-time setup:
    python -m hedgeabove.cli init
    python -m hedgeabove.cli watchlist add-ticker default <SYMBOL>
"""
from hedgeabove.scanner import run

if __name__ == "__main__":
    run()
