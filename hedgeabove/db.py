"""
SQLite persistence for portfolios and watchlists.

Database file lives alongside the app at data/hedgeabove.db.
Schema is created automatically on first use.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_DB_PATH = os.path.join(_DB_DIR, "hedgeabove.db")


def _get_conn():
    os.makedirs(_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
            symbol       TEXT    NOT NULL,
            shares       REAL   NOT NULL,
            avg_price    REAL   NOT NULL,
            added_at     TEXT   NOT NULL,
            UNIQUE(portfolio_id, symbol)
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT    NOT NULL UNIQUE,
            added_at  TEXT    NOT NULL,
            notes     TEXT
        );
    """)
    conn.commit()
    conn.close()


# ── Portfolio CRUD ──────────────────────────────────────────────

def list_portfolios():
    """Return list of (id, name, created_at) tuples."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name, created_at FROM portfolios ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return rows


def create_portfolio(name):
    """Create a new portfolio. Returns its id."""
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO portfolios (name, created_at, updated_at) VALUES (?, ?, ?)",
        (name, now, now),
    )
    pid = cur.lastrowid
    conn.commit()
    conn.close()
    return pid


def delete_portfolio(portfolio_id):
    conn = _get_conn()
    conn.execute("DELETE FROM positions WHERE portfolio_id = ?", (portfolio_id,))
    conn.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
    conn.commit()
    conn.close()


def rename_portfolio(portfolio_id, new_name):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "UPDATE portfolios SET name = ?, updated_at = ? WHERE id = ?",
        (new_name, now, portfolio_id),
    )
    conn.commit()
    conn.close()


# ── Position CRUD ───────────────────────────────────────────────

def load_positions(portfolio_id):
    """Load positions as a DataFrame with columns [Symbol, Shares, Avg Price]."""
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT symbol AS Symbol, shares AS Shares, avg_price AS 'Avg Price' "
        "FROM positions WHERE portfolio_id = ? ORDER BY symbol",
        conn,
        params=(portfolio_id,),
    )
    conn.close()
    return df


def upsert_position(portfolio_id, symbol, shares, avg_price):
    """Insert or update a position."""
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO positions (portfolio_id, symbol, shares, avg_price, added_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(portfolio_id, symbol)
           DO UPDATE SET shares = excluded.shares,
                         avg_price = excluded.avg_price""",
        (portfolio_id, symbol, shares, avg_price, now),
    )
    conn.execute(
        "UPDATE portfolios SET updated_at = ? WHERE id = ?", (now, portfolio_id)
    )
    conn.commit()
    conn.close()


def delete_position(portfolio_id, symbol):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "DELETE FROM positions WHERE portfolio_id = ? AND symbol = ?",
        (portfolio_id, symbol),
    )
    conn.execute(
        "UPDATE portfolios SET updated_at = ? WHERE id = ?", (now, portfolio_id)
    )
    conn.commit()
    conn.close()


def save_dataframe(portfolio_id, df):
    """Overwrite all positions from a DataFrame [Symbol, Shares, Avg Price]."""
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute("DELETE FROM positions WHERE portfolio_id = ?", (portfolio_id,))
    for _, row in df.iterrows():
        conn.execute(
            "INSERT INTO positions (portfolio_id, symbol, shares, avg_price, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (portfolio_id, row['Symbol'], row['Shares'], row['Avg Price'], now),
        )
    conn.execute(
        "UPDATE portfolios SET updated_at = ? WHERE id = ?", (now, portfolio_id)
    )
    conn.commit()
    conn.close()


# ── Watchlist ───────────────────────────────────────────────────

def add_to_watchlist(symbol, notes=""):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO watchlist (symbol, added_at, notes) VALUES (?, ?, ?)",
        (symbol, now, notes),
    )
    conn.commit()
    conn.close()


def remove_from_watchlist(symbol):
    conn = _get_conn()
    conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
    conn.commit()
    conn.close()


def get_watchlist():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT symbol, notes, added_at FROM watchlist ORDER BY added_at DESC"
    ).fetchall()
    conn.close()
    return rows
