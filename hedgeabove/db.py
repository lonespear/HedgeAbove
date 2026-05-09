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

        -- Scanner: named watchlist groups, alert rules, and a one-fire-per-day
        -- dedup log. Keeps headless cron and Streamlit UI sharing the same DB.
        CREATE TABLE IF NOT EXISTS watchlist_groups (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watchlist_group_tickers (
            group_id  INTEGER NOT NULL REFERENCES watchlist_groups(id) ON DELETE CASCADE,
            symbol    TEXT    NOT NULL,
            PRIMARY KEY (group_id, symbol)
        );

        CREATE TABLE IF NOT EXISTS alert_rules (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id     INTEGER NOT NULL REFERENCES watchlist_groups(id) ON DELETE CASCADE,
            rule_type    TEXT    NOT NULL,
            params_json  TEXT    NOT NULL DEFAULT '{}',
            enabled      INTEGER NOT NULL DEFAULT 1,
            created_at   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alerts_fired (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            rule_type   TEXT    NOT NULL,
            fired_date  TEXT    NOT NULL,
            fired_at    TEXT    NOT NULL,
            message     TEXT    NOT NULL,
            UNIQUE(symbol, rule_type, fired_date)
        );

        -- Per-ticker snooze: scanner skips alerts for symbols listed here
        -- when the current UTC date is on or before until_date.
        CREATE TABLE IF NOT EXISTS snooze (
            symbol      TEXT    PRIMARY KEY,
            until_date  TEXT    NOT NULL,
            reason      TEXT,
            created_at  TEXT    NOT NULL
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


# ── Watchlist groups (scanner) ──────────────────────────────────

def create_watchlist_group(name):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO watchlist_groups (name, created_at) VALUES (?, ?)",
        (name, now),
    )
    gid = cur.lastrowid
    conn.commit()
    conn.close()
    return gid


def get_watchlist_group_by_name(name):
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, name FROM watchlist_groups WHERE name = ?", (name,)
    ).fetchone()
    conn.close()
    return row


def list_watchlist_groups():
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name FROM watchlist_groups ORDER BY name"
    ).fetchall()
    conn.close()
    return rows


def delete_watchlist_group(group_id):
    conn = _get_conn()
    conn.execute("DELETE FROM watchlist_groups WHERE id = ?", (group_id,))
    conn.commit()
    conn.close()


def add_ticker_to_group(group_id, symbol):
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO watchlist_group_tickers (group_id, symbol) VALUES (?, ?)",
        (group_id, symbol),
    )
    conn.commit()
    conn.close()


def remove_ticker_from_group(group_id, symbol):
    conn = _get_conn()
    conn.execute(
        "DELETE FROM watchlist_group_tickers WHERE group_id = ? AND symbol = ?",
        (group_id, symbol),
    )
    conn.commit()
    conn.close()


def get_watchlist_group_tickers(group_id):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT symbol FROM watchlist_group_tickers WHERE group_id = ? ORDER BY symbol",
        (group_id,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ── Alert rules ─────────────────────────────────────────────────

def add_alert_rule(group_id, rule_type, params_json='{}'):
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO alert_rules (group_id, rule_type, params_json, enabled, created_at) "
        "VALUES (?, ?, ?, 1, ?)",
        (group_id, rule_type, params_json, now),
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def list_alert_rules(group_id, enabled_only=True):
    conn = _get_conn()
    if enabled_only:
        rows = conn.execute(
            "SELECT id, rule_type, params_json FROM alert_rules "
            "WHERE group_id = ? AND enabled = 1 ORDER BY id",
            (group_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, rule_type, params_json, enabled FROM alert_rules "
            "WHERE group_id = ? ORDER BY id",
            (group_id,),
        ).fetchall()
    conn.close()
    return rows


def set_alert_rule_enabled(rule_id, enabled):
    conn = _get_conn()
    conn.execute(
        "UPDATE alert_rules SET enabled = ? WHERE id = ?",
        (1 if enabled else 0, rule_id),
    )
    conn.commit()
    conn.close()


def delete_alert_rule(rule_id):
    conn = _get_conn()
    conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
    conn.commit()
    conn.close()


# ── Alert dedup / history ───────────────────────────────────────

def alert_fired_today(symbol, rule_type):
    """True if (symbol, rule_type) already fired an alert today (UTC date)."""
    today = str(datetime.utcnow().date())
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM alerts_fired WHERE symbol = ? AND rule_type = ? AND fired_date = ?",
        (symbol, rule_type, today),
    ).fetchone()
    conn.close()
    return row is not None


def log_alert(symbol, rule_type, message):
    """Record that an alert fired. Idempotent per (symbol, rule_type, day)."""
    now = datetime.utcnow().isoformat()
    today = str(datetime.utcnow().date())
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO alerts_fired (symbol, rule_type, fired_date, fired_at, message) "
        "VALUES (?, ?, ?, ?, ?)",
        (symbol, rule_type, today, now, message),
    )
    conn.commit()
    conn.close()


def recent_alerts(limit=50):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT symbol, rule_type, fired_at, message FROM alerts_fired "
        "ORDER BY fired_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def set_alert_rule_params(rule_id, params_json):
    """Update an existing rule's params_json. Used by the UI param editor."""
    conn = _get_conn()
    conn.execute(
        "UPDATE alert_rules SET params_json = ? WHERE id = ?",
        (params_json, rule_id),
    )
    conn.commit()
    conn.close()


# ── Snooze ──────────────────────────────────────────────────────

def snooze_ticker(symbol, until_date, reason=""):
    """Snooze alerts for `symbol` until `until_date` (ISO YYYY-MM-DD, inclusive).
    Idempotent: replaces any existing snooze on the same symbol."""
    now = datetime.utcnow().isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO snooze (symbol, until_date, reason, created_at) "
        "VALUES (?, ?, ?, ?)",
        (symbol.upper(), until_date, reason, now),
    )
    conn.commit()
    conn.close()


def unsnooze_ticker(symbol):
    conn = _get_conn()
    conn.execute("DELETE FROM snooze WHERE symbol = ?", (symbol.upper(),))
    conn.commit()
    conn.close()


def is_snoozed(symbol):
    """True if `symbol` has an active snooze (until_date >= today UTC)."""
    today = str(datetime.utcnow().date())
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM snooze WHERE symbol = ? AND until_date >= ?",
        (symbol.upper(), today),
    ).fetchone()
    conn.close()
    return row is not None


def list_snoozes(active_only=True):
    """Return list of (symbol, until_date, reason, created_at). active_only filters to non-expired."""
    today = str(datetime.utcnow().date())
    conn = _get_conn()
    if active_only:
        rows = conn.execute(
            "SELECT symbol, until_date, reason, created_at FROM snooze "
            "WHERE until_date >= ? ORDER BY symbol",
            (today,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT symbol, until_date, reason, created_at FROM snooze "
            "ORDER BY symbol"
        ).fetchall()
    conn.close()
    return rows
