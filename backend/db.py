from __future__ import annotations

"""
SQLite storage layer for the Multi-Agent Stock Market AI Autonomity simulator.

Purpose:
    Persistent storage for trade logs and regulation events, enabling
    post-hoc analysis of simulation runs.  This satisfies the DevHack 2026
    Phase-1 requirement: "Must use Database / Storage OR RAG (Vector Database).
    Solutions without backend and without any storage mechanism will not be
    shortlisted."

Tables:
    runs              – one row per simulation run (ticker, period, timestamps)
    trades            – every trade decision (BUY/SELL/HOLD) with regulator verdict
    regulation_events – every WARN / BLOCK event with rule and explanation

Usage:
    from db import SimulationDB
    db = SimulationDB()          # creates / opens  devhack.db
    db.create_run(run_id, ...)
    db.insert_trade(...)
    db.insert_regulation_event(...)
    db.close()

Design notes:
    - Write-only during the simulation (reads are for analytics later).
    - Errors are caught and logged so they never crash the simulation.
    - Thread-safe via check_same_thread=False (Flask may serve on threads).
"""

import sqlite3
import os
import traceback
from datetime import datetime


# Default DB file lives next to this module
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devhack.db")


class SimulationDB:
    """
    SQLite storage layer used for trade and regulation logs (post-hoc analysis).
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._connect()
        self._ensure_tables()

    # ------------------------------------------------------------------ #
    # Connection helpers
    # ------------------------------------------------------------------ #

    def _connect(self):
        """Open (or create) the SQLite database file."""
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,  # Flask may call from different threads
            )
            self.conn.execute("PRAGMA journal_mode=WAL")  # faster concurrent writes
        except Exception:
            traceback.print_exc()
            self.conn = None

    def _ensure_tables(self):
        """Create tables if they do not already exist."""
        if self.conn is None:
            return
        try:
            cur = self.conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id     TEXT PRIMARY KEY,
                    ticker     TEXT,
                    period     TEXT,
                    interval   TEXT,
                    started_at TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT,
                    step            INTEGER,
                    ticker          TEXT,
                    agent           TEXT,
                    action          TEXT,
                    price           REAL,
                    quantity         INTEGER,
                    portfolio_value REAL,
                    decision        TEXT,
                    decision_reason TEXT,
                    ts              TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS regulation_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT,
                    step        INTEGER,
                    agent       TEXT,
                    rule        TEXT,
                    decision    TEXT,
                    explanation TEXT,
                    ts          TEXT
                )
            """)

            self.conn.commit()
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #

    def create_run(self, run_id: str, ticker: str, period: str, interval: str):
        """Insert a new simulation run record."""
        if self.conn is None:
            return
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, ticker, period, interval, started_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, ticker, period, interval, datetime.now().isoformat()),
            )
            self.conn.commit()
        except Exception:
            traceback.print_exc()

    def _safe_commit(self):
        """Commit if a transaction is active; silently skip otherwise."""
        try:
            if self.conn and self.conn.in_transaction:
                self.conn.commit()
        except Exception:
            traceback.print_exc()

    def insert_trade(
        self,
        run_id: str,
        step: int,
        ticker: str,
        agent: str,
        action: str,
        price: float,
        quantity: int,
        portfolio_value: float,
        decision: str,
        decision_reason: str,
    ):
        """Insert one trade record into the trades table."""
        if self.conn is None:
            return
        try:
            self.conn.execute(
                "INSERT INTO trades "
                "(run_id, step, ticker, agent, action, price, quantity, "
                " portfolio_value, decision, decision_reason, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, step, ticker, agent, action,
                    round(price, 2), quantity, round(portfolio_value, 2),
                    decision, decision_reason,
                    datetime.now().isoformat(),
                ),
            )
            self._safe_commit()
        except Exception:
            traceback.print_exc()

    def insert_regulation_event(
        self,
        run_id: str,
        step: int,
        agent: str,
        rule: str,
        decision: str,
        explanation: str,
    ):
        """Insert one regulation event record."""
        if self.conn is None:
            return
        try:
            self.conn.execute(
                "INSERT INTO regulation_events "
                "(run_id, step, agent, rule, decision, explanation, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, step, agent, rule, decision, explanation,
                    datetime.now().isoformat(),
                ),
            )
            self._safe_commit()
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
