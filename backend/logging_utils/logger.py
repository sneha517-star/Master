"""
Simulation logger – dual-write to in-memory lists AND SQLite database.

In-memory lists power the real-time UI; SQLite provides persistent storage
for post-hoc analysis (satisfies DevHack 2026 "Database / Storage" requirement).
"""

from datetime import datetime


class SimulationLogger:
    """
    Captures every trade decision and regulatory event.

    Dual-write strategy:
        1. In-memory lists  → served to the React frontend via REST API.
        2. SQLite database   → persistent storage for post-hoc analysis.

    The DB reference and run_id are injected via ``set_db()`` after
    the OrchestratorAgent creates a new run.
    """

    def __init__(self):
        self.trade_log: list[dict] = []
        self.regulation_log: list[dict] = []

        # Database reference (set after init via set_db)
        self._db = None
        self._run_id: str = ""
        self._ticker: str = ""

    def set_db(self, db, run_id: str, ticker: str):
        """
        Attach a SimulationDB instance so that every subsequent log call
        also writes to SQLite.

        Args:
            db:      SimulationDB instance (or None to disable DB writes).
            run_id:  UUID string for the current simulation run.
            ticker:  Ticker symbol for the current run.
        """
        self._db = db
        self._run_id = run_id
        self._ticker = ticker

    def reset(self):
        """Clear all in-memory logs (DB data is kept for history)."""
        self.trade_log.clear()
        self.regulation_log.clear()

    # ------------------------------------------------------------------ #
    # Trade logging
    # ------------------------------------------------------------------ #

    def log_trade(
        self,
        step: int,
        agent_name: str,
        action: str,
        price: float,
        quantity: int,
        portfolio_value: float,
        reason: str,
        decision: str,
        decision_reason: str,
    ):
        """
        Record a single trade decision (whether executed or blocked).

        Writes to both in-memory list and SQLite trades table.
        """
        self.trade_log.append({
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "agent_name": agent_name,
            "action": action,
            "price": round(price, 2),
            "quantity": quantity,
            "portfolio_value": round(portfolio_value, 2),
            "agent_reason": reason,
            "regulator_decision": decision,
            "regulator_reason": decision_reason,
        })

        # ── SQLite dual-write ────────────────────────────────────────
        if self._db and self._run_id:
            self._db.insert_trade(
                run_id=self._run_id,
                step=step,
                ticker=self._ticker,
                agent=agent_name,
                action=action,
                price=price,
                quantity=quantity,
                portfolio_value=portfolio_value,
                decision=decision,
                decision_reason=decision_reason,
            )

    # ------------------------------------------------------------------ #
    # Regulation event logging
    # ------------------------------------------------------------------ #

    def log_regulation_event(
        self,
        step: int,
        agent_name: str,
        rule_name: str,
        decision: str,
        explanation: str,
    ):
        """
        Record a regulation event (warning or block).

        Writes to both in-memory list and SQLite regulation_events table.
        """
        self.regulation_log.append({
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "agent_name": agent_name,
            "rule_name": rule_name,
            "decision": decision,
            "explanation": explanation,
        })

        # ── SQLite dual-write ────────────────────────────────────────
        if self._db and self._run_id:
            self._db.insert_regulation_event(
                run_id=self._run_id,
                step=step,
                agent=agent_name,
                rule=rule_name,
                decision=decision,
                explanation=explanation,
            )

    # ------------------------------------------------------------------ #
    # Getters
    # ------------------------------------------------------------------ #

    def get_trade_log(self) -> list[dict]:
        return list(self.trade_log)

    def get_regulation_log(self) -> list[dict]:
        return list(self.regulation_log)
