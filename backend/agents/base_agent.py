from __future__ import annotations

"""
Base trading agent class.
Every concrete agent inherits from TradingAgent and implements
the agentic pipeline: ``perceive() -> reason() -> act()``.

All agents in this system are **autonomous, goal-driven, rule-based decision
makers** – they observe market state, apply their strategy independently,
and produce a structured decision dict with a human-readable reasoning string.
They are NOT chatbots or simple wrappers around an LLM.
"""

import math


class TradingAgent:
    """
    Abstract base class for all autonomous trading agents.

    Each subclass defines a distinct **goal** (e.g., "maximise trend profits",
    "exploit mean-reversion") and implements the ``perceive()`` and
    ``reason()`` methods.

    Attributes:
        name                   – human-readable agent name
        cash                   – available cash balance
        positions              – dict {ticker: quantity}
        avg_cost               – dict {ticker: average_cost_basis}
        portfolio_value_history – list of portfolio values over time
        last_action            – string label of last action ("BUY"/"SELL"/"HOLD")
        last_reasoning         – human-readable explanation of last decision
        last_reason            – alias kept for backward compatibility
        goal                   – strategic objective (set by subclasses)
        halted                 – True when circuit breaker has halted this agent
        active                 – whether agent participates in simulation steps
    """

    def __init__(self, name: str, initial_cash: float = 100_000.0):
        self.name = name
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, int] = {}
        self.avg_cost: dict[str, float] = {}
        self.portfolio_value_history: list[float] = []
        self.last_action: str = ""
        self.last_reasoning: str = ""
        self.last_reason: str = ""            # backward-compat alias
        self.goal: str = ""                   # set by subclasses
        self._state: dict | None = None
        self.halted: bool = False
        self.active: bool = True
        self._peak_value: float = initial_cash
        self.followers: int = 1  # number of identical traders using this strategy
        
        # Agentic Architecture additions
        self.memory: list[dict] = []
        self.performance_stats: dict = {
            "pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "max_drawdown": 0.0
        }

    # ------------------------------------------------------------------ #
    # Agentic Architecture Methods
    # ------------------------------------------------------------------ #

    def perceive(self, market_state: dict) -> dict:
        """
        Extract relevant features from the raw market state.
        Subclasses should override this to extract specific indicators.
        """
        self._state = market_state
        if isinstance(market_state, dict):
            current_bar = market_state.get("current_bar", {})
            if isinstance(current_bar, dict):
                enriched = dict(market_state)
                enriched["market_features"] = dict(current_bar)
                return enriched
        return market_state

    def reason(self, observation: dict) -> dict:
        """
        Use rules/logic to form an internal decision or plan based on the observation.
        Subclasses MUST override this.
        Returns a dict with at least 'action', 'quantity', and 'reasoning'.
        """
        return {
            "action": "HOLD",
            "quantity": 0,
            "reasoning": "Base agent does not implement a strategy."
        }

    def act(self, decision: dict) -> dict:
        """
        Convert the internal decision into a concrete action dict.
        """
        self.last_action = decision.get("action", "HOLD")
        self.last_reasoning = decision.get("reasoning", "No reason provided.")
        self.last_reason = self.last_reasoning
        
        action_dict = {
            "type": self.last_action,
            "action": self.last_action,
            "quantity": decision.get("quantity", 0),
            "reasoning": self.last_reasoning,
            "ticker": decision.get("ticker", "")
        }
        return action_dict

    def step(self, market_state: dict) -> dict:
        """
        Orchestrates the perceive -> reason -> act pipeline and updates memory.
        """
        observation = self.perceive(market_state)
        decision = self.reason(observation)
        action = self.act(decision)
        
        # Store in memory
        self.memory.append({
            "step": market_state.get("current_step", 0),
            "observation": observation,
            "decision": decision,
            "action": action,
            "result": None,
        })
        
        return action

    def explain_last_action(self) -> str:
        """Return a human-readable reason for the last action."""
        return self.last_reasoning

    # ------------------------------------------------------------------ #
    # Reasoning helper
    # ------------------------------------------------------------------ #

    def build_reasoning(self, **kwargs) -> str:
        """
        Utility to build a concise, human-readable reasoning string
        from key indicators.

        Example::

            self.build_reasoning(RSI=25, price_vs_BB="below BB_LOW",
                                 expectation="mean reversion")
            # → "RSI=25, price_vs_BB=below BB_LOW, expectation=mean reversion"
        """
        if not kwargs:
            return "No additional indicators."
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        return ", ".join(parts)

    def update_after_step(self, reward: float, new_state: dict):
        """Update memory result and performance stats after each simulation step."""
        pnl = float(self.cash - self.initial_cash)
        self.performance_stats["pnl"] = pnl

        if reward > 0:
            self.performance_stats["wins"] += 1
        elif reward < 0:
            self.performance_stats["losses"] += 1

        bar = new_state.get("current_bar", {}) if isinstance(new_state, dict) else {}
        close = bar.get("Close", 0.0)
        ticker = bar.get("ticker", "")
        portfolio_value = self.get_portfolio_value(close, ticker) if close else self.cash
        peak = max(self.initial_cash, self._peak_value)
        drawdown = ((portfolio_value - peak) / peak) if peak > 0 else 0.0
        if drawdown < self.performance_stats["max_drawdown"]:
            self.performance_stats["max_drawdown"] = drawdown

        if self.memory:
            self.memory[-1]["result"] = {
                "reward": reward,
                "portfolio_value": portfolio_value,
                "cash": self.cash,
                "positions": dict(self.positions),
                "pnl": self.performance_stats["pnl"],
                "max_drawdown": self.performance_stats["max_drawdown"],
            }

    # ------------------------------------------------------------------ #
    # Portfolio helpers
    # ------------------------------------------------------------------ #

    def get_portfolio_value(self, current_price: float, ticker: str = "") -> float:
        """
        Compute total portfolio value = cash + sum(positions * current_price).
        For single-ticker simulation, pass the ticker or leave blank
        (will sum all positions).
        """
        holdings_value = 0.0
        for t, qty in self.positions.items():
            if ticker and t != ticker:
                continue
            holdings_value += qty * current_price
        return self.cash + holdings_value

    def execute_action(self, action: dict, current_price: float):
        """
        Actually apply a trade to cash / positions.
        Assumes the action has already been reviewed by the regulator.
        """
        # Support both old ("type") and new ("action") key names
        action_type = action.get("action") or action.get("type", "HOLD")
        ticker = action.get("ticker", "")
        quantity = action.get("quantity", 0)

        if action_type == "BUY" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.cash:
                self.cash -= cost
                prev_qty = self.positions.get(ticker, 0)
                prev_cost = self.avg_cost.get(ticker, 0.0)
                new_qty = prev_qty + quantity
                # Update average cost basis
                if new_qty > 0:
                    self.avg_cost[ticker] = (
                        (prev_cost * prev_qty + current_price * quantity) / new_qty
                    )
                self.positions[ticker] = new_qty

        elif action_type == "SELL" and quantity > 0:
            current_qty = self.positions.get(ticker, 0)
            sell_qty = min(quantity, current_qty)  # cannot sell more than held
            if sell_qty > 0:
                self.cash += sell_qty * current_price
                self.positions[ticker] = current_qty - sell_qty
                if self.positions[ticker] == 0:
                    self.positions.pop(ticker, None)
                    self.avg_cost.pop(ticker, None)

        # Keep structured dict on the instance for backward compat,
        # but also update the canonical string attributes.
        self.last_action = action_type
        reasoning = action.get("reasoning", "")
        if reasoning:
            self.last_reasoning = reasoning
            self.last_reason = reasoning

    # ------------------------------------------------------------------ #
    # Risk metrics
    # ------------------------------------------------------------------ #

    def get_risk_metrics(self, current_price: float, ticker: str = "") -> dict:
        """
        Compute per-agent risk metrics:
        - return_pct       – total return since inception
        - max_drawdown_pct – worst peak-to-trough decline
        - sharpe_ratio     – annualised Sharpe (risk-free = 0%)
        """
        pv = self.get_portfolio_value(current_price, ticker)

        # Return %
        return_pct = ((pv - self.initial_cash) / self.initial_cash) * 100 if self.initial_cash > 0 else 0.0

        # Max drawdown
        peak = self.initial_cash
        max_dd = 0.0
        for v in self.portfolio_value_history:
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

        # current drawdown from peak for circuit-breaker
        if pv > self._peak_value:
            self._peak_value = pv
        current_dd = (pv - self._peak_value) / self._peak_value if self._peak_value > 0 else 0

        # Sharpe ratio (step-wise returns)
        if len(self.portfolio_value_history) >= 2:
            returns = []
            for i in range(1, len(self.portfolio_value_history)):
                prev = self.portfolio_value_history[i - 1]
                if prev > 0:
                    returns.append((self.portfolio_value_history[i] - prev) / prev)
            if returns:
                avg_r = sum(returns) / len(returns)
                std_r = math.sqrt(sum((r - avg_r) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 0
                sharpe = (avg_r / std_r) if std_r > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "return_pct": round(return_pct, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "current_drawdown_pct": round(current_dd * 100, 2),
        }

    def to_dict(self, current_price: float, ticker: str = "") -> dict:
        """Serialise agent state for the frontend."""
        pv = self.get_portfolio_value(current_price, ticker)
        risk = self.get_risk_metrics(current_price, ticker)
        latest_memory = self.memory[-1] if self.memory else None
        perf = {
            "pnl": round(float(self.performance_stats.get("pnl", 0.0)), 2),
            "wins": int(self.performance_stats.get("wins", 0)),
            "losses": int(self.performance_stats.get("losses", 0)),
            "max_drawdown": round(float(self.performance_stats.get("max_drawdown", 0.0)), 6),
        }

        # Determine agent status label (used by OrchestratorAgent snapshot)
        if not self.active:
            status = "DISABLED"
        elif self.halted:
            status = "HALTED"
        else:
            status = "ACTIVE"

        return {
            "name": self.name,
            "initial_cash": round(self.initial_cash, 2),
            "cash": round(self.cash, 2),
            "positions": dict(self.positions),
            "portfolio_value": round(pv, 2),
            "last_action": self.last_action,
            "last_reasoning": self.last_reasoning,
            "last_reason": self.last_reason,
            "goal": self.goal,
            "halted": self.halted,
            "active": self.active,
            "status": status,
            "return_pct": risk["return_pct"],
            "max_drawdown_pct": risk["max_drawdown_pct"],
            "sharpe_ratio": risk["sharpe_ratio"],
            "memory_size": len(self.memory),
            "latest_memory": latest_memory,
            "performance_stats": perf,
            "followers": self.followers,
        }
