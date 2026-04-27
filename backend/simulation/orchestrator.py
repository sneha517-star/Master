from __future__ import annotations

"""
OrchestratorAgent – Head Agent (Orchestrator) for DevHack 2026 Phase-1.

This module implements the central coordinator of the multi-agent stock-market
simulation.  It satisfies the DevHack 2026 requirement:

    "Must include 1 Head Agent (Orchestrator)."

Architecture label (for PPT diagrams):
    "Head Agent (Orchestrator) – coordinates all trading agents and the market."

Responsibilities:
    - Own the simulation clock (current_step, max_steps).
    - For each step:
        1. Collect current market state from MarketEnvironment.
        2. Ask every active trading agent for an action (BUY / SELL / HOLD).
        3. Send each proposed action to the Regulator for compliance review.
        4. Apply only APPROVED or WARN actions to the MarketEnvironment.
        5. Update agents with rewards and new state.
        6. Log trades and regulation events (in-memory + SQLite DB).
    - Monitor global risk (overall drawdown, violation counts).
    - Trigger circuit breakers when individual agents suffer >10 % drawdown.
    - Manage crash mode (price drops, volatility spikes, circuit breakers).
"""

import random
import uuid
import numpy as np
import copy
import pandas as pd

from market.market import MarketEnvironment
from agents.base_agent import TradingAgent
from agents.conservative_agent import ConservativeAgent
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.noise_trader import NoiseTrader
from agents.adversarial_agent import AdversarialAgent
from agents.custom_agent import CustomAgent
from regulator.regulator import RegulatorAgent
from logging_utils.logger import SimulationLogger
from db import SimulationDB


# ------------------------------------------------------------------ #
# Agent registry (maps lowercase key → class + display name)
# ------------------------------------------------------------------ #
AGENT_REGISTRY = {
    "conservative": ConservativeAgent,
    "momentum": MomentumAgent,
    "meanreversion": MeanReversionAgent,
    "noisetrader": NoiseTrader,
    "adversarial": AdversarialAgent,
    "custom": CustomAgent,
}

AGENT_DISPLAY_NAMES = {
    "conservative": "Conservative",
    "momentum": "Momentum",
    "meanreversion": "MeanReversion",
    "noisetrader": "NoiseTrader",
    "adversarial": "Adversarial",
    "custom": "Custom",
}


class OrchestratorAgent:
    """
    Head Agent (Orchestrator) – coordinates all trading agents and the market.

    This is the central autonomous coordinator required by DevHack 2026
    Phase-1 "Agentic AI" guidelines.  It is NOT a simple wrapper; it
    encapsulates the entire step-by-step simulation loop:

        Market  ──►  Orchestrator  ──►  Agents  ──►  Regulator
                         │                              │
                         ◄──────────────────────────────┘
                         │
                   Apply trades,
                   log to memory + SQLite DB,
                   compute risk metrics

    Attributes:
        market              – MarketEnvironment (price replay engine)
        agents              – list of TradingAgent instances
        regulator           – RegulatorAgent (compliance enforcement)
        logger              – SimulationLogger (in-memory logs)
        db                  – SimulationDB (SQLite persistent storage)
        run_id              – UUID identifying the current simulation run
        current_step        – current bar index
        max_steps           – total number of bars in the dataset
        ticker / period / interval – simulation parameters
        crash_active        – whether a crash event is in effect
        circuit_breakers_active – count of halted agents
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, db: SimulationDB | None = None):
        # Market & data
        self.market: MarketEnvironment | None = None
        self.ticker: str = ""
        self.period: str = ""
        self.interval: str = ""
        self.start_date: str | None = None
        self.end_date: str | None = None
        self.custom_data_df: pd.DataFrame | None = None

        # Agents (autonomous, goal-driven, rule-based decision makers)
        self.agents: list[TradingAgent] = []

        # Regulator agent enforces rules: MaxPositionLimit, BurstTrading, ManipulationPattern, etc.
        self.regulator: RegulatorAgent | None = None

        # Logging – in-memory for UI, SQLite for persistent storage
        self.logger = SimulationLogger()
        self.db: SimulationDB | None = db

        # Simulation clock
        self.current_step: int = 0
        self.max_steps: int = 0
        self.finished: bool = False

        # Run identifier (new UUID per init)
        self.run_id: str = ""

        # Price history for chart
        self.price_history: list[dict] = []
        self.trades_at_step: list[dict] = []

        # System risk tracking
        self._peak_total_value: float = 0.0

        # Crash / circuit-breaker state
        self.crash_active: bool = False
        self._crash_steps_remaining: int = 0
        self._crash_vol_multiplier: float = 1.0
        self.circuit_breakers_active: int = 0

        # Whale-crash / cascade state
        self.crash_mode_active: bool = False
        self._crash_triggered_step: int | None = None
        self._pre_crash_price: float | None = None
        self.trading_halted: bool = False
        self.GLOBAL_HALT_DRAWDOWN_PCT = -15.0    # halt all trading at -15 %

        # Config received at init (kept for re-init / jump)
        self._active_agent_keys: list[str] = []
        self._agent_params: dict = {}

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def init(
        self,
        ticker: str,
        period: str,
        interval: str,
        active_agents: list[str] | None = None,
        agent_params: dict | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        custom_data_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Download market data, create environment, instantiate agents and
        regulator, register a new run in the database, and return the
        initial snapshot.

        This is the entry point that sets up the entire multi-agent system.
        """
        # 1. Create market environment (downloads real data via yfinance)
        self.market = MarketEnvironment(
            ticker,
            period,
            interval,
            start_date=start_date,
            end_date=end_date,
            custom_data_df=custom_data_df,
        )
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.custom_data_df = custom_data_df
        self.current_step = 0
        self.max_steps = self.market.total_bars
        self.finished = False
        self.trades_at_step = []

        # Reset crash state
        self.crash_active = False
        self._crash_steps_remaining = 0
        self._crash_vol_multiplier = 1.0
        self.circuit_breakers_active = 0
        self.crash_mode_active = False
        self._crash_triggered_step = None
        self._pre_crash_price = None
        self.trading_halted = False

        # Reset in-memory logs
        self.logger.reset()

        # 2. Determine which agents are active
        if active_agents is None or len(active_agents) == 0:
            active_agents = list(AGENT_REGISTRY.keys())
        self._active_agent_keys = [k.lower() for k in active_agents]
        self._agent_params = agent_params or {}

        # 3. Instantiate autonomous trading agents
        self.agents = []
        for key in AGENT_REGISTRY:
            display = AGENT_DISPLAY_NAMES[key]
            cls = AGENT_REGISTRY[key]
            params = self._agent_params.get(key, {})
            # Allow the Custom agent to carry a user-defined display name
            agent_name = params.get("name", display) if key == "custom" else display
            agent = cls(agent_name, initial_cash=100_000.0, params=params)
            agent.active = key in self._active_agent_keys
            self.agents.append(agent)

        # 4. Instantiate regulator (compliance agent)
        self.regulator = RegulatorAgent()

        # 5. Seed price history with the initial bar
        state = self.market.get_state()
        self.price_history = [state["current_bar"]]

        # Record initial portfolio values for every agent
        close = state["current_bar"]["Close"]
        for agent in self.agents:
            pv = agent.get_portfolio_value(close, self.ticker)
            agent.portfolio_value_history.append(pv)

        # BUG FIX: only sum ACTIVE agents — get_system_risk() also filters by a.active.
        # Including disabled agents inflates the peak so the drawdown is immediately
        # negative at step 0 (e.g. 4 active × 100k vs peak of 5 × 100k = -20%).
        self._peak_total_value = sum(
            a.get_portfolio_value(close, self.ticker) for a in self.agents if a.active
        )

        # 6. Create a new run in the SQLite database
        self.run_id = str(uuid.uuid4())
        if self.db:
            self.db.create_run(self.run_id, ticker, period, interval)

        # Pass run_id + db ref to logger so it can dual-write
        self.logger.set_db(self.db, self.run_id, self.ticker)

        return self.get_snapshot()

    # ------------------------------------------------------------------ #
    # Core step logic  (the heart of the Orchestrator)
    # ------------------------------------------------------------------ #

    def run_step(self) -> dict:
        """
        Advance the simulation by one bar.

        Orchestrator step flow (suitable for PPT "Agent Workflow" diagram):

            ┌─────────────────────────────────────────┐
            │  1. Get current market state             │
            │  2. Ask each active agent for an action  │
            │  3. Send actions to Regulator for review  │
            │  4. Apply approved actions to portfolios │
            │  5. Log trades + regulation events       │
            │  6. Compute rewards & update agents      │
            │  7. Check circuit breakers               │
            │  8. Advance market to next bar            │
            └─────────────────────────────────────────┘

        Returns:
            Full simulation snapshot dict for the frontend.
        """
        if self.market is None:
            return {"error": "Simulation not initialised. Call /api/init first."}

        if self.finished or self.market.is_done():
            self.finished = True
            return self.get_snapshot()

        # ── Global trading halt guard ─────────────────────────────────
        if self.trading_halted:
            # Market is frozen – nobody trades.  Still advance the bar so
            # the chart keeps moving and the user can see the halt.
            state = self.market.get_state()
            close = state["current_bar"]["Close"]
            self.trades_at_step = []
            for agent in self.agents:
                if not agent.active:
                    continue
                pv = agent.get_portfolio_value(close, self.ticker)
                agent.portfolio_value_history.append(pv)
                agent.last_reason = "Trading halted by circuit breaker"
                agent.last_reasoning = agent.last_reason
                self.logger.log_trade(
                    step=self.current_step,
                    agent_name=agent.name,
                    action="HOLD",
                    price=close,
                    quantity=0,
                    portfolio_value=pv,
                    reason="Trading halted by circuit breaker",
                    decision="BLOCK",
                    decision_reason="GLOBAL_HALT: circuit breaker active",
                )
            step_result = self.market.step(0.0)
            self.current_step = self.market.current_step
            market_state = self.market.get_state()
            halt_bar = market_state["current_bar"]
            if not self.price_history or halt_bar.get("Datetime") != self.price_history[-1].get("Datetime"):
                self.price_history.append(halt_bar)
            if step_result.get("finished"):
                self.finished = True
            return self.get_snapshot()

        # ── Step 1: Get current market state ──────────────────────────
        state = self.market.get_state()
        bar = state["current_bar"]
        close = bar["Close"]

        # Inject ticker + simulated price data into bar for regulator
        bar["ticker"] = self.ticker
        bar["simulated_price"] = state.get("simulated_price", bar.get("Close", 0))
        bar["price_history_simulated"] = state.get("price_history_simulated", [])

        # Handle crash volatility decay
        if self._crash_steps_remaining > 0:
            self._crash_steps_remaining -= 1
            if self._crash_steps_remaining == 0:
                self.crash_active = False
                self._crash_vol_multiplier = 1.0

        self.trades_at_step = []

        # Track net order flow for endogenous price impact model
        net_volume: float = 0.0

        for agent in self.agents:
            # Skip inactive agents
            if not agent.active:
                agent.last_reason = "Agent disabled"
                continue

            # Skip halted agents (circuit breaker)
            if agent.halted:
                agent.last_reason = "HALTED by circuit breaker"
                agent.last_reasoning = agent.last_reason
                pv = agent.get_portfolio_value(close, self.ticker)
                agent.portfolio_value_history.append(pv)
                self.logger.log_trade(
                    step=self.current_step,
                    agent_name=agent.name,
                    action="HOLD",
                    price=close,
                    quantity=0,
                    portfolio_value=pv,
                    reason="HALTED by circuit breaker",
                    decision="BLOCK",
                    decision_reason="Circuit breaker active",
                )
                continue

            # ── Step 2: Ask agent for an action ───────────────────────
            # Each agent is an autonomous, goal-driven, rule-based decision maker.

            # If this is the whale agent on the crash-trigger step,
            # override its decision with the forced dump.
            if (
                self.crash_mode_active
                and self._crash_triggered_step == self.current_step
                and self._is_whale(agent)
            ):
                observation = agent.perceive(state)
                decision = self._build_whale_dump(agent)
                action = agent.act(decision)
                agent.memory.append({
                    "step": state.get("current_step", self.current_step),
                    "observation": observation,
                    "decision": decision,
                    "action": action,
                    "result": None,
                })
            else:
                action = agent.step(state)

            action.setdefault("ticker", self.ticker)

            # ── Step 3: Send action to Regulator for compliance review ─
            agent_state = self._build_agent_state(agent, close)
            review = self.regulator.review_trade(
                agent_name=agent.name,
                action=action,
                agent_state=agent_state,
                market_state=bar,
                current_step=self.current_step,
            )

            reg_decision = review["decision"]
            reg_reason = review["reason"]
            adjusted = review["adjusted_action"]

            # ── Step 4: Apply approved / warned action to portfolio ──
            if reg_decision in ("APPROVE", "WARN"):
                agent.execute_action(adjusted, close)

            # ── Step 5: Compute new portfolio value and log ──────────
            pv = agent.get_portfolio_value(close, self.ticker)
            agent.portfolio_value_history.append(pv)

            adj_type = adjusted.get("action") or adjusted.get("type", "HOLD")
            adj_qty = adjusted.get("quantity", 0)

            # Accumulate net volume for endogenous price impact
            # Followers multiply impact only — portfolio trades at original qty
            if reg_decision != "BLOCK" and adj_qty > 0:
                effective_qty = adj_qty * max(1, getattr(agent, "followers", 1))
                if adj_type == "BUY":
                    net_volume += effective_qty
                elif adj_type == "SELL":
                    net_volume -= effective_qty

            # Record trade marker for chart overlay
            if adj_type in ("BUY", "SELL") and adj_qty > 0 and reg_decision != "BLOCK":
                self.trades_at_step.append({
                    "agent": agent.name,
                    "type": adj_type,
                    "price": round(close, 2),
                    "quantity": adj_qty,
                    "step": self.current_step,
                    "reason": agent.last_reason,
                })

            # Log trade (in-memory + SQLite)
            self.logger.log_trade(
                step=self.current_step,
                agent_name=agent.name,
                action=adj_type,
                price=close,
                quantity=adj_qty,
                portfolio_value=pv,
                reason=agent.last_reason,
                decision=reg_decision,
                decision_reason=reg_reason,
            )

            # Log regulation event if WARN or BLOCK
            if reg_decision in ("WARN", "BLOCK"):
                self.logger.log_regulation_event(
                    step=self.current_step,
                    agent_name=agent.name,
                    rule_name="compliance_review",
                    decision=reg_decision,
                    explanation=reg_reason,
                )

            # ── Step 6: Reward / adaptation callback ─────────────────
            reward = 0.0
            if len(agent.portfolio_value_history) >= 2:
                reward = (
                    agent.portfolio_value_history[-1]
                    - agent.portfolio_value_history[-2]
                )
            agent.update_after_step(reward, state)

        # ── Step 7: Check circuit breakers (per-agent + global) ────────
        self.circuit_breakers_active = 0
        for agent in self.agents:
            if not agent.active:
                continue
            risk = agent.get_risk_metrics(close, self.ticker)
            if risk["current_drawdown_pct"] <= -10.0 and not agent.halted:
                agent.halted = True
            if agent.halted:
                self.circuit_breakers_active += 1

        # Global halt: if overall system drawdown exceeds threshold
        sys_risk = self.get_system_risk()
        if sys_risk.get("global_drawdown_pct", 0) <= self.GLOBAL_HALT_DRAWDOWN_PCT:
            if not self.trading_halted:  # log only on the transition
                self.logger.log_regulation_event(
                    step=self.current_step,
                    agent_name="SYSTEM",
                    rule_name="GlobalCircuitBreaker",
                    decision="BLOCK",
                    explanation=(
                        f"Global drawdown {sys_risk['global_drawdown_pct']:.1f}% "
                        f"breached {self.GLOBAL_HALT_DRAWDOWN_PCT}% threshold – "
                        f"ALL TRADING HALTED."
                    ),
                )
            self.trading_halted = True
        else:
            # Drawdown has recovered — lift the halt automatically
            self.trading_halted = False

        # ── Step 8: Advance market to next bar (endogenous price impact) ─
        # Pass net_volume so the market adjusts the next simulated price
        # based on aggregate agent order flow.
        # step() now returns a dict (not tuple) with a "finished" key.
        step_result = self.market.step(net_volume)
        self.current_step = self.market.current_step

        # Append new price bar to history (fetch full bar from get_state)
        market_state = self.market.get_state()
        new_bar = market_state["current_bar"]
        # Guard: skip duplicate timestamps (can happen at last bar or day boundaries)
        if not self.price_history or new_bar.get("Datetime") != self.price_history[-1].get("Datetime"):
            self.price_history.append(new_bar)

        # Update peak total value for global drawdown tracking
        sim_close = self.market.current_price
        total = sum(
            a.get_portfolio_value(sim_close, self.ticker)
            for a in self.agents if a.active
        )
        if total > self._peak_total_value:
            self._peak_total_value = total

        if step_result.get("finished"):
            self.finished = True

        return self.get_snapshot()

    # ------------------------------------------------------------------ #
    # Helper: build agent state dict for regulator
    # ------------------------------------------------------------------ #

    def _build_agent_state(self, agent: TradingAgent, close: float) -> dict:
        """Build the agent state dict that the Regulator expects."""
        return {
            "cash": agent.cash,
            "positions": dict(agent.positions),
            "portfolio_value": agent.get_portfolio_value(close, self.ticker),
        }

    # ------------------------------------------------------------------ #
    # Live agent toggle  (mid-simulation enable / disable)
    # ------------------------------------------------------------------ #

    def set_active_agents(self, active_keys: list[str]) -> dict:
        """
        Enable / disable agents by key without re-initialising the simulation.
        Called when the user toggles an agent in the UI mid-run.
        Also recomputes _peak_total_value so the global circuit-breaker
        threshold stays correct after the active set changes.
        """
        keys = [k.lower() for k in active_keys]
        if self.market is None:
            return {"error": "Simulation not initialised."}

        close = self.market.current_price
        for agent in self.agents:
            # Match by key: agent name lowercased with spaces removed
            agent_key = agent.name.lower().replace(" ", "")
            # Also check against AGENT_REGISTRY keys (display name may differ for custom)
            matched = False
            for reg_key, cls in AGENT_REGISTRY.items():
                disp = AGENT_DISPLAY_NAMES[reg_key]
                if isinstance(agent, cls) and reg_key in keys:
                    matched = True
                    break
            was_active = agent.active
            agent.active = matched
            # Re-activated agents: clear halted flag AND reset the per-agent
            # peak value so the circuit-breaker doesn't immediately re-fire
            # (the peak was set when the agent last ran; the market may have
            # moved since then, making the drawdown look >10% straight away).
            if agent.active and not was_active:
                agent.halted = False
                agent._peak_value = agent.get_portfolio_value(close, self.ticker)
            elif agent.active and agent.halted:
                # Already active but halted (manual un-halt): same reset
                agent.halted = False
                agent._peak_value = agent.get_portfolio_value(close, self.ticker)

        # Recompute peak based on the NEW active set only.
        # Using max(old_peak, new_sum) is wrong — the old peak included
        # disabled agents' funds, so the active-only sum would look like a
        # massive drawdown and immediately fire the global circuit breaker.
        # Instead, start a fresh drawdown window from the current active total.
        active_total = sum(
            a.get_portfolio_value(close, self.ticker) for a in self.agents if a.active
        )
        self._peak_total_value = active_total if active_total > 0 else self._peak_total_value

        # Clear any latched global halt so trading resumes after the
        # user reconfigures the active-agent set.
        self.trading_halted = False

        # Also update stored active keys for reinit consistency
        self._active_agent_keys = keys
        return self.get_snapshot()

    def liquidate_agent(self, agent_key: str) -> dict:
        """
        Immediately sell all open positions for the given agent at the
        current market price, then return the updated snapshot.

        Called when the user clicks "Sell & Disable" in the UI.
        The agent is NOT deactivated here — the caller should follow up
        with set_active_agents() to actually stop the agent from trading.
        """
        if self.market is None:
            return {"error": "Simulation not initialised."}

        close = self.market.current_price
        agent_key_lower = agent_key.lower()

        target = None
        for agent in self.agents:
            for reg_key, cls in AGENT_REGISTRY.items():
                if reg_key == agent_key_lower and isinstance(agent, cls):
                    target = agent
                    break
            if target:
                break

        if target is None:
            return {"error": f"Agent '{agent_key}' not found."}

        # Sell every ticker the agent holds
        for ticker, qty in list(target.positions.items()):
            if qty > 0:
                target.execute_action(
                    {"action": "SELL", "ticker": ticker, "quantity": qty,
                     "reasoning": "Liquidation requested by user before disable."},
                    close,
                )

        return self.get_snapshot()

    # ------------------------------------------------------------------ #
    # Batch step
    # ------------------------------------------------------------------ #

    def batch_step(self, n: int = 10) -> dict:
        """Run *n* steps in one call, return the final snapshot."""
        n = min(int(n), 200)
        snapshot = None
        for _ in range(n):
            snapshot = self.run_step()
            if snapshot.get("finished") or "error" in snapshot:
                break
        return snapshot or self.get_snapshot()

    # ------------------------------------------------------------------ #
    # Optimization (headless parameter sweep)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_nested_param(target: dict, path: str, value: float) -> None:
        """Set nested dict value by dot path (creates missing keys)."""
        if not path:
            raise ValueError("parameter path is required")
        parts = [p for p in str(path).split(".") if p]
        if not parts:
            raise ValueError("parameter path is invalid")
        node = target
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value

    @staticmethod
    def _sweep_values(min_value: float, max_value: float, step_value: float) -> list[float]:
        vals = []
        v = float(min_value)
        hi = float(max_value)
        step = float(step_value)
        if step <= 0:
            raise ValueError("step must be > 0")
        if v > hi:
            raise ValueError("min must be <= max")
        guard = 0
        while v <= hi + (step * 1e-6):
            vals.append(round(v, 10))
            v += step
            guard += 1
            if guard > 10_000:
                break
        return vals

    @staticmethod
    def _compute_max_drawdown_pct(values: list[float]) -> float:
        if not values:
            return 0.0
        peak = values[0]
        worst_dd = 0.0
        for val in values:
            if val > peak:
                peak = val
            if peak > 0:
                dd = (val - peak) / peak
                if dd < worst_dd:
                    worst_dd = dd
        return abs(worst_dd * 100.0)

    def optimize(
        self,
        ticker: str,
        period: str,
        interval: str,
        active_agents: list[str] | None,
        agent_params: dict | None,
        parameter: str,
        min_value: float,
        max_value: float,
        step_value: float,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Run a headless parameter sweep and return final-value/max-drawdown metrics."""
        values = self._sweep_values(min_value, max_value, step_value)
        base_params = copy.deepcopy(agent_params or {})
        base_active = list(active_agents or AGENT_REGISTRY.keys())

        results = []
        for val in values:
            try:
                trial_params = copy.deepcopy(base_params)
                self._set_nested_param(trial_params, parameter, val)

                # Run trial in an isolated orchestrator with DB disabled for speed.
                runner = OrchestratorAgent(db=None)
                snapshot = runner.init(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                    active_agents=base_active,
                    agent_params=trial_params,
                    start_date=start_date,
                    end_date=end_date,
                    custom_data_df=None,
                )
                if "error" in snapshot:
                    results.append({
                        "parameter_value": val,
                        "error": snapshot["error"],
                    })
                    continue

                value_series: list[float] = []
                trial_finished = False
                while not trial_finished:
                    total_value = sum(
                        float(a.get("portfolio_value", 0.0))
                        for a in snapshot.get("agents", [])
                        if a.get("status") != "DISABLED"
                    )
                    value_series.append(total_value)
                    trial_finished = bool(snapshot.get("finished"))
                    if trial_finished:
                        break
                    snapshot = runner.run_step()
                    if "error" in snapshot:
                        break

                final_value = value_series[-1] if value_series else 0.0
                initial_value = value_series[0] if value_series else 1.0
                final_return_pct = ((final_value - initial_value) / initial_value * 100.0) if initial_value > 0 else 0.0
                max_drawdown_pct = self._compute_max_drawdown_pct(value_series)

                trial_result = {
                    "parameter_value": val,
                    "final_portfolio_value": round(final_value, 2),
                    "final_return_pct": round(final_return_pct, 4),
                    "max_drawdown_pct": round(max_drawdown_pct, 4),
                    "steps": max(0, len(value_series) - 1),
                }
                if "error" in snapshot:
                    trial_result["error"] = snapshot["error"]
                results.append(trial_result)
            except Exception as trial_exc:
                results.append({
                    "parameter_value": val,
                    "error": str(trial_exc),
                })

        return {
            "parameter": parameter,
            "min": min_value,
            "max": max_value,
            "step": step_value,
            "results": results,
        }

    # ------------------------------------------------------------------ #
    # Jump to step
    # ------------------------------------------------------------------ #

    def _reinit_without_download(self) -> None:
        """
        Reset the simulation to step 0 **without** re-downloading
        market data from yfinance.  Used by jump_to_step for rewinding.
        """
        # Reset market environment in-place (no network call)
        self.market.reset()
        self.current_step = 0
        self.max_steps = self.market.total_bars
        self.finished = False
        self.trades_at_step = []

        # Reset crash state
        self.crash_active = False
        self._crash_steps_remaining = 0
        self._crash_vol_multiplier = 1.0
        self.circuit_breakers_active = 0
        self.crash_mode_active = False
        self._crash_triggered_step = None
        self._pre_crash_price = None
        self.trading_halted = False

        # Reset in-memory logs
        self.logger.reset()

        # Re-create agents with fresh state
        self.agents = []
        for key in AGENT_REGISTRY:
            display = AGENT_DISPLAY_NAMES[key]
            cls = AGENT_REGISTRY[key]
            params = self._agent_params.get(key, {})
            # Allow the Custom agent to carry a user-defined display name
            agent_name = params.get("name", display) if key == "custom" else display
            agent = cls(agent_name, initial_cash=100_000.0, params=params)
            agent.active = key in self._active_agent_keys
            self.agents.append(agent)

        # Re-create regulator
        self.regulator = RegulatorAgent()

        # Seed price history with initial bar
        state = self.market.get_state()
        self.price_history = [state["current_bar"]]

        close = state["current_bar"]["Close"]
        for agent in self.agents:
            pv = agent.get_portfolio_value(close, self.ticker)
            agent.portfolio_value_history.append(pv)

        # BUG FIX: same as init() — only active agents contribute to peak value.
        self._peak_total_value = sum(
            a.get_portfolio_value(close, self.ticker) for a in self.agents if a.active
        )

        # New run in DB
        self.run_id = str(uuid.uuid4())
        if self.db:
            self.db.create_run(self.run_id, self.ticker, self.period, self.interval)
        self.logger.set_db(self.db, self.run_id, self.ticker)

    def jump_to_step(self, target_step: int) -> dict:
        """
        Fast-forward (or rewind) to a specific step.
        If the target is behind the current step, resets in-place
        (no yfinance re-download) and replays.
        DB writes are disabled during replay to avoid SQLite errors.
        """
        if self.market is None:
            return {"error": "Simulation not initialised."}

        target_step = min(int(target_step), self.max_steps - 1)
        target_step = max(0, target_step)

        if target_step <= self.current_step:
            # Reset without re-downloading market data
            self._reinit_without_download()

        # Disable DB writes during replay (they are redundant and cause
        # SQLite "cannot commit" errors on rapid sequential inserts)
        saved_db = self.logger._db
        self.logger._db = None

        try:
            while self.current_step < target_step and not self.finished:
                self.run_step()
        finally:
            # Always restore DB writes regardless of errors
            self.logger._db = saved_db

        return self.get_snapshot()

    # ------------------------------------------------------------------ #
    # Whale-crash helpers
    # ------------------------------------------------------------------ #

    def _find_whale_agent(self) -> TradingAgent | None:
        """Return the first AdversarialAgent instance, or *None*."""
        for agent in self.agents:
            if isinstance(agent, AdversarialAgent):
                return agent
        return None

    def _is_whale(self, agent: TradingAgent) -> bool:
        """True if *agent* is the whale (Adversarial) agent."""
        return isinstance(agent, AdversarialAgent)

    def _build_whale_dump(self, whale: TradingAgent) -> dict:
        """Build a forced full-dump SELL decision for the whale agent."""
        qty = whale.positions.get(self.ticker, 0)
        reasoning = (
            "Forced whale dump: 100% position liquidation for crash demo."
        )
        return {
            "action": "SELL",
            "ticker": self.ticker,
            "quantity": qty,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------ #
    # Trigger crash  (whale manipulation → cascade → halt)
    # ------------------------------------------------------------------ #

    def trigger_crash(self) -> dict:
        """Public entry – called by the Simulation facade / API."""
        return self.trigger_market_crash()

    def trigger_market_crash(self) -> dict:
        """
        Demo scenario – Whale Manipulation / Cascade Crash:

        1. Force the Adversarial "whale" agent to dump 100 % of its
           holdings in one tick.
        2. The large sell order flows through the normal pipeline:
           agent action → regulator → market step with big negative
           net_volume.
        3. Subsequent ticks show momentum / other agents reacting
           (stop-loss, trend reversal).
        4. If drawdown or volatility exceeds thresholds, the
           orchestrator activates the global circuit breaker.

        The method records the crash step and activates
        ``crash_mode_active`` so that ``run_step()`` can apply
        cascade behaviour on following ticks.
        """
        if self.market is None:
            return {"error": "Simulation not initialised."}

        whale = self._find_whale_agent()

        # If whale has no position yet, give it a large one so the
        # dump is visible.  This makes the demo reliable even when
        # triggered early.
        if whale and whale.active:
            whale_qty = whale.positions.get(self.ticker, 0)
            if whale_qty == 0:
                close = self.market.current_price
                buy_qty = int(whale.cash * 0.90 / close) if close > 0 else 0
                if buy_qty > 0:
                    whale.execute_action(
                        {"action": "BUY", "ticker": self.ticker,
                         "quantity": buy_qty,
                         "reasoning": "Pre-crash whale position build-up."},
                        close,
                    )
        elif whale and not whale.active:
            # Re-enable the whale for the crash demo
            whale.active = True
            whale.halted = False

        # Also drop the underlying OHLC data 15-20 % from the current
        # step onward so technical indicators reflect a real crash.
        crash_pct = random.uniform(0.15, 0.20)
        step = self.market.current_step
        df = self.market.df

        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df.loc[step:, col] = df.loc[step:, col] * (1 - crash_pct)

        # Recompute technical indicators on crashed data
        close_series = df["Close"]
        df["SMA20"] = close_series.rolling(window=20, min_periods=1).mean()
        df["SMA50"] = close_series.rolling(window=50, min_periods=1).mean()
        rolling_std = close_series.rolling(window=20, min_periods=1).std()
        df["BB_MID"] = df["SMA20"]
        df["BB_UP"] = df["SMA20"] + 2 * rolling_std
        df["BB_LOW"] = df["SMA20"] - 2 * rolling_std
        log_returns = np.log(close_series / close_series.shift(1))
        df["Volatility"] = log_returns.rolling(window=20, min_periods=1).std() * 3
        df.fillna(0, inplace=True)

        # Record pre-crash price for cascade-drop tracking
        self._pre_crash_price = self.market.current_price

        # Mark crash mode
        self.crash_active = True
        self.crash_mode_active = True
        self._crash_triggered_step = self.current_step
        self._crash_steps_remaining = 10
        self._crash_vol_multiplier = 3.0

        # Update price history bar to reflect new crashed OHLC
        if step < len(df):
            self.price_history[-1] = self.market._bar_to_dict(step)

        # Run one step so the whale dump + cascade happen immediately
        return self.run_step()

    # ------------------------------------------------------------------ #
    # System-wide risk metrics
    # ------------------------------------------------------------------ #

    def get_system_risk(self) -> dict:
        """
        Compute system-wide risk overview.

        Includes: total exposure, open positions, global drawdown,
        violation count, circuit-breaker status.
        """
        if self.market is None:
            return {}

        state = self.market.get_state()
        close = state["current_bar"]["Close"]

        total_exposure = 0.0
        open_positions = 0
        active_count = 0

        for agent in self.agents:
            if not agent.active:
                continue
            active_count += 1
            for _t, qty in agent.positions.items():
                total_exposure += qty * close
                if qty > 0:
                    open_positions += 1

        total_aum = sum(
            a.get_portfolio_value(close, self.ticker)
            for a in self.agents if a.active
        )
        exposure_pct = (total_exposure / total_aum * 100) if total_aum > 0 else 0

        # Global drawdown
        if total_aum > self._peak_total_value:
            self._peak_total_value = total_aum
        global_dd = (
            (total_aum - self._peak_total_value) / self._peak_total_value * 100
        ) if self._peak_total_value > 0 else 0

        violation_count = len(self.logger.get_regulation_log())

        return {
            "total_exposure": round(total_exposure, 2),
            "total_aum": round(total_aum, 2),
            "exposure_pct": round(exposure_pct, 1),
            "open_positions_count": open_positions,
            "active_agents": active_count,
            "violation_count": violation_count,
            "global_drawdown_pct": round(global_dd, 2),
            "circuit_breakers_active": self.circuit_breakers_active,
            "crash_active": self.crash_active,
        }

    # ------------------------------------------------------------------ #
    # Humanized Market Summary (Part 4)
    # ------------------------------------------------------------------ #

    def generate_market_summary(self) -> str:
        """
        Produce a plain-English narrative of the current simulation state.
        Strictly rule-based – no LLMs or external calls.
        """
        if not self.agents or self.market is None:
            return "Simulation not yet started."

        trade_log = self.logger.get_trade_log()
        reg_log = self.logger.get_regulation_log()

        # Tally volume × followers per agent
        agent_impact: dict[str, float] = {}
        agent_buys: dict[str, int] = {}
        agent_sells: dict[str, int] = {}
        agent_followers: dict[str, int] = {}

        for agent in self.agents:
            agent_followers[agent.name] = max(1, getattr(agent, "followers", 1))
            agent_impact[agent.name] = 0.0
            agent_buys[agent.name] = 0
            agent_sells[agent.name] = 0

        for entry in trade_log:
            name = entry.get("agent_name", "")
            action = entry.get("action", "")
            qty = entry.get("quantity", 0) or 0
            followers = agent_followers.get(name, 1)
            if action in ("BUY", "SELL"):
                agent_impact[name] = agent_impact.get(name, 0) + qty * followers
            if action == "BUY":
                agent_buys[name] = agent_buys.get(name, 0) + 1
            elif action == "SELL":
                agent_sells[name] = agent_sells.get(name, 0) + 1

        # Find highest-impact agent
        top_agent = max(agent_impact, key=lambda k: agent_impact[k], default=None)
        top_impact = agent_impact.get(top_agent, 0)

        # Count violations per agent
        violations_by_agent: dict[str, int] = {}
        for entry in reg_log:
            name = entry.get("agent_name", "SYSTEM")
            violations_by_agent[name] = violations_by_agent.get(name, 0) + 1

        most_violated_agent = max(violations_by_agent, key=lambda k: violations_by_agent[k], default=None)
        violation_count = violations_by_agent.get(most_violated_agent, 0) if most_violated_agent else 0

        # Find best / worst performer
        close = self.market.current_price
        returns = {
            a.name: ((a.get_portfolio_value(close, self.ticker) - a.initial_cash) / a.initial_cash * 100)
            for a in self.agents if a.active
        }
        best = max(returns, key=lambda k: returns[k], default=None)
        worst = min(returns, key=lambda k: returns[k], default=None)

        # Build narrative
        parts: list[str] = []

        step_word = "step" if self.current_step == 1 else "steps"
        parts.append(f"After {self.current_step} {step_word} of trading on {self.ticker}:")

        if top_agent and top_impact > 0:
            followers_n = agent_followers.get(top_agent, 1)
            direction = "bullish" if agent_buys.get(top_agent, 0) >= agent_sells.get(top_agent, 0) else "bearish"
            follower_note = f", backed by {followers_n} follower{'s' if followers_n != 1 else ''}" if followers_n > 1 else ""
            parts.append(
                f"The {top_agent} strategy{follower_note} had the largest market impact "
                f"({int(top_impact):,} effective volume units, {direction} net bias)."
            )

        if best and worst and best != worst:
            best_ret = returns.get(best, 0)
            worst_ret = returns.get(worst, 0)
            parts.append(
                f"{best} leads the pack at {best_ret:+.1f}% return, "
                f"while {worst} trails at {worst_ret:+.1f}%."
            )

        if most_violated_agent and violation_count > 0:
            parts.append(
                f"{most_violated_agent} triggered {violation_count} regulatory violation{'s' if violation_count != 1 else ''}."
            )

        if self.trading_halted:
            parts.append("⚠ A global circuit breaker has halted all trading.")
        elif self.crash_active:
            parts.append("⚠ A market crash event is currently active.")

        if self.circuit_breakers_active:
            parts.append(
                f"{self.circuit_breakers_active} agent{'s are' if self.circuit_breakers_active > 1 else ' is'} "
                f"halted by individual circuit breakers."
            )

        return " ".join(parts) if parts else "Simulation running normally."

    # ------------------------------------------------------------------ #
    # Snapshot for frontend
    # ------------------------------------------------------------------ #

    def get_snapshot(self) -> dict:
        """
        Build a JSON-serialisable snapshot of the full simulation state.

        Includes the `head_agent` block (OrchestratorAgent metadata) required
        by DevHack 2026 PPT and the enriched per-agent status.
        """
        if self.market is None:
            return {"error": "Simulation not initialised."}

        state = self.market.get_state()
        close = state["current_bar"]["Close"]

        # Per-agent data with enriched status field
        agents_data = []
        for agent in self.agents:
            d = agent.to_dict(close, self.ticker)
            # Add explicit status label for PPT / UI
            if not agent.active:
                d["status"] = "DISABLED"
            elif agent.halted:
                d["status"] = "HALTED"
            else:
                d["status"] = "ACTIVE"
            agents_data.append(d)

        return {
            # Orchestrator metadata – Head Agent info for DevHack PPT
            "head_agent": {
                "name": "OrchestratorAgent",
                "role": "Head Agent (Orchestrator) – coordinates all trading agents and the market.",
                "current_step": self.current_step,
                "max_steps": self.max_steps,
                "crash_active": self.crash_active,
                "crash_mode_active": self.crash_mode_active,
                "trading_halted": self.trading_halted,
                "circuit_breakers_active": self.circuit_breakers_active,
                "run_id": self.run_id,
            },
            # Simulation state
            "step": self.current_step,
            "max_steps": self.max_steps,
            "ticker": self.ticker,
            "period": self.period,
            "interval": self.interval,
            "finished": self.finished,
            "current_bar": state["current_bar"],
            "price_history": self.price_history,
            # Agent data
            "agents": agents_data,
            # Logs
            "trade_log": self.logger.get_trade_log(),
            "regulation_log": self.logger.get_regulation_log(),
            "trades_at_step": self.trades_at_step,
            # Risk
            "system_risk": self.get_system_risk(),
            # Crash state
            "crash_active": self.crash_active,
            "crash_mode_active": self.crash_mode_active,
            "trading_halted": self.trading_halted,
            "trading_status": (
                "HALTED_BY_CIRCUIT_BREAKER" if self.trading_halted else "ACTIVE"
            ),
            "circuit_breakers_active": self.circuit_breakers_active,
            # Humanized narrative summary
            "market_summary": self.generate_market_summary(),
        }
