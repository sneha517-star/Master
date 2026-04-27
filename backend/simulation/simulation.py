from __future__ import annotations

"""
Simulation facade / API layer.

Thin wrapper around OrchestratorAgent (the Head Agent) that preserves
the existing REST-API interface expected by the React frontend.

All core simulation logic now lives in OrchestratorAgent
(see simulation/orchestrator.py).  This module delegates every call
to the orchestrator so that the Flask endpoints remain unchanged.

DevHack 2026 architecture:
    Flask API  ──►  Simulation (facade)  ──►  OrchestratorAgent (Head Agent)
                                                 │
                                      ┌──────────┼──────────┐
                                      ▼          ▼          ▼
                                   Agents    Regulator   SQLite DB
"""

from simulation.orchestrator import OrchestratorAgent
from db import SimulationDB
import pandas as pd


class Simulation:
    """
    Stateful simulation facade.

    Lifecycle (unchanged from the frontend's perspective):
        1. init_simulation(ticker, period, interval, ...)  → downloads data, creates agents
        2. step_simulation()  → advance one bar
        3. get_snapshot()     → return full state for the frontend

    Internally all work is delegated to ``self.orchestrator`` – the
    **Head Agent (Orchestrator)** required by DevHack 2026 guidelines.
    """

    def __init__(self):
        # SQLite storage layer used for trade and regulation logs (post-hoc analysis).
        self.db = SimulationDB()

        # Head Agent (Orchestrator) – coordinates all trading agents and the market.
        self.orchestrator = OrchestratorAgent(db=self.db)

    # ------------------------------------------------------------------ #
    # Delegated public API  (keeps REST endpoints working as-is)
    # ------------------------------------------------------------------ #

    def init_simulation(
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
        """Initialise / re-initialise the simulation.  Delegates to OrchestratorAgent."""
        return self.orchestrator.init(
            ticker, period, interval,
            active_agents=active_agents,
            agent_params=agent_params,
            start_date=start_date,
            end_date=end_date,
            custom_data_df=custom_data_df,
        )

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
        """Run headless optimization sweep. Delegates to OrchestratorAgent.optimize()."""
        return self.orchestrator.optimize(
            ticker=ticker,
            period=period,
            interval=interval,
            active_agents=active_agents,
            agent_params=agent_params,
            parameter=parameter,
            min_value=min_value,
            max_value=max_value,
            step_value=step_value,
            start_date=start_date,
            end_date=end_date,
        )

    def step_simulation(self) -> dict:
        """Advance one bar.  Delegates to OrchestratorAgent.run_step()."""
        return self.orchestrator.run_step()

    def batch_step(self, n: int = 10) -> dict:
        """Run *n* steps in one call.  Delegates to OrchestratorAgent."""
        return self.orchestrator.batch_step(n)

    def jump_to_step(self, target_step: int) -> dict:
        """Jump to a specific step.  Delegates to OrchestratorAgent."""
        return self.orchestrator.jump_to_step(target_step)

    def trigger_crash(self) -> dict:
        """Trigger a market crash event.  Delegates to OrchestratorAgent."""
        return self.orchestrator.trigger_crash()

    def set_active_agents(self, active_agents: list[str]) -> dict:
        """Enable/disable agents mid-simulation without re-init."""
        return self.orchestrator.set_active_agents(active_agents)

    def liquidate_agent(self, agent_key: str) -> dict:
        """Sell all open positions for an agent at current market price."""
        return self.orchestrator.liquidate_agent(agent_key)

    def get_snapshot(self) -> dict:
        """Return current simulation snapshot.  Delegates to OrchestratorAgent."""
        return self.orchestrator.get_snapshot()
