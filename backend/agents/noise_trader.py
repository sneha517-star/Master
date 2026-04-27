from __future__ import annotations

"""
Noise Trader – random actions to inject realistic market noise.

This agent is an **autonomous, goal-driven, rule-based decision maker**
(goal: inject realistic irrational trading activity into the market).
"""

import random
from agents.base_agent import TradingAgent


class NoiseTrader(TradingAgent):
    """
    Autonomous Noise Trading Agent.

    **Goal**: Simulate irrational / retail-style market activity by
    placing small random trades, adding realistic noise to the
    multi-agent simulation.

    **Inputs**:
        - Current price (Close)
        - Internal random number generator
        - Current position (for sell decisions)

    **Decision logic**:
        1. Each step, with probability ``trade_probability`` (default 15 %),
           decide to trade; otherwise HOLD.
        2. If trading: 50 % chance BUY a small random qty (up to 2 % of cash),
           50 % chance SELL a random portion of holdings.
        3. Provides a ``last_reason`` string explaining the random action.
    """

    def __init__(self, name: str, initial_cash: float = 100_000.0, params: dict | None = None):
        super().__init__(name, initial_cash)
        params = params or {}
        self.TRADE_PROBABILITY = params.get("trade_probability", 0.15)
        self.POSITION_FRACTION = params.get("position_size_pct", 0.02)
        self.followers = int(params.get("followers", 1))

    def perceive(self, market_state: dict) -> dict:
        super().perceive(market_state)
        bar = market_state.get("current_bar", {})
        ticker = bar.get("ticker", "")
        close = bar.get("Close", 0.0)
        return {
            "ticker": ticker,
            "close": close,
            "held_qty": self.positions.get(ticker, 0),
            "trade_roll": random.random(),
            "direction_roll": random.random(),
        }

    def reason(self, observation: dict) -> dict:
        if not observation or not observation.get("ticker"):
            return {"action": "HOLD", "ticker": "", "quantity": 0, "reasoning": "No valid observation"}

        ticker = observation["ticker"]
        close = observation["close"]
        held_qty = observation["held_qty"]

        if observation["trade_roll"] > self.TRADE_PROBABILITY:
            return {"action": "HOLD", "ticker": ticker, "quantity": 0, "reasoning": "No action this step (random skip)"}

        if observation["direction_roll"] < 0.5:
            affordable = int(
                (self.cash * self.POSITION_FRACTION) / close
            ) if close > 0 else 0
            if affordable > 0:
                qty = random.randint(1, max(1, affordable))
                reasoning = f"Random noise BUY of {qty} shares"
                return {"action": "BUY", "ticker": ticker, "quantity": qty, "reasoning": reasoning}
        else:
            if held_qty > 0:
                sell_qty = random.randint(1, max(1, held_qty))
                reasoning = f"Random noise SELL of {sell_qty} shares"
                return {"action": "SELL", "ticker": ticker, "quantity": sell_qty, "reasoning": reasoning}

        reasoning = "Random action considered but no position to sell / insufficient cash"
        return {"action": "HOLD", "ticker": ticker, "quantity": 0, "reasoning": reasoning}
