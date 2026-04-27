from __future__ import annotations

"""
Conservative Agent – low-risk, small positions, only trades in calm markets.
Supports configurable risk_pct and stop_loss_pct via params dict.

This agent is an **autonomous, goal-driven, rule-based decision maker**.
"""

from agents.base_agent import TradingAgent


class ConservativeAgent(TradingAgent):
    """
    Autonomous Conservative Trading Agent.

    **Goal**: Preserve capital and achieve small, steady returns by only
    entering positions during low-volatility (calm) market conditions.

    **Inputs**:
        - Current price (Close)
        - SMA20, SMA50 (trend indicators)
        - Rolling Volatility (risk filter)
        - Average cost basis (for stop-loss)

    **Decision logic**:
        1. If holding and price drops below stop-loss threshold → SELL all.
        2. If volatility > threshold → HOLD (market too risky).
        3. If price < SMA50 AND SMA20 > SMA50 (mild uptrend, not overextended)
           → BUY a small position (default 7 % of cash).
        4. Otherwise → HOLD.
    """

    def __init__(self, name: str, initial_cash: float = 100_000.0, params: dict | None = None):
        super().__init__(name, initial_cash)
        params = params or {}
        self.VOLATILITY_THRESHOLD = params.get("volatility_threshold", 0.02)
        self.POSITION_FRACTION = params.get("risk_pct", 0.07)
        self.STOP_LOSS_PCT = params.get("stop_loss_pct", 0.03)
        self.followers = int(params.get("followers", 1))

    def perceive(self, market_state: dict) -> dict:
        super().perceive(market_state)
        bar = market_state.get("current_bar", {})
        ticker = bar.get("ticker", "")
        close = bar.get("Close", 0.0)
        return {
            "ticker": ticker,
            "close": close,
            "sma20": bar.get("SMA20", close),
            "sma50": bar.get("SMA50", close),
            "volatility": bar.get("Volatility", 0),
            "held_qty": self.positions.get(ticker, 0),
            "avg_cost": self.avg_cost.get(ticker, 0),
        }

    def reason(self, observation: dict) -> dict:
        if not observation or not observation.get("ticker"):
            return {"action": "HOLD", "ticker": "", "quantity": 0, "reasoning": "No valid observation"}

        ticker = observation["ticker"]
        close = observation["close"]
        sma20 = observation["sma20"]
        sma50 = observation["sma50"]
        vol = observation["volatility"]
        held_qty = observation["held_qty"]
        avg = observation["avg_cost"]

        if held_qty > 0 and avg > 0:
            if close < avg * (1 - self.STOP_LOSS_PCT):
                reasoning = (
                    f"Stop-loss triggered: price {close:.2f} < "
                    f"{avg*(1-self.STOP_LOSS_PCT):.2f} "
                    f"(avg_cost {avg:.2f} - {self.STOP_LOSS_PCT*100}%)"
                )
                return {"action": "SELL", "ticker": ticker, "quantity": held_qty, "reasoning": reasoning}

        if vol > self.VOLATILITY_THRESHOLD:
            reasoning = (
                f"HOLD – volatility {vol:.4f} exceeds threshold "
                f"{self.VOLATILITY_THRESHOLD}"
            )
            return {"action": "HOLD", "ticker": ticker, "quantity": 0, "reasoning": reasoning}

        if close < sma50 and sma20 > sma50 and held_qty == 0:
            affordable = int(
                (self.cash * self.POSITION_FRACTION) / close
            ) if close > 0 else 0
            if affordable > 0:
                reasoning = (
                    f"Low volatility ({vol:.4f}), price {close:.2f} < SMA50 "
                    f"{sma50:.2f}, SMA20 {sma20:.2f} > SMA50 → small long entry"
                )
                return {"action": "BUY", "ticker": ticker, "quantity": affordable, "reasoning": reasoning}

        reasoning = "HOLD – conditions not met for conservative entry"
        return {"action": "HOLD", "ticker": ticker, "quantity": 0, "reasoning": reasoning}
