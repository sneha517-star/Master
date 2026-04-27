from __future__ import annotations

"""
Mean-Reversion Agent – buys at lower Bollinger Band, sells at upper.

This agent is an **autonomous, goal-driven, rule-based decision maker**.
"""

from agents.base_agent import TradingAgent


class MeanReversionAgent(TradingAgent):
    """
    Autonomous Mean-Reversion Trading Agent.

    **Goal**: Profit from the statistical tendency of prices to revert
    to their moving average after extreme deviations.

    **Inputs**:
        - Current price (Close)
        - Bollinger Band Mid (BB_MID = SMA20)
        - Bollinger Band Upper (BB_UP) and Lower (BB_LOW)

    **Decision logic**:
        1. If price < BB_LOW → price is oversold → BUY (default 12 %
           of cash).
        2. If price > BB_UP and holding → price overbought → SELL
           entire position.
        3. Otherwise → HOLD (price is within normal bands).
    """

    def __init__(self, name: str, initial_cash: float = 100_000.0, params: dict | None = None):
        super().__init__(name, initial_cash)
        params = params or {}
        self.POSITION_FRACTION = params.get("position_size_pct", 0.12)
        self.BAND_MULTIPLIER = params.get("band_multiplier", 2.0)
        self.followers = int(params.get("followers", 1))

    def perceive(self, market_state: dict) -> dict:
        super().perceive(market_state)
        bar = market_state.get("current_bar", {})
        ticker = bar.get("ticker", "")
        close = bar.get("Close", 0.0)
        bb_mid = bar.get("BB_MID", close)
        
        # Recompute custom bands using configurable multiplier
        # BB_UP/BB_LOW in data use 2σ; scale to our multiplier
        default_up = bar.get("BB_UP", close)
        default_low = bar.get("BB_LOW", close)
        half_width = (default_up - bb_mid) if bb_mid else 0
        scale = self.BAND_MULTIPLIER / 2.0 if half_width else 1.0
        bb_up = bb_mid + half_width * scale
        bb_low = bb_mid - half_width * scale

        return {
            "ticker": ticker,
            "close": close,
            "bb_mid": bb_mid,
            "bb_up": bb_up,
            "bb_low": bb_low,
            "held_qty": self.positions.get(ticker, 0)
        }

    def reason(self, observation: dict) -> dict:
        if not observation or not observation.get("ticker"):
            return {"action": "HOLD", "ticker": "", "quantity": 0, "reasoning": "No valid observation"}

        ticker = observation["ticker"]
        close = observation["close"]
        bb_mid = observation["bb_mid"]
        bb_up = observation["bb_up"]
        bb_low = observation["bb_low"]
        held_qty = observation["held_qty"]

        # ---------- Oversold → BUY ----------
        if close < bb_low:
            affordable = int(
                (self.cash * self.POSITION_FRACTION) / close
            ) if close > 0 else 0
            if affordable > 0:
                reasoning = (
                    f"Price {close:.2f} < BB_LOW {bb_low:.2f} → "
                    f"oversold, mean-reversion BUY "
                    f"(BB_MID={bb_mid:.2f}, BB_UP={bb_up:.2f})"
                )
                return {"action": "BUY", "ticker": ticker, "quantity": affordable, "reasoning": reasoning}

        # ---------- Overbought → SELL ----------
        if close > bb_up and held_qty > 0:
            reasoning = (
                f"Price {close:.2f} > BB_UP {bb_up:.2f} → "
                f"overbought, closing {held_qty} shares "
                f"(BB_MID={bb_mid:.2f}, BB_LOW={bb_low:.2f})"
            )
            return {"action": "SELL", "ticker": ticker, "quantity": held_qty, "reasoning": reasoning}

        reasoning = (
            f"HOLD – price {close:.2f} within bands "
            f"[{bb_low:.2f}, {bb_up:.2f}]"
        )
        return {"action": "HOLD", "ticker": ticker, "quantity": 0, "reasoning": reasoning}
