"""
Regulator / Compliance module.
Reviews every proposed trade for rule violations before execution.

Regulator agent enforces rules:
    MaxPositionLimit, BurstTrading, ManipulationPattern, etc.

This is an **autonomous compliance agent** – it receives proposed trades
from the Orchestrator, evaluates them against a rule set, and returns
APPROVE / WARN / BLOCK decisions.  It is NOT a chatbot.
"""


class RegulatorAgent:
    """
    Regulator agent enforces rules: MaxPositionLimit, BurstTrading,
    ManipulationPattern, etc.

    **Goal**: Maintain market integrity by reviewing every proposed trade
    and blocking or adjusting those that violate compliance rules.

    **Rules enforced**:
        1. **MaxPositionLimit** – no agent may hold > 30 % of portfolio
           value in a single ticker.
        2. **MaxOrderSize** – single order cannot exceed 10 % of recent
           average volume.
        3. **ManipulationPattern / BurstTrading** – detects rapid-fire
           large orders (≥ 3 large orders in 5-step window) and blocks them.
        4. **AdversarialFlag** – large orders from adversarial agents are
           always flagged with a warning.
    """

    MAX_POSITION_PCT = 0.30        # max 30 % of portfolio value in one position
    MAX_ORDER_VOLUME_PCT = 0.10    # max 10 % of recent avg volume per trade
    BURST_WINDOW = 5               # look-back window (steps) for burst detection
    BURST_THRESHOLD = 3            # max large orders in window
    CRASH_DROP_PCT = -0.05         # >5 % drop from recent avg → crash condition
    CRASH_LOOKBACK = 10            # number of bars to average for crash baseline

    def __init__(self):
        # Track recent large orders per agent for burst detection
        self._recent_orders: dict[str, list[int]] = {}   # agent -> list of steps

    def review_trade(
        self,
        agent_name: str,
        action: dict,
        agent_state: dict,
        market_state: dict,
        current_step: int = 0,
    ) -> dict:
        """
        Review a proposed trade.

        Args:
            agent_name:   name of the agent proposing the trade
            action:       {"action": "BUY"/"SELL"/"HOLD", "quantity": int,
                           "reasoning": str, ...}
                          (legacy key "type" is also accepted)
            agent_state:  {"cash": float, "positions": dict, "portfolio_value": float}
            market_state: current_bar dict (Close, Volume, Volatility,
                          simulated_price, price_history_simulated, …)
            current_step: current simulation step

        Returns:
            {
                "decision": "APPROVE" | "WARN" | "BLOCK",
                "reason": str,
                "adjusted_action": dict   (same as action, or modified)
            }
        """
        # Support both new ("action") and legacy ("type") key names
        action_type = action.get("action") or action.get("type", "HOLD")
        quantity = action.get("quantity", 0)
        ticker = action.get("ticker", "")
        agent_reasoning = action.get("reasoning", "")

        # HOLD always approved
        if action_type == "HOLD" or quantity == 0:
            return {
                "decision": "APPROVE",
                "reason": "No trade to review (HOLD).",
                "adjusted_action": action,
            }

        close = market_state.get("Close", 0)
        volume = market_state.get("Volume", 1)
        portfolio_value = agent_state.get("portfolio_value", 1)
        current_positions = agent_state.get("positions", {})
        held_qty = current_positions.get(ticker, 0)

        reasons = []
        decision = "APPROVE"
        adjusted_action = dict(action)

        # ---- Rule 1: Max position size ----
        if action_type == "BUY":
            new_qty = held_qty + quantity
            position_value = new_qty * close
            if portfolio_value > 0 and position_value / portfolio_value > self.MAX_POSITION_PCT:
                max_allowed = int(
                    (self.MAX_POSITION_PCT * portfolio_value - held_qty * close) / close
                ) if close > 0 else 0
                max_allowed = max(0, max_allowed)
                reasons.append(
                    f"Position size rule: {new_qty} shares "
                    f"({position_value/portfolio_value*100:.1f}% of portfolio) "
                    f"exceeds {self.MAX_POSITION_PCT*100:.0f}% limit. "
                    f"Reduced to {max_allowed} shares."
                )
                adjusted_action["quantity"] = max_allowed
                decision = "WARN"

        # ---- Rule 2: Max order size vs volume ----
        avg_volume = max(volume, 1)
        if quantity > avg_volume * self.MAX_ORDER_VOLUME_PCT:
            allowed_qty = int(avg_volume * self.MAX_ORDER_VOLUME_PCT)
            allowed_qty = max(1, allowed_qty)
            reasons.append(
                f"Order size rule: {quantity} shares exceeds "
                f"{self.MAX_ORDER_VOLUME_PCT*100:.0f}% of volume ({avg_volume:.0f}). "
                f"Reduced to {allowed_qty}."
            )
            adjusted_action["quantity"] = min(
                adjusted_action["quantity"], allowed_qty
            )
            decision = "WARN"

        # ---- Rule 3: Manipulation / Burst detection ----
        is_large = quantity > (avg_volume * 0.05)  # >5% of volume = "large"
        if is_large:
            if agent_name not in self._recent_orders:
                self._recent_orders[agent_name] = []
            self._recent_orders[agent_name].append(current_step)
            # prune old entries
            self._recent_orders[agent_name] = [
                s for s in self._recent_orders[agent_name]
                if current_step - s <= self.BURST_WINDOW
            ]
            if len(self._recent_orders[agent_name]) >= self.BURST_THRESHOLD:
                reasons.append(
                    f"Manipulation alert: {agent_name} placed "
                    f"{len(self._recent_orders[agent_name])} large orders "
                    f"in the last {self.BURST_WINDOW} steps. Trade BLOCKED."
                )
                decision = "BLOCK"
                adjusted_action["type"] = "HOLD"
                adjusted_action["quantity"] = 0

        # Adversarial-specific flag
        if "adversarial" in agent_name.lower() and is_large and decision != "BLOCK":
            reasons.append(
                f"Adversarial agent '{agent_name}' placing large order – flagged."
            )
            if decision == "APPROVE":
                decision = "WARN"

        # ---- Rule 5: Contrarian trade during market crash ----
        if action_type == "BUY" and decision != "BLOCK":
            price_drop_pct = self._compute_crash_drop(market_state)
            if price_drop_pct < self.CRASH_DROP_PCT:
                reasons.append(
                    f"Market is down {price_drop_pct:.1%}, agent chose BUY. "
                    f"Risky contrarian behaviour: {agent_reasoning}"
                )
                if decision == "APPROVE":
                    decision = "WARN"

        reason_text = " | ".join(reasons) if reasons else "Trade compliant."

        # Violation / warn count at this step
        count_at_step = len(reasons)

        return {
            "decision": decision,
            "reason": reason_text,
            "adjusted_action": adjusted_action,
            "count_at_step": count_at_step,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_crash_drop(self, market_state: dict) -> float:
        """
        Compute the percentage change between the current simulated price
        and the average of the last ``CRASH_LOOKBACK`` simulated prices.

        Returns a negative float when the market is falling (e.g. -0.07
        for a 7 % drop).  Returns 0.0 if insufficient history is available.
        """
        current_price = market_state.get("simulated_price", 0)
        history = market_state.get("price_history_simulated", [])

        if not history or current_price <= 0:
            return 0.0

        lookback = history[-self.CRASH_LOOKBACK:]   # last N prices
        if len(lookback) < 2:
            return 0.0

        avg_recent = sum(lookback) / len(lookback)
        if avg_recent <= 0:
            return 0.0

        return (current_price - avg_recent) / avg_recent
