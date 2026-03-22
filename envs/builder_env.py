"""
Builder Trading Environment (MODERATE Risk Profile)
=====================================================
Goal: Steady growth with controlled volatility. Capture trends.

Reward shaping:
- 1.5x asymmetric loss penalty
- Drawdown penalty kicks in above 15%
- Trend-following bonus (reward for riding sustained moves)
- Moderate funding bleed awareness
"""

from __future__ import annotations

from envs.base_trading_env import BaseTradingEnv


class BuilderTradingEnv(BaseTradingEnv):
    """Moderate RL environment — The Builder."""

    def __init__(self, **kwargs):
        kwargs.setdefault("max_leverage", 2)
        kwargs.setdefault("max_positions", 4)
        kwargs.setdefault("max_sl_pct", 0.05)
        kwargs.setdefault("min_sl_pct", 0.01)
        kwargs.setdefault("max_tp_pct", 0.10)
        kwargs.setdefault("min_tp_pct", 0.02)
        kwargs.setdefault("max_drawdown_pct", 0.20)  # 20% kill — matches risk_manager.py
        kwargs.setdefault("profile_name", "builder")
        super().__init__(**kwargs)

        self._prev_pnl = 0.0  # Track trend continuity

    def _calculate_reward(self, ctx: dict) -> float:
        pnl = ctx["pnl_pct"]
        drawdown = ctx["drawdown"]
        has_position = ctx["has_position"]

        reward = 0.0

        # --- Core PnL reward with 1.5x loss asymmetry ---
        if pnl < 0:
            reward += pnl * 1.5
        else:
            reward += pnl

        # --- Trend-following bonus ---
        # Reward consecutive profitable steps (riding a trend)
        if pnl > 0 and self._prev_pnl > 0 and has_position:
            reward += pnl * 0.2  # 20% bonus for sustained gains
        self._prev_pnl = pnl

        # --- Drawdown penalty (escalating — episode terminates at 20% via base env) ---
        if drawdown > 0.15:
            reward -= (drawdown - 0.15) * 10.0  # Steep ramp from 15-20%
        elif drawdown > 0.10:
            reward -= (drawdown - 0.10) * 2.0  # Gentle ramp from 10-15%

        # --- Funding bleed (moderate awareness) ---
        reward -= abs(ctx.get("funding_cost", 0)) * 0.0005

        # --- Small time penalty to encourage activity ---
        if not has_position:
            reward -= 0.000005

        return reward

    def reset(self, **kwargs):
        self._prev_pnl = 0.0
        return super().reset(**kwargs)
