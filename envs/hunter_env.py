"""
Hunter Trading Environment (AGGRESSIVE Risk Profile)
======================================================
Goal: Aggressive capital appreciation. Maximize absolute profit.

Reward shaping:
- No loss asymmetry (symmetric PnL)
- Large-win multiplier (bonus for >5% gains)
- Soft drawdown penalty only past 25%
- No funding bleed penalty (Hunter rides positions)
- No time penalty (Hunter can wait for big setups)
"""

from __future__ import annotations

from envs.base_trading_env import BaseTradingEnv


class HunterTradingEnv(BaseTradingEnv):
    """Aggressive RL environment — The Hunter."""

    def __init__(self, **kwargs):
        kwargs.setdefault("max_leverage", 3)
        kwargs.setdefault("max_positions", 5)
        kwargs.setdefault("max_sl_pct", 0.07)
        kwargs.setdefault("min_sl_pct", 0.02)
        kwargs.setdefault("max_tp_pct", 0.15)
        kwargs.setdefault("min_tp_pct", 0.03)
        kwargs.setdefault("profile_name", "hunter")
        super().__init__(**kwargs)

    def _calculate_reward(self, ctx: dict) -> float:
        pnl = ctx["pnl_pct"]
        drawdown = ctx["drawdown"]

        reward = 0.0

        # --- Symmetric PnL (no loss penalty multiplier) ---
        reward += pnl

        # --- Large-win multiplier ---
        if pnl > 0.05:
            reward += pnl * 0.5  # 50% bonus for >5% single-step gains
        elif pnl > 0.03:
            reward += pnl * 0.2  # 20% bonus for >3% gains

        # --- Soft drawdown penalty only past 25% ---
        if drawdown > 0.35:
            reward -= 1.0  # Hard penalty at 35%
        elif drawdown > 0.25:
            reward -= (drawdown - 0.25) * 3.0  # Moderate ramp from 25-35%

        # --- No funding penalty (Hunter rides positions through funding) ---
        # --- No time penalty (Hunter waits for the right moment) ---

        return reward
