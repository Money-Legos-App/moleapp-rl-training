"""
Shield Trading Environment (CONSERVATIVE Risk Profile)
========================================================
Goal: Capital preservation. Small, consistent wins. Avoid volatility.

Reward shaping:
- 3x asymmetric loss penalty (losses hurt 3x more than gains help)
- Heavy drawdown penalty (kill at 10%)
- Bonus for staying flat during high-volatility periods
- Funding bleed penalty
- Time penalty when flat (prevents "never trade" trivial solution)
- Bonus for profitable low-risk trades (incentivizes safe arbitrage)
"""

from __future__ import annotations

from envs.base_trading_env import BaseTradingEnv


class ShieldTradingEnv(BaseTradingEnv):
    """Conservative RL environment — The Shield."""

    def __init__(self, **kwargs):
        kwargs.setdefault("max_leverage", 1)
        kwargs.setdefault("max_positions", 2)
        kwargs.setdefault("max_sl_pct", 0.03)
        kwargs.setdefault("min_sl_pct", 0.005)
        kwargs.setdefault("max_tp_pct", 0.06)
        kwargs.setdefault("min_tp_pct", 0.01)
        kwargs.setdefault("profile_name", "shield")
        super().__init__(**kwargs)

    def _calculate_reward(self, ctx: dict) -> float:
        pnl = ctx["pnl_pct"]
        drawdown = ctx["drawdown"]
        has_position = ctx["has_position"]
        action = ctx["action"]

        reward = 0.0

        # --- Core PnL reward with 3x loss asymmetry ---
        if pnl < 0:
            reward += pnl * 3.0  # Losses hurt 3x
        else:
            reward += pnl

        # --- Drawdown penalty (heavy, kills at 10%) ---
        if drawdown > 0.10:
            reward -= 2.0  # Hard penalty — agent should learn to avoid this
        elif drawdown > 0.05:
            reward -= drawdown * 5.0  # Escalating penalty above 5%
        else:
            reward -= drawdown * 0.5  # Mild awareness below 5%

        # --- High-volatility avoidance ---
        # If ATR/price indicates high vol (feature[20] is atr_norm)
        is_high_vol = abs(ctx.get("current_price", 0)) > 0 and drawdown > 0.03
        if is_high_vol and not has_position:
            reward += 0.02  # Bonus for correctly staying flat

        # --- Funding bleed penalty ---
        reward -= abs(ctx.get("funding_cost", 0)) * 0.001

        # --- Anti-trivial-solution: time penalty when flat ---
        if not has_position:
            reward -= 0.00001  # Tiny penalty per step when not trading
            # This forces the agent to seek safe opportunities

        # --- Safe trade bonus ---
        # Reward profitable trades that had low drawdown
        if pnl > 0.01 and drawdown < 0.005:
            reward += 0.01  # Bonus for clean, safe wins

        return reward
