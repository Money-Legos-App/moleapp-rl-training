"""
Builder Trading Environment (MODERATE Risk Profile) — "High-Yield Engine"
===========================================================================
Goal: Targeted growth with controlled risk. Wealth generation for risk-tolerant users.

Risk Matrix:
- Max Leverage: 2x (maximized capital efficiency)
- Max Position Size: 50% of portfolio
- Max Stop Loss: 5%
- Min R/R Ratio: 1.8x
- Min LLM Confidence: 0.60
- Funding Block: > 0.03% (block only extreme funding spikes)

Reward shaping:
- 1.5x asymmetric loss penalty
- Drawdown penalty kicks in above 15%
- Trend-following bonus (reward for riding sustained moves)
- Funding penalty only above 0.03% threshold (tolerate normal funding)
"""

from __future__ import annotations

from envs.base_trading_env import BaseTradingEnv

# Funding threshold: only penalize above 0.03% per 8h (0.0003)
FUNDING_PENALTY_THRESHOLD = 0.0003


class BuilderTradingEnv(BaseTradingEnv):
    """Moderate RL environment — The Builder (High-Yield Engine)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("max_leverage", 2)
        kwargs.setdefault("max_positions", 4)
        kwargs.setdefault("max_sl_pct", 0.05)    # 5% max stop loss
        kwargs.setdefault("min_sl_pct", 0.01)
        kwargs.setdefault("max_tp_pct", 0.10)
        kwargs.setdefault("min_tp_pct", 0.018)   # Min R/R 1.8x → TP ≥ 1.8 * SL
        kwargs.setdefault("max_drawdown_pct", 0.20)  # 20% kill — matches risk_manager.py
        kwargs.setdefault("max_position_size_pct", 0.50)  # 50% of portfolio max
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

        # --- Funding bleed: only penalize above 0.03% threshold ---
        # Normal funding is tolerated; only extreme spikes are penalized
        funding_cost = abs(ctx.get("funding_cost", 0))
        if funding_cost > FUNDING_PENALTY_THRESHOLD:
            reward -= (funding_cost - FUNDING_PENALTY_THRESHOLD) * 0.005

        # --- Small time penalty to encourage activity ---
        if not has_position:
            reward -= 0.000005

        return reward

    def reset(self, **kwargs):
        self._prev_pnl = 0.0
        return super().reset(**kwargs)
