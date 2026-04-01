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

Reward shaping (V5 — Carrot + Delta Drawdown, ported from Shield V9/V10):
- 3x PnL multiplier on wins + 0.1 flat bonus (the carrot)
- 1.5x asymmetric loss penalty (unchanged — Builder is more loss-tolerant than Shield)
- Delta-based drawdown penalty (only penalizes increases, not state)
- Per-step reward floor at -5.0 (prevents -50K episode accumulation)
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

        self._prev_pnl = 0.0
        self._prev_drawdown = 0.0

    def reset(self, **kwargs):
        self._prev_pnl = 0.0
        self._prev_drawdown = 0.0
        return super().reset(**kwargs)

    def _calculate_reward(self, ctx: dict) -> float:
        """V5 reward: Carrot + delta drawdown (ported from Shield V9/V10).

        Builder gets a 3x win multiplier (vs Shield's 5x) because Builder
        already has higher base returns from 2x leverage and larger positions.
        """
        pnl = ctx["pnl_pct"]
        drawdown = ctx["drawdown"]
        has_position = ctx["has_position"]
        unrealized = ctx.get("unrealized_pnl_pct", 0.0)

        reward = 0.0

        # ═══════════════════════════════════════════════════════════════
        # THE CARROT — positive signal for profitable trades
        # ═══════════════════════════════════════════════════════════════
        if pnl > 0:
            reward += pnl * 3.0           # 3x multiplier (Shield uses 5x)
            reward += 0.1                 # +10 (after ×100) flat bonus per win
        elif pnl < 0:
            reward += pnl * 1.5           # 1.5x loss asymmetry (unchanged)

        # --- Dense unrealized PnL signal ---
        if has_position:
            reward += unrealized * 0.3

        # --- Trend-following bonus (Builder specialty) ---
        if pnl > 0 and self._prev_pnl > 0 and has_position:
            reward += pnl * 0.2           # 20% bonus for sustained gains
        self._prev_pnl = pnl

        # --- Drawdown penalty: DELTA-BASED (ported from Shield V6) ---
        dd_delta = max(0.0, drawdown - self._prev_drawdown)
        if dd_delta > 0:
            if drawdown > 0.15:
                reward -= dd_delta * 10.0  # Steep near 20% kill zone
            elif drawdown > 0.10:
                reward -= dd_delta * 3.0   # Moderate 10-15% zone
            else:
                reward -= dd_delta * 1.0   # Mild awareness below 10%
        self._prev_drawdown = drawdown

        # --- Funding bleed: only penalize above 0.03% threshold ---
        funding_cost = abs(ctx.get("funding_cost", 0))
        if funding_cost > FUNDING_PENALTY_THRESHOLD:
            reward -= (funding_cost - FUNDING_PENALTY_THRESHOLD) * 0.005

        # --- Anti-trivial-solution: time penalty when flat ---
        if not has_position:
            reward -= 0.0005

        # --- Scale ×100 + per-step floor ---
        raw = reward * 100.0
        return max(raw, -5.0)
