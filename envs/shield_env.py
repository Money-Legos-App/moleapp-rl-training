"""
Shield Trading Environment (CONSERVATIVE Risk Profile) — "Flexible USD Vault"
================================================================================
Goal: Capital preservation. Inflation protection with zero liquidation risk.

Risk Matrix:
- Max Leverage: 1x (spot equivalent, no liquidation)
- Max Position Size: 25% of portfolio
- Max Stop Loss: 3%
- Min R/R Ratio: 2.5x
- Min LLM Confidence: 0.70
- Funding Block: Always (never pay funding)

Reward shaping (V6 — delta-based drawdown fix):
- 2x asymmetric loss penalty (was 3x — too punitive, caused reward/performance disconnect)
- Delta-based drawdown penalty (only penalizes drawdown *increases*, not state)
- Per-step reward floor at -5.0 (prevents -50K episode accumulation)
- Bonus for staying flat during high-volatility periods
- Funding bleed penalty (Shield NEVER pays funding)
- Time penalty when flat (prevents "never trade" trivial solution)
- Bonus for profitable low-risk trades (incentivizes safe arbitrage)
"""

from __future__ import annotations

from envs.base_trading_env import BaseTradingEnv


class ShieldTradingEnv(BaseTradingEnv):
    """Conservative RL environment — The Shield (Flexible USD Vault)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("max_leverage", 1)
        kwargs.setdefault("max_positions", 2)
        kwargs.setdefault("max_sl_pct", 0.03)    # 3% max stop loss
        kwargs.setdefault("min_sl_pct", 0.005)
        kwargs.setdefault("max_tp_pct", 0.075)   # Min R/R 2.5x → TP ≥ 2.5 * SL
        kwargs.setdefault("min_tp_pct", 0.0125)  # 2.5x * 0.5% min SL
        kwargs.setdefault("max_drawdown_pct", 0.10)  # 10% kill — matches risk_manager.py
        kwargs.setdefault("max_position_size_pct", 0.25)  # 25% of portfolio max
        kwargs.setdefault("profile_name", "shield")
        super().__init__(**kwargs)
        self._last_close_step = 0
        self._had_position = False
        self._prev_drawdown = 0.0

    def reset(self, **kwargs):
        self._last_close_step = 0
        self._had_position = False
        self._prev_drawdown = 0.0
        return super().reset(**kwargs)

    def _calculate_reward(self, ctx: dict) -> float:
        """V6 reward: delta-based drawdown + per-step floor to prevent -50K accumulation.

        V5 postmortem: drawdown penalty applied every step as a state penalty,
        accumulating -35/step at 7% drawdown × 2880 steps = -100K. The agent
        learned profitable trades (27% return) but the reward said -50K.
        Fix: penalize drawdown *increases* only, not the state itself.
        """
        pnl = ctx["pnl_pct"]
        drawdown = ctx["drawdown"]
        has_position = ctx["has_position"]
        step = ctx["step"]
        unrealized = ctx.get("unrealized_pnl_pct", 0.0)

        reward = 0.0

        # --- Core PnL reward with 2x loss asymmetry (was 3x — too punitive) ---
        if pnl < 0:
            reward += pnl * 2.0
        else:
            reward += pnl

        # --- Dense unrealized PnL signal (breadcrumbs for the Critic) ---
        if has_position:
            reward += unrealized * 0.3  # Reduced from 0.5 — less noise from paper P&L

        # --- Drawdown penalty: DELTA-BASED (only when drawdown increases) ---
        # V5 bug: state-based penalty accumulated -35/step for 2880 steps = -100K
        # Fix: only penalize the *change* in drawdown, not the level
        dd_delta = max(0.0, drawdown - self._prev_drawdown)  # Only penalize increases
        if dd_delta > 0:
            if drawdown > 0.05:
                reward -= dd_delta * 10.0  # Strong signal when approaching 10% kill
            else:
                reward -= dd_delta * 2.0   # Mild awareness below 5%
        self._prev_drawdown = drawdown

        # --- Track position close events (for volatility bonus) ---
        if self._had_position and not has_position:
            self._last_close_step = step
        self._had_position = has_position

        # --- Volatility avoidance (decay-based, not unconditional) ---
        if not has_position and self._last_close_step > 0:
            steps_since_close = step - self._last_close_step
            if 0 < steps_since_close <= 48 and drawdown > 0.03:
                decay = max(0.0, 1.0 - steps_since_close / 48.0)
                reward += 0.02 * decay

        # --- Funding bleed penalty: ALWAYS block (Shield NEVER pays funding) ---
        reward -= abs(ctx.get("funding_cost", 0)) * 0.01

        # --- Anti-trivial-solution: time penalty when flat ---
        if not has_position:
            reward -= 0.0005  # Halved from 0.001 — 0.001 × 2880 steps = -2.88 per ep

        # --- Safe trade bonus ---
        if pnl > 0.01 and drawdown < 0.005:
            reward += 0.01

        # --- Scale ×100 + per-step floor (prevents catastrophic accumulation) ---
        raw = reward * 100.0
        return max(raw, -5.0)  # Floor at -5 per step → max episode penalty ~-14K vs old -50K
