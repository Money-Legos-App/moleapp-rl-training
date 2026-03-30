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

Reward shaping (V9 — The Sniper's Carrot):
- V6 foundation: delta-based drawdown, per-step floor at -5.0
- NEW: +10 flat bonus for profitable trade close (the carrot)
- NEW: PnL multiplier on wins (5x) to make winning feel MUCH better than losing feels bad
- 2x asymmetric loss penalty (unchanged from V6)
- Funding bleed penalty (Shield NEVER pays funding)
- Time penalty when flat (prevents "never trade" trivial solution)
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
        """V9 reward: The Sniper's Carrot.

        V8 postmortem: V6 reward was all stick — agent learned trading is a
        minefield, turtled up, took zero trades, win_rate hit 15%, eval NaN'd.
        V9 fix: massive positive spike for profitable trades. The agent must
        feel that winning is 5x better than losing is bad.
        """
        pnl = ctx["pnl_pct"]
        drawdown = ctx["drawdown"]
        has_position = ctx["has_position"]
        step = ctx["step"]
        unrealized = ctx.get("unrealized_pnl_pct", 0.0)

        reward = 0.0

        # ═══════════════════════════════════════════════════════════════
        # THE CARROT — massive positive signal for profitable trades
        # ═══════════════════════════════════════════════════════════════
        if pnl > 0:
            reward += pnl * 5.0           # 5x multiplier on winning PnL
            reward += 0.1                 # +10 (after ×100 scale) flat bonus per win
        elif pnl < 0:
            reward += pnl * 2.0           # 2x loss asymmetry (unchanged from V6)

        # --- Dense unrealized PnL signal (breadcrumbs for the Critic) ---
        if has_position:
            reward += unrealized * 0.3

        # --- Drawdown penalty: DELTA-BASED (V6, unchanged) ---
        dd_delta = max(0.0, drawdown - self._prev_drawdown)
        if dd_delta > 0:
            if drawdown > 0.05:
                reward -= dd_delta * 10.0
            else:
                reward -= dd_delta * 2.0
        self._prev_drawdown = drawdown

        # --- Track position close events (for volatility bonus) ---
        if self._had_position and not has_position:
            self._last_close_step = step
        self._had_position = has_position

        # --- Volatility avoidance (decay-based) ---
        if not has_position and self._last_close_step > 0:
            steps_since_close = step - self._last_close_step
            if 0 < steps_since_close <= 48 and drawdown > 0.03:
                decay = max(0.0, 1.0 - steps_since_close / 48.0)
                reward += 0.02 * decay

        # --- Funding bleed penalty: ALWAYS block ---
        reward -= abs(ctx.get("funding_cost", 0)) * 0.01

        # --- Anti-trivial-solution: time penalty when flat ---
        if not has_position:
            reward -= 0.0005

        # --- Safe trade bonus ---
        if pnl > 0.01 and drawdown < 0.005:
            reward += 0.01

        # --- Scale ×100 + per-step floor ---
        raw = reward * 100.0
        return max(raw, -5.0)
