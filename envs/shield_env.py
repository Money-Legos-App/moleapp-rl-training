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

Reward shaping:
- 3x asymmetric loss penalty (losses hurt 3x more than gains help)
- Heavy drawdown penalty (kill at 10%)
- Bonus for staying flat during high-volatility periods
- HEAVY funding bleed penalty (Shield NEVER pays funding)
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

    def reset(self, **kwargs):
        self._last_close_step = 0
        self._had_position = False
        return super().reset(**kwargs)

    def _calculate_reward(self, ctx: dict) -> float:
        pnl = ctx["pnl_pct"]
        drawdown = ctx["drawdown"]
        has_position = ctx["has_position"]
        step = ctx["step"]
        unrealized = ctx.get("unrealized_pnl_pct", 0.0)

        reward = 0.0

        # --- Core PnL reward with 3x loss asymmetry ---
        if pnl < 0:
            reward += pnl * 3.0  # Losses hurt 3x
        else:
            reward += pnl

        # --- Dense unrealized PnL signal (breadcrumbs for the Critic) ---
        # Small reward/penalty every step while holding, proportional to unrealized P&L
        if has_position:
            reward += unrealized * 0.5  # Half-weight vs realized — don't overweight paper gains

        # --- Drawdown penalty (escalating — episode terminates at 10% via base env) ---
        if drawdown > 0.05:
            reward -= drawdown * 5.0  # Escalating penalty above 5%
        else:
            reward -= drawdown * 0.5  # Mild awareness below 5%

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
            reward -= 0.001  # Stronger penalty — force the agent to trade

        # --- Safe trade bonus ---
        if pnl > 0.01 and drawdown < 0.005:
            reward += 0.01

        # --- Scale ×100: prevent vanishing gradients from micro-PnL rewards ---
        return reward * 100.0
