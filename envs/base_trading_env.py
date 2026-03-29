"""
Base Trading Environment for Hyperliquid Perpetuals
=====================================================
Gymnasium environment simulating HL perp trading with realistic constraints:
- 0.035% taker fee per trade (HL mainnet rate)
- Funding rate charged every 8 hours
- Minimum $10 notional per order
- Leverage caps per risk profile
- Liquidation simulation

Subclassed by ShieldTradingEnv and BuilderTradingEnv
which only override `_calculate_reward()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data.preprocessors.feature_engineer import OBS_DIM, MarketFeatures, build_observation

logger = logging.getLogger(__name__)

# HL mainnet constants
TAKER_FEE_PCT = 0.00035  # 0.035%
MAKER_FEE_PCT = 0.0001   # 0.01% (not used — we assume taker for worst case)
FUNDING_INTERVAL_HOURS = 8
MIN_NOTIONAL_USD = 10.0
LIQUIDATION_MAINTENANCE_MARGIN = 0.03  # 3% maintenance margin

# Market impact model: slippage_pct = IMPACT_COEFF * sqrt(position_usd / bar_volume_usd)
# Square-root impact is standard in TCA literature (Kyle, 1985).
# 0.1 coefficient means: trading 1% of bar volume → ~0.1% * sqrt(0.01) = 0.01% slippage.
# Trading 100% of bar volume → 0.1% * sqrt(1.0) = 0.1% slippage.
# Conservative but realistic for HL perpetuals.
IMPACT_COEFF = 0.001  # 0.1% at full-bar volume


@dataclass
class Position:
    """Tracks an open position."""
    direction: int  # 1=long, -1=short
    entry_price: float
    size_usd: float
    leverage: float
    stop_loss_pct: float
    take_profit_pct: float
    entry_step: int
    funding_paid: float = 0.0
    peak_pnl_pct: float = 0.0


@dataclass
class EpisodeState:
    """Mutable state for a single episode."""
    account_value: float = 0.0
    initial_capital: float = 0.0
    peak_account_value: float = 0.0
    position: Optional[Position] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    step: int = 0
    last_trade_step: int = 0
    total_fees_paid: float = 0.0
    total_funding_paid: float = 0.0


class BaseTradingEnv(gym.Env):
    """
    Base Gymnasium environment for HL perp trading.

    Observation: 47-dim float vector (from shared feature_engineer.py)
    Action: Box(5) continuous [-1, 1]
      [0] direction+size: [-1,-0.1]=SHORT, [-0.1,0.1]=HOLD, [0.1,1]=LONG
      [1] leverage: mapped to [1, max_leverage]
      [2] stop_loss_pct: mapped to [min_sl, max_sl]
      [3] take_profit_pct: mapped to [min_tp, max_tp]
      [4] confidence: scales position size ([-1,1] → [5%, 100%] of intended size)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_data: np.ndarray | None = None,  # shape: (timesteps, n_features) — raw OHLCV+indicators
        feature_data: list[MarketFeatures] | None = None,  # pre-built MarketFeatures per timestep
        market_data_path: str | None = None,  # Alternative: load from file (avoids Ray serialization)
        feature_data_path: str | None = None,
        max_leverage: int = 3,
        max_positions: int = 5,
        initial_capital: float = 1000.0,
        max_sl_pct: float = 0.07,
        min_sl_pct: float = 0.01,
        max_tp_pct: float = 0.15,
        min_tp_pct: float = 0.02,
        episode_length: int = 2880,  # 30 days of 15-min steps
        max_drawdown_pct: float = 1.0,  # Drawdown kill: 0.10=10%. 1.0=disabled (base default)
        max_position_size_pct: float = 1.0,  # Max single position as % of portfolio (Shield=0.25, Builder=0.50)
        profile_name: str = "base",
    ):
        super().__init__()

        # Load data from files if paths provided (avoids 200K+ object serialization through Ray)
        if market_data is None and market_data_path:
            market_data = np.load(market_data_path)
        if feature_data is None and feature_data_path:
            import pickle as _pkl
            with open(feature_data_path, "rb") as _f:
                feature_data = _pkl.load(_f)

        assert market_data is not None, "Must provide market_data or market_data_path"
        assert feature_data is not None, "Must provide feature_data or feature_data_path"

        self.market_data = market_data
        self.feature_data = feature_data
        self.max_leverage = max_leverage
        self.max_positions = max_positions
        self.initial_capital = initial_capital
        self.max_sl_pct = max_sl_pct
        self.min_sl_pct = min_sl_pct
        self.max_tp_pct = max_tp_pct
        self.min_tp_pct = min_tp_pct
        self.episode_length = min(episode_length, len(market_data))
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size_pct = max_position_size_pct
        self.profile_name = profile_name

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Episode state
        self.state = EpisodeState()
        self._episode_start_idx = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Random start point within data (with enough room for full episode)
        max_start = len(self.market_data) - self.episode_length
        if max_start > 0:
            self._episode_start_idx = self.np_random.integers(0, max_start)
        else:
            self._episode_start_idx = 0

        self.state = EpisodeState(
            account_value=self.initial_capital,
            initial_capital=self.initial_capital,
            peak_account_value=self.initial_capital,
        )

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.state.step += 1
        current_idx = self._episode_start_idx + self.state.step
        prev_value = self.state.account_value

        # 1. Process funding for open position
        self._process_funding(current_idx)

        # 2. Check SL/TP/liquidation for open position
        sl_tp_pnl = self._check_sl_tp_liquidation(current_idx)

        # 3. Parse action and potentially open new position
        trade_pnl = 0.0
        if self.state.position is None:
            trade_pnl = self._process_action(action, current_idx)

        # 4. Update mark-to-market
        self._update_mtm(current_idx)

        # 5. Calculate reward (overridden by subclasses)
        total_step_pnl = sl_tp_pnl + trade_pnl
        pnl_pct = total_step_pnl / max(prev_value, 1.0)
        drawdown = (self.state.peak_account_value - self.state.account_value) / max(
            self.state.peak_account_value, 1.0
        )
        drawdown = max(0.0, drawdown)

        # Dense reward: unrealized PnL breadcrumb every step while holding
        # Gives the Critic network a signal to follow, not just at trade close
        unrealized_pnl_pct = self._unrealized_pnl(current_idx) / max(prev_value, 1.0)

        reward_context = {
            "pnl_pct": pnl_pct,
            "drawdown": drawdown,
            "action": action,
            "has_position": self.state.position is not None,
            "step": self.state.step,
            "funding_cost": self.state.total_funding_paid,
            "current_price": self._get_price(current_idx),
            "unrealized_pnl_pct": unrealized_pnl_pct,
        }
        reward = self._calculate_reward(reward_context)

        # NaN guard — chaotic actions can produce NaN via gradient explosion
        if np.isnan(reward) or np.isinf(reward):
            reward = -5.0  # Safe fallback penalty

        # 6. Check termination
        terminated = False
        truncated = False

        # Account blown (< $1)
        if self.state.account_value < 1.0:
            terminated = True
            reward -= 10.0  # Universal heavy penalty for blowup

        # Drawdown kill — matches production risk_manager.py kill switch
        # Shield=10%, Builder=20%. Episode ENDS, not just penalty.
        if drawdown >= self.max_drawdown_pct:
            terminated = True
            reward -= 5.0  # Heavy but less than blowup (agent should learn to avoid)

        # Episode time limit
        if self.state.step >= self.episode_length - 1:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, ctx: dict) -> float:
        """Override in subclasses for profile-specific reward shaping."""
        return ctx["pnl_pct"]

    # ──────────────────────────────────────────────────────────────
    # Internal methods
    # ──────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current market state."""
        idx = self._episode_start_idx + self.state.step
        idx = min(idx, len(self.feature_data) - 1)

        features = self.feature_data[idx]
        # Override portfolio state fields with actual episode state
        features.account_value = self.state.account_value
        features.initial_capital = self.state.initial_capital
        features.peak_account_value = self.state.peak_account_value
        features.open_position_count = 1 if self.state.position else 0
        features.max_positions = self.max_positions
        features.margin_utilization = self._margin_utilization()
        features.unrealized_pnl = self._unrealized_pnl(idx)
        features.days_since_last_trade = (
            (self.state.step - self.state.last_trade_step) * 15 / 1440
        )  # 15-min steps
        features.has_open_position_this_asset = self.state.position is not None
        features.existing_direction = (
            self.state.position.direction if self.state.position else 0
        )

        # Distance to liquidation (v1.1.0)
        if self.state.position is not None:
            pos = self.state.position
            price = self._get_price(idx)
            entry = max(pos.entry_price, 1e-8)  # NaN guard
            if pos.direction == 1:  # long
                pnl_pct = (price - entry) / entry * pos.leverage
            else:  # short
                pnl_pct = (entry - price) / entry * pos.leverage
            liq_threshold = -1.0 / pos.leverage + LIQUIDATION_MAINTENANCE_MARGIN
            features.distance_to_liquidation = max(0.0, min(1.0,
                (pnl_pct - liq_threshold) / max(abs(liq_threshold), 0.01)
            ))
        else:
            features.distance_to_liquidation = 1.0

        return build_observation(features)

    def _get_price(self, idx: int) -> float:
        """Get close price at index. Assumes column 3 = close in OHLCV."""
        idx = min(idx, len(self.market_data) - 1)
        return float(self.market_data[idx, 3])  # close price

    def _get_funding_rate(self, idx: int) -> float:
        """Get funding rate at index. Assumes column 5 = funding in market_data."""
        idx = min(idx, len(self.market_data) - 1)
        if self.market_data.shape[1] > 5:
            return float(self.market_data[idx, 5])
        return 0.0

    def _get_volume(self, idx: int) -> float:
        """Get bar volume at index. Assumes column 4 = volume in market_data."""
        idx = min(idx, len(self.market_data) - 1)
        return max(float(self.market_data[idx, 4]), 1.0)  # Floor at 1 to avoid div-by-zero

    def _estimate_slippage(self, position_usd: float, price: float, idx: int) -> float:
        """
        Estimate market impact slippage using square-root model.

        Returns slippage as a fraction (e.g., 0.001 = 0.1%).
        The entry price is adjusted adversely by this amount.

        Model: slippage = IMPACT_COEFF * sqrt(position_usd / bar_volume_usd)
        - Small orders (<<bar volume): negligible slippage
        - Large orders (~bar volume): ~0.1% slippage
        - Huge orders (>bar volume): >0.1% slippage (agent learns to avoid)
        """
        bar_volume_usd = self._get_volume(idx) * price
        if bar_volume_usd <= 0:
            return 0.0
        participation_rate = position_usd / bar_volume_usd
        return IMPACT_COEFF * np.sqrt(participation_rate)

    def _process_action(self, action: np.ndarray, idx: int) -> float:
        """Parse Box(5) action and open a position if signaled."""
        direction_size = float(action[0])

        # Dead zone: HOLD
        if -0.1 <= direction_size <= 0.1:
            return 0.0

        # Determine direction and size
        direction = 1 if direction_size > 0 else -1
        size_frac = abs(direction_size)  # [0.1, 1.0] → use as fraction of account

        # Map action[1] to leverage [1, max_leverage]
        leverage = 1.0 + (float(action[1]) + 1.0) / 2.0 * (self.max_leverage - 1.0)
        leverage = np.clip(leverage, 1.0, self.max_leverage)

        # Map action[2] to SL
        sl_pct = self.min_sl_pct + (float(action[2]) + 1.0) / 2.0 * (
            self.max_sl_pct - self.min_sl_pct
        )

        # Map action[3] to TP
        tp_pct = self.min_tp_pct + (float(action[3]) + 1.0) / 2.0 * (
            self.max_tp_pct - self.min_tp_pct
        )

        # Map action[4] to confidence scaling [0.05, 1.0]
        confidence = (float(action[4]) + 1.0) / 2.0  # [-1,1] → [0,1]
        confidence = np.clip(confidence, 0.05, 1.0)   # Floor 5% — prevents zero-position exploit

        # Calculate position size in USD (scaled by confidence, capped by max_position_size_pct)
        available_margin = self.state.account_value * (1.0 - self._margin_utilization())
        max_position = self.state.account_value * self.max_position_size_pct
        position_usd = min(available_margin * size_frac * leverage * confidence, max_position * leverage)

        # Minimum notional check
        if position_usd < MIN_NOTIONAL_USD:
            return 0.0

        # Entry fee
        fee = position_usd * TAKER_FEE_PCT
        self.state.account_value -= fee
        self.state.total_fees_paid += fee

        # Market impact slippage: worse fill price for larger orders
        price = self._get_price(idx)
        slippage_pct = self._estimate_slippage(position_usd, price, idx)
        # Latency slippage: LLM inference + AA bundler/relayer jitter (1-5 bps)
        latency_slippage = self.np_random.uniform(0.0001, 0.0005)
        slippage_pct += latency_slippage
        # Adverse fill: longs pay more, shorts receive less
        fill_price = price * (1.0 + direction * slippage_pct)
        slippage_cost = position_usd * slippage_pct
        self.state.account_value -= slippage_cost
        self.state.total_fees_paid += slippage_cost  # Track with fees

        self.state.position = Position(
            direction=direction,
            entry_price=fill_price,
            size_usd=position_usd,
            leverage=leverage,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            entry_step=self.state.step,
        )
        self.state.total_trades += 1
        self.state.last_trade_step = self.state.step

        return -(fee + slippage_cost) / max(self.state.account_value, 1.0)

    def _check_sl_tp_liquidation(self, idx: int) -> float:
        """Check stop loss, take profit, and liquidation for open position."""
        if self.state.position is None:
            return 0.0

        pos = self.state.position
        price = self._get_price(idx)
        entry = max(pos.entry_price, 1e-8)  # NaN guard

        # PnL calculation
        if pos.direction == 1:  # long
            pnl_pct = (price - entry) / entry
        else:  # short
            pnl_pct = (entry - price) / entry

        pnl_pct *= pos.leverage  # leveraged PnL

        # Track peak PnL for trailing
        pos.peak_pnl_pct = max(pos.peak_pnl_pct, pnl_pct)

        # Liquidation check (simplified)
        liq_threshold = -1.0 / pos.leverage + LIQUIDATION_MAINTENANCE_MARGIN
        if pnl_pct <= liq_threshold:
            return self._close_position(pnl_pct=-1.0 / pos.leverage, reason="liquidation")

        # Stop loss
        if pnl_pct <= -pos.stop_loss_pct:
            return self._close_position(pnl_pct=-pos.stop_loss_pct, reason="stop_loss")

        # Take profit
        if pnl_pct >= pos.take_profit_pct:
            return self._close_position(pnl_pct=pos.take_profit_pct, reason="take_profit")

        return 0.0

    def _close_position(self, pnl_pct: float, reason: str) -> float:
        """Close position, apply fees + exit slippage, update account."""
        if self.state.position is None:
            return 0.0

        pos = self.state.position
        realized_pnl = pos.size_usd * pnl_pct / pos.leverage

        # Exit fee
        fee = pos.size_usd * TAKER_FEE_PCT

        # Exit slippage (closing = reverse direction, same impact model + latency)
        current_idx = self._episode_start_idx + self.state.step
        price = self._get_price(current_idx)
        exit_slippage = self._estimate_slippage(pos.size_usd, price, current_idx)
        exit_slippage += self.np_random.uniform(0.0001, 0.0005)  # LLM+AA latency jitter
        slippage_cost = pos.size_usd * exit_slippage

        total_cost = fee + slippage_cost
        self.state.account_value += realized_pnl - total_cost
        self.state.total_fees_paid += total_cost
        self.state.total_pnl += realized_pnl - total_cost

        if realized_pnl > 0:
            self.state.winning_trades += 1

        # Update peak
        self.state.peak_account_value = max(
            self.state.peak_account_value, self.state.account_value
        )

        self.state.position = None
        return (realized_pnl - total_cost) / max(self.state.account_value, 1.0)

    def _process_funding(self, idx: int) -> None:
        """Apply funding rate every 8 hours (32 steps at 15-min interval)."""
        if self.state.position is None:
            return

        # Funding applied every 32 steps (8h / 15min)
        steps_in_position = self.state.step - self.state.position.entry_step
        if steps_in_position > 0 and steps_in_position % 32 == 0:
            funding_rate = self._get_funding_rate(idx)
            # Longs pay positive funding, shorts pay negative funding
            funding_cost = (
                self.state.position.size_usd
                * funding_rate
                * self.state.position.direction
            )
            self.state.account_value -= funding_cost
            self.state.total_funding_paid += funding_cost
            self.state.position.funding_paid += funding_cost

    def _update_mtm(self, idx: int) -> None:
        """Update mark-to-market account value."""
        if self.state.position is not None:
            unrealized = self._unrealized_pnl(idx)
            # Account value = initial + realized + unrealized
            # (unrealized is already reflected in position)
        self.state.peak_account_value = max(
            self.state.peak_account_value, self.state.account_value
        )

    def _unrealized_pnl(self, idx: int) -> float:
        """Calculate unrealized PnL for open position."""
        if self.state.position is None:
            return 0.0

        pos = self.state.position
        price = self._get_price(idx)
        entry = max(pos.entry_price, 1e-8)  # NaN guard

        if pos.direction == 1:
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry

        return pos.size_usd * pnl_pct  # per unit of margin

    def _margin_utilization(self) -> float:
        """Current margin utilization as fraction."""
        if self.state.position is None:
            return 0.0
        margin_used = self.state.position.size_usd / self.state.position.leverage
        return min(margin_used / max(self.state.account_value, 1.0), 1.0)

    def _get_info(self) -> dict[str, Any]:
        """Return info dict for logging/evaluation."""
        return {
            "account_value": self.state.account_value,
            "total_pnl": self.state.total_pnl,
            "total_trades": self.state.total_trades,
            "winning_trades": self.state.winning_trades,
            "win_rate": (
                self.state.winning_trades / max(self.state.total_trades, 1)
            ),
            "total_fees": self.state.total_fees_paid,
            "total_funding": self.state.total_funding_paid,
            "max_drawdown": (
                (self.state.peak_account_value - self.state.account_value)
                / max(self.state.peak_account_value, 1.0)
            ),
            "has_position": self.state.position is not None,
            "step": self.state.step,
            "profile": self.profile_name,
        }
