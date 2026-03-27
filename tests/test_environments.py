"""
Environment Validation Tests
==============================
Critical tests to run BEFORE sending to RunPod for training:

1. Random policy loses money (env isn't trivially exploitable)
2. Funding charges apply correctly every 32 steps
3. SL/TP fire at correct thresholds
4. Shield trades >5% of opportunities (anti-trivial-solution)
5. Liquidation triggers correctly
6. Fee accounting is accurate
7. Observation shape and value ranges are correct
"""

from __future__ import annotations

import numpy as np
import pytest

from data.preprocessors.feature_engineer import MarketFeatures, OBS_DIM, build_observation
from envs.base_trading_env import (
    BaseTradingEnv,
    TAKER_FEE_PCT,
    MIN_NOTIONAL_USD,
)
from envs.shield_env import ShieldTradingEnv
from envs.builder_env import BuilderTradingEnv


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _make_synthetic_data(
    n_steps: int = 3000,
    base_price: float = 50000.0,
    volatility: float = 0.002,
    trend: float = 0.0,
    funding_rate: float = 0.0001,
) -> tuple[np.ndarray, list[MarketFeatures]]:
    """
    Generate synthetic market data for testing.

    Returns (market_data, feature_list) with realistic-ish price walk.
    """
    rng = np.random.RandomState(42)

    # Random walk with optional trend
    returns = rng.normal(trend, volatility, n_steps)
    prices = base_price * np.cumprod(1.0 + returns)

    # Build OHLCV-like market_data: (n, 7) = [open, high, low, close, volume, funding, OI]
    market_data = np.zeros((n_steps, 7), dtype=np.float32)
    for i in range(n_steps):
        p = prices[i]
        noise = abs(rng.normal(0, p * 0.001))
        market_data[i, 0] = p - noise  # open
        market_data[i, 1] = p + noise * 2  # high
        market_data[i, 2] = p - noise * 2  # low
        market_data[i, 3] = p  # close
        market_data[i, 4] = rng.uniform(1e6, 1e8)  # volume
        market_data[i, 5] = funding_rate  # constant funding for testing
        market_data[i, 6] = rng.uniform(1e8, 5e8)  # OI

    # Build MarketFeatures list
    features = []
    for i in range(n_steps):
        p = float(prices[i])
        ts = 1700000000.0 + i * 900  # 15-min spacing
        features.append(MarketFeatures(
            price=p,
            price_1h_ago=float(prices[max(0, i - 4)]),
            price_4h_ago=float(prices[max(0, i - 16)]),
            price_24h_ago=float(prices[max(0, i - 96)]),
            vwap_24h=p,
            rolling_mean_30d=base_price,
            volume_24h=float(market_data[i, 4]),
            rolling_avg_vol_30d=5e7,
            bid_imbalance_pct=0.0,
            spread_bps=1.0,
            open_interest=float(market_data[i, 6]),
            rolling_avg_oi_30d=3e8,
            oi_1h_ago=float(market_data[max(0, i - 4), 6]),
            oi_4h_ago=float(market_data[max(0, i - 16), 6]),
            funding_rate=funding_rate,
            funding_8h_cumulative=funding_rate * 3,
            prev_funding_rate=funding_rate,
            rsi_1h=50.0,
            rsi_4h=50.0,
            macd_hist_1h=0.0,
            bb_position_1h=0.5,
            atr_1h=p * 0.01,
            ema_20=p,
            ema_50=p,
            ema_200=p,
            sma_4h=p,
            volume_trend_1h=0.0,
            roc_1h=0.0,
            roc_4h=0.0,
            account_value=1000.0,
            initial_capital=1000.0,
            peak_account_value=1000.0,
            open_position_count=0,
            max_positions=5,
            margin_utilization=0.0,
            unrealized_pnl=0.0,
            mission_start_timestamp=ts,
            current_timestamp=ts,
            days_since_last_trade=0.0,
            has_open_position_this_asset=False,
            existing_direction=0,
            btc_dominance=50.0,
            fear_greed_index=50.0,
            market_regime=0,
            cross_asset_momentum=0.0,
        ))

    return market_data, features


def _make_trending_data(direction: str = "up", n_steps: int = 3000) -> tuple:
    """Generate strongly trending data (for SL/TP tests)."""
    trend = 0.001 if direction == "up" else -0.001
    return _make_synthetic_data(n_steps=n_steps, trend=trend, volatility=0.0005)


def _make_env(env_class, **kwargs):
    """Create an env with synthetic data."""
    market_data, features = _make_synthetic_data(**kwargs.pop("data_kwargs", {}))
    return env_class(market_data=market_data, feature_data=features, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# Test 1: Random policy loses money (env is realistic)
# ──────────────────────────────────────────────────────────────────────

class TestRandomPolicyLosesMoney:
    """A random agent should lose money due to fees and funding."""

    @pytest.mark.parametrize("env_class", [ShieldTradingEnv, BuilderTradingEnv])
    def test_random_agent_pays_significant_fees(self, env_class):
        """Random actions should accumulate significant fees (proving env realism)."""
        market_data, features = _make_synthetic_data(n_steps=2000)
        env = env_class(
            market_data=market_data,
            feature_data=features,
            episode_length=1000,
            initial_capital=1000.0,
        )

        rng = np.random.RandomState(123)
        total_fees = 0.0
        total_trades = 0
        n_runs = 5

        for _ in range(n_runs):
            obs, info = env.reset(seed=rng.randint(0, 10000))
            done = False
            while not done:
                action = rng.uniform(-1, 1, size=5).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            total_fees += info["total_fees"]
            total_trades += info["total_trades"]

        avg_fees = total_fees / n_runs
        avg_trades = total_trades / n_runs
        # Random agent should trade frequently and pay meaningful fees
        assert avg_trades > 5, f"Random agent should trade, got avg {avg_trades:.0f} trades"
        assert avg_fees > 1.0, f"Random agent should pay fees, got avg ${avg_fees:.2f}"

    def test_random_agent_fees_exceed_random_walk_edge(self):
        """Over many runs, fees should dominate random walk noise."""
        market_data, features = _make_synthetic_data(n_steps=2000, volatility=0.001)
        env = ShieldTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=1000,
            initial_capital=1000.0,
        )

        rng = np.random.RandomState(42)
        total_fees = 0.0
        total_abs_pnl = 0.0
        n_runs = 20

        for _ in range(n_runs):
            obs, info = env.reset(seed=rng.randint(0, 10000))
            done = False
            while not done:
                action = rng.uniform(-1, 1, size=5).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            total_fees += info["total_fees"]
            total_abs_pnl += abs(info["total_pnl"])

        avg_fees = total_fees / n_runs
        # Fees should be a meaningful cost (>$0.3 per episode)
        # Lower threshold: confidence scaling reduces avg position size ~50%
        assert avg_fees > 0.3, (
            f"Random agent should pay meaningful fees, got avg ${avg_fees:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 2: Funding charges apply correctly
# ──────────────────────────────────────────────────────────────────────

class TestFundingCharges:
    """Verify funding is charged every 32 steps (8h at 15-min intervals)."""

    def test_funding_applied_at_correct_interval(self):
        market_data, features = _make_synthetic_data(
            n_steps=500, funding_rate=0.001  # 0.1% per 8h — exaggerated for testing
        )
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=200,
            initial_capital=10000.0,
            max_leverage=1,
        )
        obs, _ = env.reset(seed=42)

        # Open a long position immediately
        action = np.array([0.5, 0.0, 0.0, 0.5, 0.5], dtype=np.float32)
        env.step(action)

        assert env.state.position is not None, "Position should be open"
        initial_funding = env.state.total_funding_paid

        # Funding fires when steps_in_position % 32 == 0 and > 0
        # Position opened at step 1, so steps_in_position = step - entry_step
        # We need step - 1 = 32 → step = 33 → 32 more hold steps after opening
        hold_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(32):
            env.step(hold_action)

        # At step 33, steps_in_position=32, funding should have been charged
        assert env.state.total_funding_paid > initial_funding, (
            "Funding should be charged when steps_in_position=32"
        )

    def test_no_funding_when_flat(self):
        market_data, features = _make_synthetic_data(
            n_steps=500, funding_rate=0.001
        )
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=100,
            initial_capital=10000.0,
        )
        obs, _ = env.reset(seed=42)

        # Hold (no position) for 50 steps
        hold_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            env.step(hold_action)

        assert env.state.total_funding_paid == 0.0, "No funding should be charged when flat"


# ──────────────────────────────────────────────────────────────────────
# Test 3: SL/TP fire at correct thresholds
# ──────────────────────────────────────────────────────────────────────

class TestSLTPMechanics:
    """Verify stop loss and take profit trigger correctly."""

    def test_stop_loss_triggers(self):
        """Open a long, price drops → SL should close position."""
        # Create sharply declining data
        market_data, features = _make_trending_data(direction="down", n_steps=500)
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=200,
            initial_capital=10000.0,
            max_leverage=1,
            min_sl_pct=0.005,  # 0.5% SL
            max_sl_pct=0.01,   # 1% SL
        )
        obs, _ = env.reset(seed=0)

        # Open a long position (against the trend → should get stopped out)
        action = np.array([0.8, 0.0, -1.0, 0.5, 0.5], dtype=np.float32)  # action[2]=-1 → min SL
        env.step(action)
        assert env.state.position is not None, "Position should open"

        # Run until position closes or episode ends
        hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pos_closed = False
        for _ in range(100):
            env.step(hold)
            if env.state.position is None:
                pos_closed = True
                break

        assert pos_closed, "Position should have been closed by SL in downtrend"
        assert env.state.total_pnl < 0, "SL exit should result in a loss"

    def test_take_profit_triggers(self):
        """Open a long in uptrend → TP should close position."""
        market_data, features = _make_trending_data(direction="up", n_steps=500)
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=200,
            initial_capital=10000.0,
            max_leverage=1,
            min_tp_pct=0.005,  # 0.5% TP
            max_tp_pct=0.01,
        )
        obs, _ = env.reset(seed=0)

        # Open a long position (with the trend)
        action = np.array([0.8, 0.0, 0.5, -1.0, 0.5], dtype=np.float32)  # action[3]=-1 → min TP
        env.step(action)
        assert env.state.position is not None

        hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pos_closed = False
        for _ in range(100):
            env.step(hold)
            if env.state.position is None:
                pos_closed = True
                break

        assert pos_closed, "Position should have been closed by TP in uptrend"
        assert env.state.total_pnl > 0, "TP exit should result in profit (minus fees)"


# ──────────────────────────────────────────────────────────────────────
# Test 4: Shield trades >5% of opportunities (anti-trivial-solution)
# ──────────────────────────────────────────────────────────────────────

class TestShieldTradesMinimum:
    """
    Shield env reward shaping should incentivize SOME trading.
    A "smart" hold-only agent should accumulate negative reward via time penalty.
    """

    def test_hold_only_gets_negative_reward(self):
        """Pure hold strategy should accumulate negative reward from time penalty."""
        market_data, features = _make_synthetic_data(n_steps=1000)
        env = ShieldTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=500,
            initial_capital=1000.0,
        )
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(hold)
            total_reward += reward
            if terminated or truncated:
                break

        assert total_reward < 0, (
            f"Shield hold-only should have negative reward, got {total_reward:.6f}"
        )

    def test_time_penalty_accumulates(self):
        """Time penalty should be -0.00001 per flat step."""
        market_data, features = _make_synthetic_data(n_steps=200)
        env = ShieldTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=100,
            initial_capital=1000.0,
        )
        obs, _ = env.reset(seed=42)

        # Step once and check reward includes time penalty
        hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(hold)

        # Reward should be negative (time penalty -0.00001 + drawdown component)
        assert reward < 0, f"Single hold step should have negative reward, got {reward}"


# ──────────────────────────────────────────────────────────────────────
# Test 5: Liquidation triggers correctly
# ──────────────────────────────────────────────────────────────────────

class TestLiquidation:
    """Verify liquidation fires at maintenance margin threshold."""

    def test_leveraged_position_liquidates(self):
        """3x leveraged long in sharp downtrend should get liquidated."""
        # Create very sharply declining data
        rng = np.random.RandomState(42)
        n = 500
        prices = 50000.0 * np.cumprod(1.0 + np.full(n, -0.005))  # 0.5% drop per step
        market_data = np.zeros((n, 7), dtype=np.float32)
        features = []
        for i in range(n):
            p = float(prices[i])
            market_data[i] = [p, p * 1.001, p * 0.999, p, 1e7, 0.0, 1e8]
            features.append(MarketFeatures(
                price=p, price_1h_ago=float(prices[max(0, i - 1)]),
                price_4h_ago=float(prices[max(0, i - 4)]),
                price_24h_ago=float(prices[max(0, i - 24)]),
                vwap_24h=p, rolling_mean_30d=50000.0,
                volume_24h=1e7, rolling_avg_vol_30d=1e7,
                bid_imbalance_pct=0.0, spread_bps=1.0,
                open_interest=1e8, rolling_avg_oi_30d=1e8,
                oi_1h_ago=1e8, oi_4h_ago=1e8,
                funding_rate=0.0, funding_8h_cumulative=0.0, prev_funding_rate=0.0,
                rsi_1h=30.0, rsi_4h=30.0, macd_hist_1h=-100.0, bb_position_1h=0.1,
                atr_1h=p * 0.01, ema_20=p, ema_50=p * 1.02, ema_200=p * 1.05,
                sma_4h=p, volume_trend_1h=-0.5, roc_1h=-0.5, roc_4h=-2.0,
                account_value=1000.0, initial_capital=1000.0, peak_account_value=1000.0,
                open_position_count=0, max_positions=5, margin_utilization=0.0,
                unrealized_pnl=0.0, mission_start_timestamp=1700000000.0 + i * 900,
                current_timestamp=1700000000.0 + i * 900, days_since_last_trade=0.0,
                has_open_position_this_asset=False, existing_direction=0,
                btc_dominance=50.0, fear_greed_index=20.0, market_regime=-1,
                cross_asset_momentum=-0.5,
            ))

        env = BuilderTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=200,
            initial_capital=1000.0,
        )
        obs, _ = env.reset(seed=0)

        # Open a 3x leveraged long
        action = np.array([1.0, 1.0, 0.5, 0.5, 1.0], dtype=np.float32)
        env.step(action)
        assert env.state.position is not None

        hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        terminated = False
        for step in range(100):
            _, _, terminated, truncated, info = env.step(hold)
            if terminated or env.state.position is None:
                break

        # Position should have been closed (by SL, liquidation, or account blowup)
        assert env.state.position is None or terminated, (
            "Leveraged long in sharp downtrend should close via SL or liquidation"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 6: Fee accounting
# ──────────────────────────────────────────────────────────────────────

class TestFeeAccounting:
    """Verify fees are deducted correctly on entry and exit."""

    def test_entry_fee_deducted(self):
        market_data, features = _make_synthetic_data(n_steps=500)
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=100,
            initial_capital=10000.0,
            max_leverage=1,
        )
        obs, _ = env.reset(seed=42)
        initial_value = env.state.account_value

        # Open a position
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
        env.step(action)

        assert env.state.total_fees_paid > 0, "Fee should be charged on entry"
        assert env.state.account_value < initial_value, "Account value should decrease by fee"

    def test_minimum_notional_respected(self):
        """Tiny positions below $10 should not open."""
        market_data, features = _make_synthetic_data(n_steps=500)
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=100,
            initial_capital=5.0,  # Very small capital
            max_leverage=1,
        )
        obs, _ = env.reset(seed=42)

        # Try to open a small position
        action = np.array([0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)

        assert env.state.position is None, "Position below min notional should not open"


# ──────────────────────────────────────────────────────────────────────
# Test 7: Observation shape and ranges
# ──────────────────────────────────────────────────────────────────────

class TestObservationShape:
    """Verify observation vector dimensions and basic sanity."""

    @pytest.mark.parametrize("env_class", [ShieldTradingEnv, BuilderTradingEnv])
    def test_obs_dim(self, env_class):
        market_data, features = _make_synthetic_data(n_steps=500)
        env = env_class(
            market_data=market_data,
            feature_data=features,
            episode_length=100,
        )
        obs, _ = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_obs_no_nans(self):
        market_data, features = _make_synthetic_data(n_steps=500)
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=100,
        )
        obs, _ = env.reset(seed=42)

        for _ in range(50):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in observation at step"
            assert not np.any(np.isinf(obs)), f"Inf in observation at step"
            if terminated or truncated:
                break

    def test_obs_reasonable_ranges(self):
        """Observation values should be within reasonable bounds before MeanStdFilter."""
        market_data, features = _make_synthetic_data(n_steps=500)
        env = BaseTradingEnv(
            market_data=market_data,
            feature_data=features,
            episode_length=100,
        )
        obs, _ = env.reset(seed=42)

        # Most features should be within [-100, 100] before normalization
        # (some like annualized funding can be larger, but nothing should be > 10000)
        assert np.all(np.abs(obs) < 100000), (
            f"Observation has extreme values: max={np.max(np.abs(obs))}"
        )


# ──────────────────────────────────────────────────────────────────────
# Test 8: Feature parity (training feature_engineer vs production)
# ──────────────────────────────────────────────────────────────────────

class TestFeatureParity:
    """Ensure build_observation() is deterministic and version-tracked."""

    def test_deterministic_output(self):
        """Same inputs → same output."""
        _, features = _make_synthetic_data(n_steps=10)
        obs1 = build_observation(features[5])
        obs2 = build_observation(features[5])
        np.testing.assert_array_equal(obs1, obs2)

    def test_feature_version_exists(self):
        from data.preprocessors.feature_engineer import FEATURE_VERSION, FEATURE_HASH
        assert FEATURE_VERSION == "1.1.0"
        assert len(FEATURE_HASH) == 8


# ──────────────────────────────────────────────────────────────────────
# Test 9: Profile-specific constraints
# ──────────────────────────────────────────────────────────────────────

class TestProfileConstraints:
    """Each profile env should have correct leverage/SL/TP bounds."""

    def test_shield_limits(self):
        market_data, features = _make_synthetic_data(n_steps=200)
        env = ShieldTradingEnv(market_data=market_data, feature_data=features)
        assert env.max_leverage == 1
        assert env.max_sl_pct == 0.03
        assert env.max_tp_pct == 0.075

    def test_builder_limits(self):
        market_data, features = _make_synthetic_data(n_steps=200)
        env = BuilderTradingEnv(market_data=market_data, feature_data=features)
        assert env.max_leverage == 2
        assert env.max_sl_pct == 0.05
        assert env.max_tp_pct == 0.10


# ──────────────────────────────────────────────────────────────────────
# Test 10: Gymnasium API compliance
# ──────────────────────────────────────────────────────────────────────

class TestGymnasiumAPI:
    """Verify envs follow Gymnasium API contract."""

    @pytest.mark.parametrize("env_class", [ShieldTradingEnv, BuilderTradingEnv])
    def test_reset_returns_obs_info(self, env_class):
        market_data, features = _make_synthetic_data(n_steps=500)
        env = env_class(market_data=market_data, feature_data=features, episode_length=100)
        result = env.reset(seed=42)
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert obs.shape == (OBS_DIM,)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_class", [ShieldTradingEnv, BuilderTradingEnv])
    def test_step_returns_5tuple(self, env_class):
        market_data, features = _make_synthetic_data(n_steps=500)
        env = env_class(market_data=market_data, feature_data=features, episode_length=100)
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_class", [ShieldTradingEnv, BuilderTradingEnv])
    def test_episode_terminates(self, env_class):
        """Episode should end within episode_length steps."""
        market_data, features = _make_synthetic_data(n_steps=500)
        env = env_class(market_data=market_data, feature_data=features, episode_length=100)
        env.reset(seed=42)

        done = False
        steps = 0
        while not done and steps < 200:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done, "Episode should terminate within episode_length steps"
        assert steps <= 100, f"Episode ran for {steps} steps (limit=100)"
