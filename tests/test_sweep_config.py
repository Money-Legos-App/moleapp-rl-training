"""
Tests for the Ray Tune sweep configuration, trading callbacks, and observation v1.1.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from data.preprocessors.feature_engineer import (
    ASSET_ID_MAP,
    FEATURE_VERSION,
    OBS_DIM,
    MarketFeatures,
    build_observation,
)


# ──────────────────────────────────────────────────────────────────────
# Test: Tune Sweep YAML Config
# ──────────────────────────────────────────────────────────────────────

SHIELD_TUNE_PATH = Path("training/configs/tune_shield.yaml")
BUILDER_TUNE_PATH = Path("training/configs/tune_builder.yaml")


class TestShieldSweepConfig:
    @pytest.fixture(autouse=True)
    def load_config(self):
        assert SHIELD_TUNE_PATH.exists()
        with open(SHIELD_TUNE_PATH) as f:
            self.config = yaml.safe_load(f)

    def test_method_is_asha(self):
        assert self.config["sweep"]["method"] == "asha"

    def test_shield_optimizes_risk_adjusted_return(self):
        """Shield must optimize risk_adjusted_return, NOT raw return."""
        assert self.config["sweep"]["metric"] == "env_runners/risk_adjusted_return"
        assert self.config["sweep"]["mode"] == "max"

    def test_grace_period_at_least_500k(self):
        """500K grace period protects late-blooming agents."""
        assert self.config["sweep"]["grace_period"] >= 500_000

    def test_swept_params_present(self):
        params = self.config["sweep"]["parameters"]
        for key in ("_peak_lr", "_start_entropy", "minibatch_size", "gamma", "clip_param"):
            assert key in params

    def test_shield_tight_clip_range(self):
        """Shield clip_param should be tighter than Builder."""
        cp = self.config["sweep"]["parameters"]["clip_param"]
        assert cp["max"] <= 0.20  # institutional grade — no wild swings

    def test_shield_high_gamma(self):
        """Shield gamma should favor long-horizon planning."""
        g = self.config["sweep"]["parameters"]["gamma"]
        assert g["min"] >= 0.99

    def test_shield_low_entropy(self):
        """Shield entropy should start lower — less random exploration."""
        ec = self.config["sweep"]["parameters"]["_start_entropy"]
        assert float(ec["max"]) <= 0.01

    def test_num_samples_reasonable(self):
        assert 10 <= self.config["sweep"]["num_samples"] <= 100


class TestBuilderSweepConfig:
    @pytest.fixture(autouse=True)
    def load_config(self):
        assert BUILDER_TUNE_PATH.exists()
        with open(BUILDER_TUNE_PATH) as f:
            self.config = yaml.safe_load(f)

    def test_builder_optimizes_raw_return(self):
        """Builder optimizes episode_return_mean — aggressive alpha."""
        assert self.config["sweep"]["metric"] == "env_runners/episode_return_mean"

    def test_grace_period_at_least_500k(self):
        assert self.config["sweep"]["grace_period"] >= 500_000

    def test_builder_wider_clip_range(self):
        """Builder clip_param can be looser for faster adaptation."""
        cp = self.config["sweep"]["parameters"]["clip_param"]
        assert cp["max"] >= 0.25

    def test_builder_higher_entropy(self):
        """Builder allows higher entropy for more exploration."""
        ec = self.config["sweep"]["parameters"]["_start_entropy"]
        assert float(ec["max"]) >= 0.015

    def test_builder_lower_gamma(self):
        """Builder can use shorter horizons for near-term alpha."""
        g = self.config["sweep"]["parameters"]["gamma"]
        assert g["min"] <= 0.995

    def test_divergent_from_shield(self):
        """Shield and Builder configs must NOT be identical."""
        with open(SHIELD_TUNE_PATH) as f:
            shield = yaml.safe_load(f)
        assert self.config["sweep"]["metric"] != shield["sweep"]["metric"]


# ──────────────────────────────────────────────────────────────────────
# Test: Sharpe Computation
# ──────────────────────────────────────────────────────────────────────

class TestSharpeComputation:
    def test_positive_returns_positive_sharpe(self):
        from training.callbacks.trading_callbacks import compute_sharpe

        returns = [0.05, 0.03, 0.04, 0.06, 0.02]
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        from training.callbacks.trading_callbacks import compute_sharpe

        returns = [-0.05, -0.03, -0.04, -0.06, -0.02]
        sharpe = compute_sharpe(returns)
        assert sharpe < 0

    def test_zero_variance_returns_zero(self):
        from training.callbacks.trading_callbacks import compute_sharpe

        returns = [0.05, 0.05, 0.05, 0.05]
        sharpe = compute_sharpe(returns)
        assert sharpe == 0.0

    def test_single_episode_returns_zero(self):
        from training.callbacks.trading_callbacks import compute_sharpe

        sharpe = compute_sharpe([0.05])
        assert sharpe == 0.0

    def test_empty_returns_zero(self):
        from training.callbacks.trading_callbacks import compute_sharpe

        sharpe = compute_sharpe([])
        assert sharpe == 0.0

    def test_annualization_factor(self):
        from training.callbacks.trading_callbacks import compute_sharpe

        returns = [0.1, 0.2, 0.1, 0.2, 0.1]
        sharpe_annual = compute_sharpe(returns, annualize_factor=3.49)
        sharpe_raw = compute_sharpe(returns, annualize_factor=1.0)
        assert abs(sharpe_annual / sharpe_raw - 3.49) < 0.01


# ──────────────────────────────────────────────────────────────────────
# Test: TradingCallbacks
# ──────────────────────────────────────────────────────────────────────

class TestTradingCallbacks:
    def test_callback_import(self):
        """TradingCallbacks imports without error."""
        from training.callbacks.trading_callbacks import TradingCallbacks
        assert TradingCallbacks is not None

    def test_callback_is_default_callbacks_subclass(self):
        """TradingCallbacks extends RLlib DefaultCallbacks."""
        from ray.rllib.algorithms.callbacks import DefaultCallbacks
        from training.callbacks.trading_callbacks import TradingCallbacks

        assert issubclass(TradingCallbacks, DefaultCallbacks)


# ──────────────────────────────────────────────────────────────────────
# Test: Tune Sweep Module Imports
# ──────────────────────────────────────────────────────────────────────

class TestTuneSweepImports:
    def test_tune_sweep_imports(self):
        """tune_sweep.py imports without error."""
        from training.tune_sweep import SHIELD_ENV_KWARGS, PROFILE_ENV_KWARGS, PROFILE_SEARCH
        assert SHIELD_ENV_KWARGS["max_leverage"] == 1
        assert "shield" in PROFILE_ENV_KWARGS
        assert "builder" in PROFILE_ENV_KWARGS
        assert "shield" in PROFILE_SEARCH
        assert "builder" in PROFILE_SEARCH

    def test_shield_env_kwargs_match_config(self):
        """Tune sweep's Shield params match shield_config.yaml."""
        from training.tune_sweep import PROFILE_ENV_KWARGS

        shield = PROFILE_ENV_KWARGS["shield"]
        assert shield["max_leverage"] == 1
        assert shield["max_positions"] == 2
        assert shield["max_sl_pct"] == 0.03
        assert shield["max_tp_pct"] == 0.075
        assert shield["initial_capital"] == 1000.0

    def test_builder_env_kwargs_match_config(self):
        """Tune sweep's Builder params match builder_config.yaml."""
        from training.tune_sweep import PROFILE_ENV_KWARGS

        builder = PROFILE_ENV_KWARGS["builder"]
        assert builder["max_leverage"] == 2
        assert builder["max_positions"] == 4
        assert builder["max_drawdown_pct"] == 0.20

    def test_lr_schedule_structure(self):
        """LR schedule should have warmup-peak-decay structure."""
        from training.tune_sweep import _make_lr_schedule
        schedule = _make_lr_schedule(3e-4)
        assert len(schedule) == 3
        assert schedule[0][1] < schedule[1][1]  # warmup
        assert schedule[2][1] < schedule[1][1]  # decay

    def test_entropy_schedule_decays(self):
        """Entropy schedule should decay from start to near-zero."""
        from training.tune_sweep import _make_entropy_schedule
        schedule = _make_entropy_schedule(0.005)
        assert schedule[0][1] > schedule[-1][1]
        assert schedule[-1][1] == 0.0001

    def test_shield_metric_is_risk_adjusted(self):
        """Shield sweep must optimize risk_adjusted_return."""
        from training.tune_sweep import PROFILE_SEARCH
        assert PROFILE_SEARCH["shield"]["metric"] == "env_runners/risk_adjusted_return"

    def test_builder_metric_is_raw_return(self):
        """Builder sweep must optimize episode_return_mean."""
        from training.tune_sweep import PROFILE_SEARCH
        assert PROFILE_SEARCH["builder"]["metric"] == "env_runners/episode_return_mean"


# ──────────────────────────────────────────────────────────────────────
# Test: Feature Version v1.1.0
# ──────────────────────────────────────────────────────────────────────

class TestFeatureVersion:
    def test_version_is_1_1_0(self):
        assert FEATURE_VERSION == "1.1.0"

    def test_obs_dim_unchanged(self):
        assert OBS_DIM == 47


# ──────────────────────────────────────────────────────────────────────
# Test: Asset Identity Encoding (Index 45)
# ──────────────────────────────────────────────────────────────────────

class TestAssetIdentity:
    def test_asset_id_map_has_15_assets(self):
        assert len(ASSET_ID_MAP) == 15

    def test_all_ids_in_range(self):
        for asset, val in ASSET_ID_MAP.items():
            assert 0.0 <= val <= 1.0, f"{asset} has ID {val} outside [0, 1]"

    def test_btc_id(self):
        assert abs(ASSET_ID_MAP["BTC"] - 2.0 / 14.0) < 1e-6

    def test_first_and_last(self):
        assert ASSET_ID_MAP["ARB"] == 0.0
        assert ASSET_ID_MAP["kPEPE"] == 1.0  # lowercase 'k' sorts last in ASCII

    def test_unknown_asset_defaults_to_zero(self):
        assert ASSET_ID_MAP.get("UNKNOWN", 0.0) == 0.0

    def test_obs_45_reflects_asset_id(self):
        """build_observation() puts asset_id_normalized at index 45."""
        mf = _make_default_features(asset_id_normalized=ASSET_ID_MAP["SOL"])
        obs = build_observation(mf)
        assert abs(obs[45] - ASSET_ID_MAP["SOL"]) < 1e-6

    def test_obs_45_zero_for_default(self):
        mf = _make_default_features()
        obs = build_observation(mf)
        assert obs[45] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Test: Distance to Liquidation (Index 46)
# ──────────────────────────────────────────────────────────────────────

class TestDistanceToLiquidation:
    def test_no_position_is_safe(self):
        mf = _make_default_features(distance_to_liquidation=1.0)
        obs = build_observation(mf)
        assert obs[46] == 1.0

    def test_near_liquidation_is_low(self):
        mf = _make_default_features(distance_to_liquidation=0.05)
        obs = build_observation(mf)
        assert abs(obs[46] - 0.05) < 1e-6

    def test_at_liquidation_is_zero(self):
        mf = _make_default_features(distance_to_liquidation=0.0)
        obs = build_observation(mf)
        assert obs[46] == 0.0

    def test_clamped_to_unit_range(self):
        """Values should always be in [0, 1]."""
        for val in [0.0, 0.5, 1.0]:
            mf = _make_default_features(distance_to_liquidation=val)
            obs = build_observation(mf)
            assert 0.0 <= obs[46] <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Test: Shield Volatility Bonus Fix
# ──────────────────────────────────────────────────────────────────────

class TestShieldVolatilityFix:
    def test_never_trade_gets_no_vol_bonus(self):
        """Agent that never trades should get zero volatility bonus."""
        from envs.shield_env import ShieldTradingEnv

        env = ShieldTradingEnv.__new__(ShieldTradingEnv)
        env._last_close_step = 0
        env._had_position = False
        env._prev_drawdown = 0.0

        ctx = {"pnl_pct": 0.0, "drawdown": 0.05, "has_position": False,
               "step": 100, "funding_cost": 0.0, "current_price": 100.0,
               "unrealized_pnl_pct": 0.0}
        reward = env._calculate_reward(ctx)
        # Should only have drawdown penalty + time penalty, NO volatility bonus
        assert reward < 0.0

    def test_bonus_after_close(self):
        """Agent should get decaying bonus after closing during drawdown."""
        from envs.shield_env import ShieldTradingEnv

        env = ShieldTradingEnv.__new__(ShieldTradingEnv)
        env._last_close_step = 0
        env._had_position = True  # had position last step
        env._prev_drawdown = 0.0

        # Step where position closes (drawdown > 3%)
        ctx1 = {"pnl_pct": 0.0, "drawdown": 0.04, "has_position": False,
                "step": 50, "funding_cost": 0.0, "current_price": 100.0,
                "unrealized_pnl_pct": 0.0}
        env._calculate_reward(ctx1)
        assert env._last_close_step == 50

        # Next step: should get bonus (steps_since_close = 1, decay ~ 0.98)
        ctx2 = {"pnl_pct": 0.0, "drawdown": 0.04, "has_position": False,
                "step": 51, "funding_cost": 0.0, "current_price": 100.0,
                "unrealized_pnl_pct": 0.0}
        reward = env._calculate_reward(ctx2)
        # Rewards are ×100 scaled. Reward includes: drawdown penalty + vol bonus + time penalty
        # The vol bonus should make it less negative than without
        assert reward > -2.5  # ×100 scaled: was -0.025

    def test_bonus_expires_after_48_steps(self):
        """Bonus should be zero after 48 steps."""
        from envs.shield_env import ShieldTradingEnv

        env = ShieldTradingEnv.__new__(ShieldTradingEnv)
        env._last_close_step = 50
        env._had_position = False
        env._prev_drawdown = 0.0

        # 49 steps after close: no bonus
        ctx = {"pnl_pct": 0.0, "drawdown": 0.04, "has_position": False,
               "step": 99, "funding_cost": 0.0, "current_price": 100.0,
               "unrealized_pnl_pct": 0.0}
        reward_expired = env._calculate_reward(ctx)

        # 1 step after close: has bonus
        ctx2 = {"pnl_pct": 0.0, "drawdown": 0.04, "has_position": False,
                "step": 51, "funding_cost": 0.0, "current_price": 100.0,
                "unrealized_pnl_pct": 0.0}
        reward_active = env._calculate_reward(ctx2)

        assert reward_active > reward_expired


# ──────────────────────────────────────────────────────────────────────
# Helper: Default MarketFeatures for tests
# ──────────────────────────────────────────────────────────────────────

def _make_default_features(**overrides) -> MarketFeatures:
    """Create a MarketFeatures with sensible defaults for testing."""
    defaults = dict(
        price=100.0, price_1h_ago=99.0, price_4h_ago=98.0, price_24h_ago=95.0,
        vwap_24h=99.5, rolling_mean_30d=97.0,
        volume_24h=1e6, rolling_avg_vol_30d=1e6,
        bid_imbalance_pct=0.0, spread_bps=5.0,
        open_interest=1e8, rolling_avg_oi_30d=1e8,
        oi_1h_ago=1e8, oi_4h_ago=1e8,
        funding_rate=0.0001, funding_8h_cumulative=0.0003, prev_funding_rate=0.0001,
        rsi_1h=50.0, rsi_4h=50.0, macd_hist_1h=0.0, bb_position_1h=0.5,
        atr_1h=2.0, ema_20=100.0, ema_50=99.0, ema_200=95.0, sma_4h=99.5,
        volume_trend_1h=0.0, roc_1h=1.0, roc_4h=2.0,
        account_value=1000.0, initial_capital=1000.0, peak_account_value=1000.0,
        open_position_count=0, max_positions=2, margin_utilization=0.0,
        unrealized_pnl=0.0, mission_start_timestamp=1700000000.0,
        current_timestamp=1700086400.0, days_since_last_trade=0.0,
        has_open_position_this_asset=False, existing_direction=0,
        btc_dominance=50.0, fear_greed_index=50.0, market_regime=0,
        cross_asset_momentum=0.0,
    )
    defaults.update(overrides)
    return MarketFeatures(**defaults)
