"""
Tests for training configuration parsing and PPO config builder.

Validates that:
1. YAML configs parse correctly (both shield and builder)
2. build_ppo_config() produces valid PPOConfig with expected values
3. Entropy schedule is passed correctly via entropy_coeff (not deprecated param)
4. Evaluation parallel setting propagates
5. Minibatch size and gamma values match configs
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.train import build_ppo_config, load_config


SHIELD_CONFIG_PATH = Path("training/configs/shield_config.yaml")
BUILDER_CONFIG_PATH = Path("training/configs/builder_config.yaml")


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def shield_config():
    return load_config(str(SHIELD_CONFIG_PATH))


@pytest.fixture
def builder_config():
    return load_config(str(BUILDER_CONFIG_PATH))


def _dummy_env_config():
    """Minimal env_config for build_ppo_config (no real data needed)."""
    return {
        "market_data_path": "/tmp/fake_market.npy",
        "feature_data_path": "/tmp/fake_features.pkl",
        "max_leverage": 1,
        "max_positions": 2,
        "initial_capital": 1000.0,
        "max_sl_pct": 0.03,
        "min_sl_pct": 0.005,
        "max_tp_pct": 0.075,
        "min_tp_pct": 0.01,
        "episode_length": 100,
        "max_drawdown_pct": 0.10,
    }


# ──────────────────────────────────────────────────────────────────────
# Test: YAML Config Parsing
# ──────────────────────────────────────────────────────────────────────

class TestYAMLConfigParsing:
    def test_shield_config_exists(self):
        assert SHIELD_CONFIG_PATH.exists()

    def test_builder_config_exists(self):
        assert BUILDER_CONFIG_PATH.exists()

    def test_shield_has_entropy_schedule(self, shield_config):
        """V8: entropy annealing — 0.01 → 0.001 over 10M steps."""
        schedule = shield_config.get("entropy_coeff_schedule")
        assert schedule is not None, "Shield config missing entropy_coeff_schedule"
        assert isinstance(schedule, list)
        assert len(schedule) >= 3  # At least start, mid, end
        assert schedule[0][1] == 0.01  # Start high for exploration
        assert schedule[-1][1] <= 0.001  # End low for exploitation

    def test_shield_no_flat_entropy(self, shield_config):
        """V8: shield uses entropy schedule — flat coeff should NOT be present."""
        assert "entropy_coeff" not in shield_config, (
            "Shield V8 should use entropy_coeff_schedule, not flat entropy_coeff"
        )

    def test_shield_minibatch_256(self, shield_config):
        assert shield_config["minibatch_size"] == 256

    def test_builder_minibatch_256(self, builder_config):
        assert builder_config["minibatch_size"] == 256

    def test_shield_parallelism(self, shield_config):
        assert shield_config["num_env_runners"] == 10
        assert shield_config["num_envs_per_env_runner"] == 4

    def test_builder_parallelism(self, builder_config):
        assert builder_config["num_env_runners"] == 10
        assert builder_config["num_envs_per_env_runner"] == 4

    def test_shield_batch_size(self, shield_config):
        assert shield_config["train_batch_size_per_learner"] == 16384

    def test_builder_batch_size(self, builder_config):
        assert builder_config["train_batch_size_per_learner"] == 16384

    def test_shield_gamma(self, shield_config):
        assert shield_config["gamma"] == 0.9967  # V4 sweep-optimized

    def test_builder_gamma(self, builder_config):
        assert builder_config["gamma"] == 0.9963  # V4 sweep-optimized

    def test_shield_eval_parallel(self, shield_config):
        eval_cfg = shield_config.get("evaluation", {})
        assert eval_cfg.get("evaluation_parallel_to_training") is True

    def test_builder_eval_parallel(self, builder_config):
        eval_cfg = builder_config.get("evaluation", {})
        assert eval_cfg.get("evaluation_parallel_to_training") is True

    def test_builder_has_entropy_schedule(self, builder_config):
        """Builder still uses entropy schedule (higher LR/clip survives decay)."""
        schedule = builder_config.get("entropy_coeff_schedule")
        assert schedule is not None, "Builder config missing entropy_coeff_schedule"
        assert isinstance(schedule, list)
        assert len(schedule) >= 2

    def test_entropy_values(self, shield_config, builder_config):
        """Both use entropy schedules — shield starts at 0.01, builder at 0.005."""
        shield_start = shield_config["entropy_coeff_schedule"][0][1]
        builder_start = builder_config["entropy_coeff_schedule"][0][1]
        assert shield_start == 0.01  # V8: higher start for V6 reward exploration
        assert builder_start == 0.005

    # --- LR Schedule ---

    def test_shield_has_lr_schedule(self, shield_config):
        schedule = shield_config.get("lr_schedule")
        assert schedule is not None, "Shield config missing lr_schedule"
        assert isinstance(schedule, list)
        assert len(schedule) == 2  # V5: start at peak, linear decay to end

    def test_builder_has_lr_schedule(self, builder_config):
        schedule = builder_config.get("lr_schedule")
        assert schedule is not None
        assert len(schedule) == 3

    def test_shield_lr_decays_from_peak(self, shield_config):
        """V5: LR starts at sweep-optimal peak and decays linearly."""
        schedule = shield_config["lr_schedule"]
        start_lr = schedule[0][1]
        end_lr = schedule[-1][1]
        assert start_lr == 0.00005, "Should start at sweep-optimal 5e-5"
        assert end_lr < start_lr, "LR should decay"

    def test_builder_lr_warmup_then_decay(self, builder_config):
        """Builder still uses warmup+decay (higher LR survives it)."""
        schedule = builder_config["lr_schedule"]
        start_lr = schedule[0][1]
        peak_lr = schedule[1][1]
        end_lr = schedule[2][1]
        assert start_lr < peak_lr, "LR should warm up"
        assert end_lr < peak_lr, "LR should decay after peak"

    def test_shield_no_flat_lr(self, shield_config):
        """When using lr_schedule, flat lr should NOT be present."""
        assert "lr" not in shield_config

    # --- Clip Param ---

    def test_shield_clip_param(self, shield_config):
        assert shield_config["clip_param"] == 0.156  # V4 sweep-optimized

    def test_builder_clip_param(self, builder_config):
        assert builder_config["clip_param"] == 0.277  # V4 sweep-optimized


# ──────────────────────────────────────────────────────────────────────
# Test: build_ppo_config() Integration
# ──────────────────────────────────────────────────────────────────────

class TestBuildPPOConfig:
    def test_shield_config_builds_without_error(self, shield_config):
        """build_ppo_config should not raise for shield profile."""
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config is not None

    def test_builder_config_builds_without_error(self, builder_config):
        env_config = _dummy_env_config()
        env_config["max_leverage"] = 2
        env_config["max_positions"] = 4
        ppo_config = build_ppo_config("builder", builder_config, env_config)
        assert ppo_config is not None

    def test_entropy_schedule_propagated(self, shield_config):
        """V8: entropy_coeff should be the annealing schedule."""
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        ec = ppo_config.entropy_coeff
        assert isinstance(ec, list), (
            f"Expected entropy_coeff to be a schedule (list), got {type(ec)}: {ec}"
        )
        assert ec[0][1] == 0.01  # Start
        assert ec[-1][1] <= 0.001  # End

    def test_minibatch_size_propagated(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.minibatch_size == 256

    def test_gamma_propagated(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.gamma == 0.9967  # V4 sweep-optimized

    def test_clip_param_propagated(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.clip_param == 0.156  # V4 sweep-optimized

    def test_lr_schedule_propagated(self, shield_config):
        """lr should be the 2-point schedule (start at peak, decay)."""
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert isinstance(ppo_config.lr, list), (
            f"Expected lr to be a schedule (list), got {type(ppo_config.lr)}"
        )
        assert len(ppo_config.lr) == 2

    def test_eval_parallel_propagated(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.evaluation_parallel_to_training is True

    def test_eval_duration_unit_is_episodes(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.evaluation_duration_unit == "episodes"

    def test_eval_timeout_is_600s(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.evaluation_sample_timeout_s == 600.0

    def test_flat_lr_fallback(self):
        """When no lr_schedule, flat lr is used."""
        config = {
            "lr": 0.001,
            "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
        }
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", config, env_config)
        assert ppo_config.lr == 0.001

    def test_flat_entropy_fallback(self):
        """When no schedule is in config, flat entropy_coeff is used."""
        config = {
            "entropy_coeff": 0.05,
            "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
        }
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", config, env_config)
        assert ppo_config.entropy_coeff == 0.05


# ──────────────────────────────────────────────────────────────────────
# Test: Reward Normalization Wrapper
# ──────────────────────────────────────────────────────────────────────

class TestEnvWrappers:
    def test_env_has_normalize_observation_but_not_reward(self):
        """_make_env wraps with NormalizeObservation only — no NormalizeReward.

        NormalizeReward with high gamma + long episodes undoes ×100 reward scaling.
        """
        from gymnasium.wrappers import NormalizeObservation
        from envs import _make_env
        from envs.shield_env import ShieldTradingEnv
        from tests.test_environments import _make_synthetic_data

        market_data, features = _make_synthetic_data(n_steps=200)
        cfg = {
            "market_data": market_data,
            "feature_data": features,
            "max_leverage": 1,
            "max_positions": 2,
            "initial_capital": 1000.0,
            "max_sl_pct": 0.03,
            "min_sl_pct": 0.005,
            "max_tp_pct": 0.075,
            "min_tp_pct": 0.01,
            "episode_length": 100,
            "max_drawdown_pct": 0.10,
        }
        env = _make_env(ShieldTradingEnv, cfg)
        assert isinstance(env, NormalizeObservation)
        # NormalizeReward should NOT be present
        assert type(env).__name__ != "NormalizeReward"

    def test_reward_magnitude_with_100x_scaling(self):
        """Rewards should be in meaningful range (0.01-1.0) with ×100 scaling."""
        import numpy as np
        from envs.shield_env import ShieldTradingEnv
        from tests.test_environments import _make_synthetic_data

        market_data, features = _make_synthetic_data(n_steps=200, trend=0.001)
        env = ShieldTradingEnv(
            market_data=market_data, feature_data=features,
            initial_capital=1000.0, episode_length=50,
        )
        obs, _ = env.reset()
        rewards = []
        for _ in range(30):
            action = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
            obs, reward, term, trunc, _ = env.step(action)
            rewards.append(abs(reward))
            if term or trunc:
                break
        max_reward = max(rewards)
        assert max_reward > 0.001, f"Max reward {max_reward} is too small — ×100 scaling may not be working"
