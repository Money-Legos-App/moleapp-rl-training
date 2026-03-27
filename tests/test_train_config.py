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
        """entropy_coeff_schedule should be a list of [timestep, value] pairs."""
        schedule = shield_config.get("entropy_coeff_schedule")
        assert schedule is not None, "Shield config missing entropy_coeff_schedule"
        assert isinstance(schedule, list)
        assert len(schedule) >= 2
        # Each entry is [timestep, coeff]
        for entry in schedule:
            assert len(entry) == 2
            assert isinstance(entry[0], (int, float))
            assert isinstance(entry[1], float)

    def test_builder_has_entropy_schedule(self, builder_config):
        schedule = builder_config.get("entropy_coeff_schedule")
        assert schedule is not None, "Builder config missing entropy_coeff_schedule"
        assert isinstance(schedule, list)
        assert len(schedule) >= 2

    def test_entropy_schedule_decays(self, shield_config):
        """Entropy should decrease over training (exploration annealing)."""
        schedule = shield_config["entropy_coeff_schedule"]
        assert schedule[0][1] > schedule[-1][1], "Entropy should decay, not increase"

    def test_shield_no_flat_entropy_coeff(self, shield_config):
        """When using schedule, flat entropy_coeff should NOT be present."""
        assert "entropy_coeff" not in shield_config, (
            "Config has both entropy_coeff and entropy_coeff_schedule — "
            "remove flat entropy_coeff to avoid confusion"
        )

    def test_shield_minibatch_256(self, shield_config):
        assert shield_config["minibatch_size"] == 256

    def test_builder_minibatch_256(self, builder_config):
        assert builder_config["minibatch_size"] == 256

    def test_shield_parallelism(self, shield_config):
        assert shield_config["num_env_runners"] == 32
        assert shield_config["num_envs_per_env_runner"] == 4

    def test_builder_parallelism(self, builder_config):
        assert builder_config["num_env_runners"] == 32
        assert builder_config["num_envs_per_env_runner"] == 4

    def test_shield_batch_size(self, shield_config):
        assert shield_config["train_batch_size_per_learner"] == 16384

    def test_builder_batch_size(self, builder_config):
        assert builder_config["train_batch_size_per_learner"] == 16384

    def test_shield_gamma_0995(self, shield_config):
        assert shield_config["gamma"] == 0.995

    def test_builder_gamma_0995(self, builder_config):
        assert builder_config["gamma"] == 0.995

    def test_shield_eval_parallel(self, shield_config):
        eval_cfg = shield_config.get("evaluation", {})
        assert eval_cfg.get("evaluation_parallel_to_training") is True

    def test_builder_eval_parallel(self, builder_config):
        eval_cfg = builder_config.get("evaluation", {})
        assert eval_cfg.get("evaluation_parallel_to_training") is True

    def test_configs_have_same_entropy_schedule(self, shield_config, builder_config):
        """Both strategies should use the same entropy schedule."""
        assert shield_config["entropy_coeff_schedule"] == builder_config["entropy_coeff_schedule"]


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
        """entropy_coeff should be the schedule list, not a flat float."""
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        ec = ppo_config.entropy_coeff
        assert isinstance(ec, list), (
            f"Expected entropy_coeff to be a schedule (list), got {type(ec)}: {ec}"
        )
        assert len(ec) >= 2

    def test_minibatch_size_propagated(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.minibatch_size == 256

    def test_gamma_propagated(self, shield_config):
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", shield_config, env_config)
        assert ppo_config.gamma == 0.995

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

    def test_flat_entropy_fallback(self):
        """When no schedule is in config, flat entropy_coeff is used."""
        config = {
            "entropy_coeff": 0.05,
            "model": {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"},
        }
        env_config = _dummy_env_config()
        ppo_config = build_ppo_config("shield", config, env_config)
        assert ppo_config.entropy_coeff == 0.05
