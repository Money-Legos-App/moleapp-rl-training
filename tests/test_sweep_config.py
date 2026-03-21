"""
Tests for the W&B Sweep configuration and evaluation callback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml


# ──────────────────────────────────────────────────────────────────────
# Test: Sweep YAML Config
# ──────────────────────────────────────────────────────────────────────

SWEEP_CONFIG_PATH = Path("training/configs/sweep_shield.yaml")


class TestSweepConfig:
    @pytest.fixture(autouse=True)
    def load_config(self):
        assert SWEEP_CONFIG_PATH.exists(), f"Sweep config not found at {SWEEP_CONFIG_PATH}"
        with open(SWEEP_CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)

    def test_method_is_bayes(self):
        assert self.config["method"] == "bayes"

    def test_metric_is_sharpe(self):
        assert self.config["metric"]["name"] == "eval/sharpe_ratio"
        assert self.config["metric"]["goal"] == "maximize"

    def test_all_four_params_present(self):
        params = self.config["parameters"]
        assert "learning_rate" in params
        assert "ent_coef" in params
        assert "batch_size" in params
        assert "gamma" in params

    def test_learning_rate_range(self):
        lr = self.config["parameters"]["learning_rate"]
        assert float(lr["min"]) >= 1e-6
        assert float(lr["max"]) <= 1e-2
        assert lr["distribution"] == "log_uniform_values"

    def test_ent_coef_range(self):
        ec = self.config["parameters"]["ent_coef"]
        assert float(ec["min"]) >= 1e-4
        assert float(ec["max"]) <= 0.1
        assert ec["distribution"] == "log_uniform_values"

    def test_batch_size_values(self):
        bs = self.config["parameters"]["batch_size"]
        assert "values" in bs
        for v in bs["values"]:
            # Must be powers of 2
            assert v > 0 and (v & (v - 1)) == 0, f"{v} is not a power of 2"

    def test_gamma_range(self):
        g = self.config["parameters"]["gamma"]
        assert g["min"] >= 0.9
        assert g["max"] <= 1.0

    def test_early_termination_configured(self):
        et = self.config["early_terminate"]
        assert et["type"] == "hyperband"
        assert et["min_iter"] > 0

    def test_run_cap_is_reasonable(self):
        assert 10 <= self.config["run_cap"] <= 100

    def test_program_points_to_sweep_agent(self):
        assert "sweep_agent" in self.config["program"]


# ──────────────────────────────────────────────────────────────────────
# Test: Sharpe Computation
# ──────────────────────────────────────────────────────────────────────

class TestSharpeComputation:
    def test_positive_returns_positive_sharpe(self):
        from training.callbacks.sweep_eval_callback import compute_sharpe

        returns = [0.05, 0.03, 0.04, 0.06, 0.02]
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        from training.callbacks.sweep_eval_callback import compute_sharpe

        returns = [-0.05, -0.03, -0.04, -0.06, -0.02]
        sharpe = compute_sharpe(returns)
        assert sharpe < 0

    def test_zero_variance_returns_zero(self):
        from training.callbacks.sweep_eval_callback import compute_sharpe

        returns = [0.05, 0.05, 0.05, 0.05]
        sharpe = compute_sharpe(returns)
        assert sharpe == 0.0

    def test_single_episode_returns_zero(self):
        from training.callbacks.sweep_eval_callback import compute_sharpe

        sharpe = compute_sharpe([0.05])
        assert sharpe == 0.0

    def test_empty_returns_zero(self):
        from training.callbacks.sweep_eval_callback import compute_sharpe

        sharpe = compute_sharpe([])
        assert sharpe == 0.0

    def test_annualization_factor(self):
        from training.callbacks.sweep_eval_callback import compute_sharpe

        returns = [0.1, 0.2, 0.1, 0.2, 0.1]
        sharpe_annual = compute_sharpe(returns, annualize_factor=3.49)
        sharpe_raw = compute_sharpe(returns, annualize_factor=1.0)
        assert abs(sharpe_annual / sharpe_raw - 3.49) < 0.01


# ──────────────────────────────────────────────────────────────────────
# Test: SweepEvalCallback
# ──────────────────────────────────────────────────────────────────────

class TestSweepEvalCallback:
    def test_callback_import(self):
        """SweepEvalCallback imports without error."""
        from training.callbacks.sweep_eval_callback import SweepEvalCallback
        assert SweepEvalCallback is not None

    def test_callback_init(self):
        """SweepEvalCallback can be instantiated with a mock env."""
        from unittest.mock import MagicMock
        from training.callbacks.sweep_eval_callback import SweepEvalCallback

        mock_env = MagicMock()
        cb = SweepEvalCallback(
            eval_env=mock_env,
            eval_freq=1000,
            n_eval_episodes=5,
        )
        assert cb.eval_freq == 1000
        assert cb.n_eval_episodes == 5


# ──────────────────────────────────────────────────────────────────────
# Test: Sweep Agent Imports
# ──────────────────────────────────────────────────────────────────────

class TestSweepAgentImports:
    def test_sweep_agent_imports(self):
        """sweep_agent.py imports without error."""
        from training.sweep_agent import SHIELD_ENV_KWARGS, FIXED_PPO, load_episode_data
        assert SHIELD_ENV_KWARGS["max_leverage"] == 1
        assert "n_steps" in FIXED_PPO

    def test_shield_env_kwargs_match_config(self):
        """Sweep agent's Shield params match shield_config.yaml."""
        from training.sweep_agent import SHIELD_ENV_KWARGS

        assert SHIELD_ENV_KWARGS["max_leverage"] == 1
        assert SHIELD_ENV_KWARGS["max_positions"] == 2
        assert SHIELD_ENV_KWARGS["max_sl_pct"] == 0.03
        assert SHIELD_ENV_KWARGS["max_tp_pct"] == 0.06
        assert SHIELD_ENV_KWARGS["initial_capital"] == 1000.0
