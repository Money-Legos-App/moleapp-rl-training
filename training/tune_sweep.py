"""
Ray Tune Hyperparameter Sweep — Shield & Builder (V4)
======================================================
Divergent sweeps for two risk mandates:
  Shield: Optimizes risk_adjusted_return (return/drawdown) — institutional grade
  Builder: Optimizes episode_return_mean — aggressive alpha extraction

Uses ASHA scheduler with 500K grace period to protect late-blooming agents
that need time to map the value function before generating alpha.

Usage:
    # Shield sweep (risk-adjusted metric)
    python -m training.tune_sweep --profile shield

    # Builder sweep (return-maximizing)
    python -m training.tune_sweep --profile builder

    # Dry-run
    python -m training.tune_sweep --profile shield --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.tune import RunConfig as TuneRunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig

from envs import ENV_MAP
from training.callbacks.trading_callbacks import TradingCallbacks
from training.train import _load_training_data

logger = logging.getLogger(__name__)

# ── Sweep budget ────────────────────────────────────────────────────
SWEEP_TOTAL_TIMESTEPS = 2_000_000
EVAL_INTERVAL = 20

# ── Environment params per profile ──────────────────────────────────
PROFILE_ENV_KWARGS = {
    "shield": {
        "max_leverage": 1,
        "max_positions": 2,
        "initial_capital": 1000.0,
        "max_sl_pct": 0.03,
        "min_sl_pct": 0.005,
        "max_tp_pct": 0.075,
        "min_tp_pct": 0.01,
        "episode_length": 2880,
        "max_drawdown_pct": 0.10,
    },
    "builder": {
        "max_leverage": 2,
        "max_positions": 4,
        "initial_capital": 1000.0,
        "max_sl_pct": 0.05,
        "min_sl_pct": 0.01,
        "max_tp_pct": 0.10,
        "min_tp_pct": 0.02,
        "episode_length": 2880,
        "max_drawdown_pct": 0.20,
    },
}

# ── Divergent search spaces per profile ─────────────────────────────
# Shield: tighter clip, higher gamma, larger batch — stability over speed
# Builder: looser clip, lower gamma, higher entropy — exploit short-term alpha
PROFILE_SEARCH = {
    "shield": {
        "peak_lrs": [5e-5, 1e-4, 2e-4, 3e-4],       # Conservative LR range
        "start_entropies": [0.002, 0.005, 0.008],      # Lower entropy — less random exploration
        "gamma_range": (0.995, 0.999),                  # Long horizon — patient risk management
        "clip_range": (0.08, 0.18),                     # Tight — slow, methodical policy updates
        "minibatch_choices": [256, 512],                 # Larger — smoother gradients
        "epoch_choices": [4, 6],                         # Fewer — avoid overfitting
        "batch_choices": [16384],                        # Large — stable updates
        "metric": "env_runners/risk_adjusted_return",    # Return / drawdown
        "num_samples": 20,
        "grace_period": 500_000,                         # Don't kill late bloomers
    },
    "builder": {
        "peak_lrs": [1e-4, 3e-4, 5e-4, 1e-3],         # Wider LR range — can be aggressive
        "start_entropies": [0.005, 0.01, 0.015, 0.02], # Higher entropy — explore more
        "gamma_range": (0.99, 0.997),                    # Shorter horizon — exploit near-term
        "clip_range": (0.15, 0.30),                      # Looser — faster policy adaptation
        "minibatch_choices": [128, 256, 512],             # Include smaller for noise injection
        "epoch_choices": [4, 6, 8],                       # More passes OK with higher entropy
        "batch_choices": [8192, 16384],                   # Both sizes
        "metric": "env_runners/episode_return_mean",      # Pure return maximization
        "num_samples": 20,
        "grace_period": 500_000,
    },
}

# Keep SHIELD_ENV_KWARGS as alias for backward compatibility with tests
SHIELD_ENV_KWARGS = PROFILE_ENV_KWARGS["shield"]


def _make_lr_schedule(peak_lr: float) -> list:
    """Build V3-style warmup+decay LR schedule from a peak LR value."""
    return [
        [0, peak_lr / 30],              # Start at 1/30 of peak
        [100_000, peak_lr],              # Ramp to peak by 100K
        [1_600_000, peak_lr / 10],       # Decay to 10% by 80% of sweep budget
    ]


def _make_entropy_schedule(start_entropy: float) -> list:
    """Build V3-style entropy annealing schedule from start value."""
    return [
        [0, start_entropy],
        [1_000_000, start_entropy / 5],     # 5x reduction by midpoint
        [2_000_000, 0.0001],                # Near-zero at end
    ]


def build_sweep_config(
    profile: str,
    train_market: np.ndarray,
    train_features: list,
    eval_market: np.ndarray,
    eval_features: list,
) -> PPOConfig:
    """Build RLlib PPOConfig with profile-specific search spaces."""
    env_kwargs = PROFILE_ENV_KWARGS[profile]
    search = PROFILE_SEARCH[profile]

    train_env_config = {
        "market_data": train_market,
        "feature_data": train_features,
        **env_kwargs,
    }
    eval_env_config = {
        "market_data": eval_market,
        "feature_data": eval_features,
        **env_kwargs,
    }

    config = (
        PPOConfig()
        .environment(
            env=ENV_MAP[profile],
            env_config=train_env_config,
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=2,
        )
        .training(
            lr=tune.choice([_make_lr_schedule(p) for p in search["peak_lrs"]]),
            gamma=tune.uniform(*search["gamma_range"]),
            entropy_coeff=tune.choice([_make_entropy_schedule(e) for e in search["start_entropies"]]),
            minibatch_size=tune.choice(search["minibatch_choices"]),
            clip_param=tune.uniform(*search["clip_range"]),
            lambda_=0.95,
            num_epochs=tune.choice(search["epoch_choices"]),
            train_batch_size_per_learner=tune.choice(search["batch_choices"]),
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "tanh",
            },
        )
        .evaluation(
            evaluation_interval=EVAL_INTERVAL,
            evaluation_duration=3,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=1,
            evaluation_sample_timeout_s=600,
            evaluation_parallel_to_training=True,
            evaluation_config=PPOConfig.overrides(
                env_config=eval_env_config,
                explore=False,
            ),
        )
        .callbacks(callbacks_class=TradingCallbacks)
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
    )

    return config


def run_sweep(
    profile: str,
    config_path: str | None = None,
    episode_dir: str = "data/episodes",
):
    """Run profile-specific Ray Tune hyperparameter sweep."""
    search = PROFILE_SEARCH[profile]
    ray.init(ignore_reinit_error=True)

    # Load multi-asset data (same pipeline as train.py)
    market_data, feature_data = _load_training_data(
        data_dir="data/datasets",
        episode_dir=episode_dir,
    )

    # Split train/eval (80/20)
    split_idx = int(len(market_data) * 0.8)
    train_md = market_data[:split_idx]
    train_fd = feature_data[:split_idx]
    eval_md = market_data[split_idx:]
    eval_fd = feature_data[split_idx:]

    logger.info(f"[{profile.upper()}] Sweep data: train={len(train_fd)}, eval={len(eval_fd)} timesteps")
    logger.info(f"[{profile.upper()}] Optimization metric: {search['metric']}")
    logger.info(f"[{profile.upper()}] Grace period: {search['grace_period']:,} steps")

    ppo_config = build_sweep_config(profile, train_md, train_fd, eval_md, eval_fd)

    # ASHA with 500K grace period — protects late-blooming agents
    scheduler = ASHAScheduler(
        time_attr="num_env_steps_sampled_lifetime",
        max_t=SWEEP_TOTAL_TIMESTEPS,
        grace_period=search["grace_period"],
        reduction_factor=3,
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=ppo_config,
        tune_config=tune.TuneConfig(
            metric=search["metric"],
            mode="max",
            num_samples=search["num_samples"],
            scheduler=scheduler,
        ),
        run_config=TuneRunConfig(
            name=f"{profile}-sweep-v4",
            verbose=2,
        ),
    )

    results = tuner.fit()

    # Report best
    best_result = results.get_best_result()
    best_config = best_result.config
    best_metrics = best_result.metrics

    logger.info("=" * 70)
    logger.info(f"SWEEP COMPLETE — Best {profile.upper()} Trial:")
    logger.info(f"  LR Schedule:   {best_config.get('lr', 'N/A')}")
    logger.info(f"  Gamma:         {best_config.get('gamma', 'N/A'):.6f}")
    logger.info(f"  Entropy Sched: {best_config.get('entropy_coeff', 'N/A')}")
    logger.info(f"  Clip Param:    {best_config.get('clip_param', 'N/A'):.4f}")
    logger.info(f"  Minibatch:     {best_config.get('minibatch_size', 'N/A')}")
    logger.info(f"  Num Epochs:    {best_config.get('num_epochs', 'N/A')}")
    logger.info(f"  Batch Size:    {best_config.get('train_batch_size_per_learner', 'N/A')}")
    er = best_metrics.get("env_runners", {})
    reward = er.get("episode_return_mean", er.get("episode_reward_mean", "N/A"))
    risk_adj = er.get("risk_adjusted_return", "N/A")
    logger.info(f"  Return:        {reward}")
    logger.info(f"  Risk-Adj Ret:  {risk_adj}")
    logger.info(f"  Win Rate:      {er.get('win_rate', 'N/A')}")
    logger.info(f"  Max Drawdown:  {er.get('max_drawdown', 'N/A')}")
    logger.info("=" * 70)

    ray.shutdown()
    return results


def run_dry_run(profile: str, episode_dir: str = "data/episodes", steps: int = 100):
    """Quick sanity check — 1 trial, minimal steps."""
    logger.info(f"Dry-run mode ({profile}): {steps} steps, 1 trial")

    ray.init(ignore_reinit_error=True, num_cpus=2)

    market_data, feature_data = _load_training_data(
        data_dir="data/datasets",
        episode_dir=episode_dir,
    )

    split_idx = int(len(market_data) * 0.8)
    train_md = market_data[:split_idx]
    train_fd = feature_data[:split_idx]

    env_kwargs = PROFILE_ENV_KWARGS[profile]
    train_env_config = {
        "market_data": train_md,
        "feature_data": train_fd,
        **env_kwargs,
    }

    config = (
        PPOConfig()
        .environment(
            env=ENV_MAP[profile],
            env_config=train_env_config,
        )
        .env_runners(num_env_runners=0)
        .training(
            lr=_make_lr_schedule(3e-4),
            gamma=0.995,
            entropy_coeff=_make_entropy_schedule(0.005),
            minibatch_size=64,
            lambda_=0.95,
            clip_param=0.15,
            num_epochs=6,
            train_batch_size_per_learner=min(steps, 2048),
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "tanh",
            },
        )
        .callbacks(callbacks_class=TradingCallbacks)
    )

    algo = config.build_algo()

    for i in range(max(1, steps // 2048)):
        result = algo.train()
        _er = result.get("env_runners", {})
        reward = _er.get("episode_return_mean", _er.get("episode_reward_mean", 0))
        risk_adj = _er.get("risk_adjusted_return", 0)
        total = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))
        logger.info(f"  Iter {i + 1}: steps={total}, reward={reward:.4f}, risk_adj={risk_adj:.4f}")

    algo.stop()
    ray.shutdown()
    logger.info("Dry-run complete")


def main():
    parser = argparse.ArgumentParser(description="Ray Tune Sweep — Shield & Builder (V4)")
    parser.add_argument(
        "--profile", required=True, choices=["shield", "builder"],
        help="Agent profile: shield (risk-adjusted) or builder (return-max)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Quick sanity check (1 trial)")
    parser.add_argument("--episode-dir", default="data/episodes", help="Cached episode data dir")
    parser.add_argument("--config", default=None, help="Sweep YAML config (optional)")
    parser.add_argument("--dry-run-steps", type=int, default=100, help="Steps for dry-run")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(profile=args.profile, episode_dir=args.episode_dir, steps=args.dry_run_steps)
    else:
        run_sweep(profile=args.profile, config_path=args.config, episode_dir=args.episode_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
