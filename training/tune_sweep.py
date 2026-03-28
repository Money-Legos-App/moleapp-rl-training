"""
Ray Tune Hyperparameter Sweep for Shield PPO (V3)
===================================================
Uses ASHA scheduler for efficient early stopping.
Sweeps learning rate peak, entropy start, gamma, clip_param, and minibatch_size
while keeping LR warmup/decay and entropy schedule structure from V3.

Usage:
    # Full sweep on RunPod (30 trials, 2M steps each, ASHA kills bad ones early)
    python -m training.tune_sweep --config training/configs/tune_shield.yaml

    # Dry-run: test locally (1 trial, ~2K steps)
    python -m training.tune_sweep --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig

from envs import ENV_MAP
from training.callbacks.trading_callbacks import TradingCallbacks
from training.train import _load_training_data, _resolve_num_gpus

logger = logging.getLogger(__name__)

# ── Shield-specific fixed params (not swept) ────────────────────────
SHIELD_ENV_KWARGS = {
    "max_leverage": 1,
    "max_positions": 2,
    "initial_capital": 1000.0,
    "max_sl_pct": 0.03,
    "min_sl_pct": 0.005,
    "max_tp_pct": 0.075,
    "min_tp_pct": 0.01,
    "episode_length": 2880,
    "max_drawdown_pct": 0.10,
}

# Sweep run length (shorter than production 10M — enough to see signal)
SWEEP_TOTAL_TIMESTEPS = 2_000_000
EVAL_INTERVAL = 20


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
    train_market: np.ndarray,
    train_features: list,
    eval_market: np.ndarray,
    eval_features: list,
) -> dict:
    """Build RLlib PPOConfig with Tune search spaces for V3 params."""
    train_env_config = {
        "market_data": train_market,
        "feature_data": train_features,
        **SHIELD_ENV_KWARGS,
    }
    eval_env_config = {
        "market_data": eval_market,
        "feature_data": eval_features,
        **SHIELD_ENV_KWARGS,
    }

    # Pre-sample peak LR and start entropy as grid values,
    # then build schedules from them in each trial.
    # We sweep scalar values and convert to schedules at trial init.
    peak_lrs = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
    start_entropies = [0.002, 0.005, 0.01, 0.02]

    config = (
        PPOConfig()
        .environment(
            env=ENV_MAP["shield"],
            env_config=train_env_config,
        )
        .env_runners(
            num_env_runners=4,      # Fewer runners per trial (multiple trials share GPU)
            num_envs_per_env_runner=2,
        )
        .training(
            lr=tune.choice([_make_lr_schedule(p) for p in peak_lrs]),
            gamma=tune.uniform(0.99, 0.999),
            entropy_coeff=tune.choice([_make_entropy_schedule(e) for e in start_entropies]),
            minibatch_size=tune.choice([128, 256, 512]),
            clip_param=tune.uniform(0.1, 0.25),
            # Fixed params
            lambda_=0.95,
            num_epochs=tune.choice([4, 6, 8]),
            train_batch_size_per_learner=tune.choice([8192, 16384]),
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


def run_sweep(config_path: str | None = None, episode_dir: str = "data/episodes"):
    """Run full Ray Tune hyperparameter sweep."""
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

    logger.info(f"Sweep data: train={len(train_fd)}, eval={len(eval_fd)} timesteps")

    # Build config with search spaces
    ppo_config = build_sweep_config(train_md, train_fd, eval_md, eval_fd)

    # ASHA scheduler — kills bad trials early, keeps promising ones
    scheduler = ASHAScheduler(
        time_attr="num_env_steps_sampled_lifetime",
        max_t=SWEEP_TOTAL_TIMESTEPS,
        grace_period=400_000,   # Min steps before killing
        reduction_factor=3,
    )

    # W&B callback
    callbacks = []
    try:
        try:
            from ray.tune.logger.wandb import WandbLoggerCallback
        except ImportError:
            from ray.air.integrations.wandb import WandbLoggerCallback

        wandb_cfg = {}
        if config_path:
            import yaml
            with open(config_path) as f:
                file_cfg = yaml.safe_load(f)
            wandb_cfg = file_cfg.get("wandb", {})

        callbacks.append(WandbLoggerCallback(
            project=wandb_cfg.get("project", "moleapp-rl"),
            group=wandb_cfg.get("group", "shield-sweep-v3"),
            log_config=True,
        ))
        logger.info("W&B logger enabled — each trial streams to W&B dashboard")
    except ImportError:
        logger.info("W&B not available, skipping WandbLoggerCallback")

    # Run sweep
    tuner = tune.Tuner(
        "PPO",
        param_space=ppo_config,
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            num_samples=20,     # 20 trials (ASHA kills ~60% early)
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            name="shield-sweep-v3",
            callbacks=callbacks,
            stop={"num_env_steps_sampled_lifetime": SWEEP_TOTAL_TIMESTEPS},
            checkpoint_config=ray.train.CheckpointConfig(
                checkpoint_frequency=20,
                num_to_keep=2,
            ),
        ),
    )

    results = tuner.fit()

    # Report best
    best_result = results.get_best_result()
    best_config = best_result.config
    best_metrics = best_result.metrics

    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE — Best Trial:")
    logger.info(f"  LR Schedule:   {best_config.get('lr', 'N/A')}")
    logger.info(f"  Gamma:         {best_config.get('gamma', 'N/A')}")
    logger.info(f"  Entropy Sched: {best_config.get('entropy_coeff', 'N/A')}")
    logger.info(f"  Clip Param:    {best_config.get('clip_param', 'N/A')}")
    logger.info(f"  Minibatch:     {best_config.get('minibatch_size', 'N/A')}")
    logger.info(f"  Num Epochs:    {best_config.get('num_epochs', 'N/A')}")
    logger.info(f"  Batch Size:    {best_config.get('train_batch_size_per_learner', 'N/A')}")
    er = best_metrics.get("env_runners", {})
    reward = er.get("episode_return_mean", er.get("episode_reward_mean", "N/A"))
    logger.info(f"  Reward:        {reward}")
    logger.info("=" * 70)

    ray.shutdown()
    return results


def run_dry_run(episode_dir: str = "data/episodes", steps: int = 100):
    """Local dry-run: quick sanity check without full sweep."""
    logger.info(f"Dry-run mode: {steps} steps, 1 trial")

    ray.init(ignore_reinit_error=True, num_cpus=2)

    market_data, feature_data = _load_training_data(
        data_dir="data/datasets",
        episode_dir=episode_dir,
    )

    split_idx = int(len(market_data) * 0.8)
    train_md = market_data[:split_idx]
    train_fd = feature_data[:split_idx]

    train_env_config = {
        "market_data": train_md,
        "feature_data": train_fd,
        **SHIELD_ENV_KWARGS,
    }

    config = (
        PPOConfig()
        .environment(
            env=ENV_MAP["shield"],
            env_config=train_env_config,
        )
        .env_runners(
            num_env_runners=0,
        )
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
        total = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))
        logger.info(f"  Iter {i + 1}: steps={total}, reward={reward:.4f}")

    algo.stop()
    ray.shutdown()
    logger.info("Dry-run complete")


def main():
    parser = argparse.ArgumentParser(description="Ray Tune Sweep for Shield PPO (V3)")
    parser.add_argument("--dry-run", action="store_true", help="Quick sanity check (1 trial)")
    parser.add_argument("--episode-dir", default="data/episodes", help="Cached episode data dir")
    parser.add_argument("--config", default=None, help="Sweep YAML config (optional)")
    parser.add_argument("--dry-run-steps", type=int, default=100, help="Steps for dry-run")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(episode_dir=args.episode_dir, steps=args.dry_run_steps)
    else:
        run_sweep(config_path=args.config, episode_dir=args.episode_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
