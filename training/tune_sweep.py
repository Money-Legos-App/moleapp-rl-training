"""
Ray Tune Hyperparameter Sweep for Shield PPO
==============================================
Replaces W&B Sweeps with Ray Tune for hyperparameter optimization.
Uses ASHA scheduler (more efficient than Hyperband) and optional W&B logging.

Usage:
    # Full sweep on RunPod
    python -m training.tune_sweep --config training/configs/tune_shield.yaml

    # Dry-run: test locally (1 trial, 100 steps)
    python -m training.tune_sweep --dry-run

    # Dry-run with specific episode dir
    python -m training.tune_sweep --dry-run --episode-dir data/episodes
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.ppo import PPOConfig

from envs import ENV_MAP
from training.callbacks.trading_callbacks import TradingCallbacks

logger = logging.getLogger(__name__)

# ── Shield-specific fixed params (not swept) ────────────────────────
SHIELD_ENV_KWARGS = {
    "max_leverage": 1,
    "max_positions": 2,
    "initial_capital": 1000.0,
    "max_sl_pct": 0.03,
    "min_sl_pct": 0.005,
    "max_tp_pct": 0.06,
    "min_tp_pct": 0.01,
    "episode_length": 2880,  # 30 days of 15-min steps
}

# Sweep run length (shorter than production 10M)
SWEEP_TOTAL_TIMESTEPS = 2_000_000
EVAL_INTERVAL = 10  # evaluate every 10 training iterations


def load_episode_data(
    episode_dir: str = "data/episodes",
    asset: str = "BTC",
) -> tuple[np.ndarray, list]:
    """Load pre-built episode data from cache."""
    ep_path = Path(episode_dir) / asset
    market_path = ep_path / "market_data.npy"
    features_path = ep_path / "features.pkl"

    if market_path.exists() and features_path.exists():
        market_data = np.load(market_path)
        with open(features_path, "rb") as f:
            feature_data = pickle.load(f)
        logger.info(f"Loaded cached episodes for {asset}: {len(feature_data)} timesteps")
        return market_data, feature_data

    logger.info(f"No cached episodes for {asset}, building from parquets...")
    from data.preprocessors.episode_builder import EpisodeBuilder

    builder = EpisodeBuilder(data_dir="data/datasets")
    market_data, feature_data = builder.build_episodes(asset)
    return market_data, feature_data


def build_sweep_config(
    train_market: np.ndarray,
    train_features: list,
    eval_market: np.ndarray,
    eval_features: list,
) -> dict:
    """Build RLlib PPOConfig dict with Tune search spaces."""
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

    config = (
        PPOConfig()
        .environment(
            env=ENV_MAP["shield"],
            env_config=train_env_config,
        )
        .env_runners(
            num_env_runners=2,
            # observation normalization handled by env wrapper
        )
        .training(
            lr=tune.loguniform(1e-5, 1e-3),
            gamma=tune.uniform(0.98, 0.999),
            entropy_coeff=tune.loguniform(0.001, 0.05),
            minibatch_size=tune.choice([256, 512, 1024, 2048]),
            # Fixed params (not swept)
            lambda_=0.95,
            clip_param=0.2,
            num_epochs=10,
            train_batch_size_per_learner=2048,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "tanh",
            },
        )
        .evaluation(
            evaluation_interval=EVAL_INTERVAL,
            evaluation_duration=10,
            evaluation_num_env_runners=1,
            evaluation_config=PPOConfig.overrides(
                env_config=eval_env_config,
                explore=False,
            ),
        )
        .callbacks(callbacks_class=TradingCallbacks)
    )

    return config


def run_sweep(config_path: str | None = None, episode_dir: str = "data/episodes"):
    """Run full Ray Tune hyperparameter sweep."""
    ray.init(ignore_reinit_error=True)

    # Load data
    market_data, feature_data = load_episode_data(episode_dir=episode_dir)

    # Split train/eval (80/20)
    split_idx = int(len(market_data) * 0.8)
    train_md, train_fd = market_data[:split_idx], feature_data[:split_idx]
    eval_md, eval_fd = market_data[split_idx:], feature_data[split_idx:]

    # Build config with search spaces
    ppo_config = build_sweep_config(train_md, train_fd, eval_md, eval_fd)

    # ASHA scheduler — more efficient than Hyperband
    scheduler = ASHAScheduler(
        time_attr="timesteps_total",
        max_t=SWEEP_TOTAL_TIMESTEPS,
        grace_period=500_000,  # min steps before killing
        reduction_factor=3,
    )

    # W&B callback — Ray Tune logs every trial's metrics to W&B automatically
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
            group=wandb_cfg.get("group", "shield-sweep-rllib"),
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
            num_samples=30,
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            name="shield-sweep",
            callbacks=callbacks,
            stop={"timesteps_total": SWEEP_TOTAL_TIMESTEPS},
            checkpoint_config=ray.train.CheckpointConfig(
                checkpoint_frequency=10,
                num_to_keep=3,
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
    logger.info(f"  LR:        {best_config.get('lr', 'N/A')}")
    logger.info(f"  Gamma:     {best_config.get('gamma', 'N/A')}")
    logger.info(f"  Ent Coeff: {best_config.get('entropy_coeff', 'N/A')}")
    logger.info(f"  Batch:     {best_config.get('minibatch_size', 'N/A')}")
    er = best_metrics.get("env_runners", {})
    logger.info(f"  Reward:    {er.get('episode_return_mean', er.get('episode_reward_mean', 'N/A'))}")
    logger.info("=" * 70)

    ray.shutdown()
    return results


def run_dry_run(episode_dir: str = "data/episodes", steps: int = 100):
    """Local dry-run: quick sanity check without full sweep."""
    logger.info(f"Dry-run mode: {steps} steps, 1 trial")

    ray.init(ignore_reinit_error=True, num_cpus=2)

    # Load data
    market_data, feature_data = load_episode_data(episode_dir=episode_dir)

    split_idx = int(len(market_data) * 0.8)
    train_md, train_fd = market_data[:split_idx], feature_data[:split_idx]
    eval_md, eval_fd = market_data[split_idx:], feature_data[split_idx:]

    train_env_config = {
        "market_data": train_md,
        "feature_data": train_fd,
        **SHIELD_ENV_KWARGS,
    }

    # Fixed config (no search space for dry run)
    config = (
        PPOConfig()
        .environment(
            env=ENV_MAP["shield"],
            env_config=train_env_config,
        )
        .env_runners(
            num_env_runners=0,  # local worker only
            # observation normalization handled by env wrapper
        )
        .training(
            lr=1e-4,
            gamma=0.99,
            entropy_coeff=0.01,
            minibatch_size=64,
            lambda_=0.95,
            clip_param=0.2,
            num_epochs=10,
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
        total = result.get("timesteps_total", result.get("num_env_steps_sampled_lifetime", 0))
        logger.info(f"  Iter {i + 1}: steps={total}, reward={reward:.4f}")

    algo.stop()
    ray.shutdown()
    logger.info("Dry-run complete")


def main():
    parser = argparse.ArgumentParser(description="Ray Tune Sweep for Shield PPO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run locally without full sweep (1 trial, 100 steps)",
    )
    parser.add_argument(
        "--episode-dir",
        default="data/episodes",
        help="Directory with cached episode data",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to sweep YAML config (optional)",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=100,
        help="Number of steps for dry-run",
    )
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
