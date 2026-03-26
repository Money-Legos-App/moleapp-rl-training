"""
Unified Training Script for Shield, Builder, and Hunter Models (RLlib)
=======================================================================
Usage:
    python -m training.train --profile shield --config training/configs/shield_config.yaml
    python -m training.train --profile builder --config training/configs/builder_config.yaml
    python -m training.train --profile hunter --config training/configs/hunter_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import ray
import torch
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

from envs import ENV_MAP

logger = logging.getLogger(__name__)


def _resolve_num_gpus(config_val) -> int:
    """Resolve num_gpus from config — supports 'auto', int, or 0."""
    if config_val == "auto":
        n = torch.cuda.device_count()
        logger.info(f"GPU auto-detect: {n} GPU(s) found")
        return min(n, 1)  # Use 1 GPU for single-learner PPO
    return int(config_val)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_training_data(
    data_dir: str,
    episode_dir: str = "data/episodes",
    assets: list[str] | None = None,
):
    """
    Load market data and features for training.

    When multiple assets are specified, their episodes are concatenated
    sequentially. Each asset's data is independent (the env resets between
    asset boundaries via episode_length).

    Args:
        data_dir: Directory with .parquet files
        episode_dir: Directory with cached .npy/.pkl episodes
        assets: List of asset names to load (default: all 15 MoleApp assets)

    Returns:
        market_data: np.ndarray of shape (total_timesteps, 7)
        feature_data: list of MarketFeatures
    """
    if assets is None:
        from data.collectors.asset_config import ALLOWED_ASSETS
        assets = ALLOWED_ASSETS  # All 15 supported assets

    all_market = []
    all_features = []

    for asset in assets:
        ep_path = Path(episode_dir) / asset
        market_path = ep_path / "market_data.npy"
        features_path = ep_path / "features.pkl"

        if market_path.exists() and features_path.exists():
            m = np.load(market_path)
            with open(features_path, "rb") as f:
                feat = pickle.load(f)
            logger.info(f"Loaded {asset}: {len(feat)} timesteps, shape={m.shape}")
        else:
            logger.info(f"No cached episodes for {asset}, building from parquets...")
            from data.preprocessors.episode_builder import EpisodeBuilder
            builder = EpisodeBuilder(data_dir=data_dir)
            m, feat = builder.build_episodes(asset)
            logger.info(f"Built {asset}: {len(feat)} timesteps")

        all_market.append(m)
        all_features.extend(feat)

    market_data = np.concatenate(all_market, axis=0)
    logger.info(
        f"Total training data: {len(assets)} asset(s), "
        f"{len(all_features)} timesteps, shape={market_data.shape}"
    )
    return market_data, all_features


def build_ppo_config(profile: str, config: dict, env_config: dict) -> PPOConfig:
    """Build an RLlib PPOConfig from YAML config dict."""
    env_name = ENV_MAP[profile]

    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=env_name,
            env_config=env_config,
        )
        .env_runners(
            num_env_runners=config.get("num_env_runners", 4),
            num_envs_per_env_runner=config.get("num_envs_per_env_runner", 1),
            observation_filter=config.get("observation_filter", "MeanStdFilter"),
        )
        .training(
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            lambda_=config.get("lambda_", 0.95),
            clip_param=config.get("clip_param", 0.2),
            num_epochs=config.get("num_epochs", 10),
            minibatch_size=config.get("minibatch_size", 64),
            train_batch_size_per_learner=config.get("train_batch_size_per_learner", 2048),
            entropy_coeff=config.get("entropy_coeff", 0.01),
            vf_loss_coeff=config.get("vf_loss_coeff", 0.5),
            grad_clip=config.get("grad_clip", 0.5),
            model=config.get("model", {
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "tanh",
            }),
        )
        .evaluation(
            evaluation_interval=config.get("evaluation", {}).get("evaluation_interval", 25),
            evaluation_duration=config.get("evaluation", {}).get("evaluation_duration", 20),
            evaluation_num_env_runners=config.get("evaluation", {}).get("num_env_runners", 1),
            evaluation_config=PPOConfig.overrides(explore=False),
        )
        .callbacks(
            callbacks_class=_get_callbacks_class(),
        )
    )

    # GPU support (auto-detect or explicit) — old stack uses resources()
    num_gpus = _resolve_num_gpus(config.get("num_gpus", 0))
    if num_gpus > 0:
        ppo_config.resources(num_gpus=num_gpus)

    return ppo_config


def _get_callbacks_class():
    """Import callbacks lazily to avoid circular imports."""
    from training.callbacks.trading_callbacks import TradingCallbacks
    return TradingCallbacks


def train(
    profile: str,
    config_path: str,
    data_dir: str = "data/datasets",
    episode_dir: str = "data/episodes",
    output_dir: str = "models",
    resume_from: str | None = None,
    assets: list[str] | None = None,
):
    """
    Train a PPO model for the specified risk profile using RLlib.

    Args:
        profile: "shield", "builder", or "hunter"
        config_path: Path to YAML config file
        data_dir: Directory containing .parquet market data files
        episode_dir: Directory with cached EpisodeBuilder output
        output_dir: Directory for model artifacts
        resume_from: Path to checkpoint to resume from (optional)
        assets: Assets to train on (default: all 15 supported assets)
    """
    config = load_config(config_path)
    output_path = (Path(output_dir) / profile).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {profile} model with config: {config_path}")

    # Load market data
    market_data, feature_data = _load_training_data(data_dir, episode_dir, assets=assets)
    logger.info(f"Loaded {len(market_data)} timesteps of market data")

    # Split train/eval (80/20)
    split_idx = int(len(market_data) * 0.8)
    train_market = market_data[:split_idx]
    train_features = feature_data[:split_idx]

    # Build env_config with market data + env_kwargs
    env_kwargs = config.get("env_config", config.get("env_kwargs", {}))
    env_config = {
        "market_data": train_market,
        "feature_data": train_features,
        **env_kwargs,
    }

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Build algorithm
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        algo = Algorithm.from_checkpoint(resume_from)
    else:
        ppo_config = build_ppo_config(profile, config, env_config)

        # Set eval env config with eval data
        eval_market = market_data[split_idx:]
        eval_features = feature_data[split_idx:]
        eval_env_config = {
            "market_data": eval_market,
            "feature_data": eval_features,
            **env_kwargs,
        }
        ppo_config.evaluation(evaluation_config=PPOConfig.overrides(
            env_config=eval_env_config,
            explore=False,
        ))

        algo = ppo_config.build()

    # W&B integration
    wandb_run = None
    try:
        import wandb
        wandb_cfg = config.get("wandb", {})
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "moleapp-rl"),
            name=wandb_cfg.get("run_name", f"{profile}-rllib"),
            tags=wandb_cfg.get("tags", [profile, "ppo", "rllib"]),
            config=config,
        )
        logger.info("W&B initialized")
    except ImportError:
        logger.info("W&B not installed, skipping")

    # Training loop
    total_timesteps = config.get("total_timesteps", 10_000_000)
    checkpoint_freq = config.get("checkpoint_freq", 10)  # iterations
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {total_timesteps} timesteps...")

    iteration = 0
    total_steps = 0

    while total_steps < total_timesteps:
        result = algo.train()
        iteration += 1
        total_steps = result.get("timesteps_total", result.get("num_env_steps_sampled_lifetime", 0))

        # Log to W&B
        if wandb_run:
            log_data = {
                "train/timesteps": total_steps,
                "train/episode_reward_mean": result.get("episode_reward_mean", result.get("env_runners", {}).get("episode_reward_mean", 0)),
                "train/episode_len_mean": result.get("episode_len_mean", result.get("env_runners", {}).get("episode_len_mean", 0)),
            }

            # Learner stats (old stack uses info.learner.default_policy)
            learner_info = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
            if learner_info:
                log_data["train/policy_loss"] = learner_info.get("policy_loss", 0)
                log_data["train/vf_loss"] = learner_info.get("vf_loss", 0)
                log_data["train/entropy"] = learner_info.get("entropy", 0)

            # Trading-specific metrics from TradingCallbacks (old stack: custom_metrics)
            custom_metrics = result.get("custom_metrics", {})
            for metric_key in ("total_return", "max_drawdown", "win_rate", "total_trades", "total_pnl"):
                val = custom_metrics.get(f"{metric_key}_mean", None)
                if val is not None:
                    log_data[f"train/{metric_key}"] = val

            # Eval metrics
            eval_results = result.get("evaluation", {})
            if eval_results:
                log_data["eval/episode_reward_mean"] = eval_results.get("episode_reward_mean", 0)
                eval_custom = eval_results.get("custom_metrics", {})
                for metric_key in ("total_return", "max_drawdown", "win_rate", "total_trades", "total_pnl"):
                    val = eval_custom.get(f"{metric_key}_mean", None)
                    if val is not None:
                        log_data[f"eval/{metric_key}"] = val

            wandb.log(log_data, step=int(total_steps))

        # Console progress
        if iteration % 5 == 0:
            reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
            logger.info(
                f"[Iter {iteration}] steps={total_steps:,} "
                f"reward={reward_mean:.2f}"
            )

        # Checkpoint
        if iteration % checkpoint_freq == 0:
            ckpt_path = algo.save_to_path(str(checkpoint_dir / f"checkpoint_{iteration:06d}"))
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = str(output_path / f"{profile}_final")
    algo.save_to_path(final_path)
    logger.info(f"Saved final model to {final_path}")

    # Extract and save observation normalization stats (MeanStdFilter)
    _save_norm_stats(algo, output_path, profile)

    # Cleanup
    if wandb_run:
        wandb.finish()
    algo.stop()

    logger.info(f"Training complete for {profile}")
    return algo


def _save_norm_stats(algo, output_path: Path, profile: str):
    """Extract MeanStdFilter stats and save in production-compatible format."""
    try:
        # Old API stack uses workers.local_worker().filters
        filters = algo.workers.local_worker().filters
        if "default_policy" in filters:
            obs_filter = filters["default_policy"]
            norm_stats = {
                "obs_rms_mean": np.array(obs_filter.rs.mean, dtype=np.float64),
                "obs_rms_var": np.array(obs_filter.rs.var, dtype=np.float64),
                "obs_rms_count": float(obs_filter.rs.count),
                "clip_obs": 10.0,
            }
            stats_path = output_path / f"{profile}_norm_stats.pkl"
            with open(stats_path, "wb") as f:
                pickle.dump(norm_stats, f)
            logger.info(f"Saved norm stats to {stats_path}")
    except Exception as e:
        logger.warning(f"Could not extract norm stats: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train MoleApp RL models (RLlib)")
    parser.add_argument(
        "--profile",
        required=True,
        choices=["shield", "builder", "hunter"],
        help="Risk profile to train",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data-dir",
        default="data/datasets",
        help="Directory containing market data parquets",
    )
    parser.add_argument(
        "--episode-dir",
        default="data/episodes",
        help="Directory with cached EpisodeBuilder output",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--assets",
        nargs="*",
        default=None,
        help="Assets to train on (default: all 15). Use --assets BTC ETH to limit.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    train(
        profile=args.profile,
        config_path=args.config,
        data_dir=args.data_dir,
        episode_dir=args.episode_dir,
        output_dir=args.output_dir,
        resume_from=args.resume,
        assets=args.assets,
    )


if __name__ == "__main__":
    main()
