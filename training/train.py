"""
Unified Training Script for Shield and Builder Models (RLlib)
==============================================================
Uses the new API stack (RLModule + Learner + ConnectorV2 + EnvRunner).

2-Strategy System:
  Shield  — "Flexible USD Vault" (1x leverage, 25% max position, never pay funding)
  Builder — "High-Yield Engine"  (2x leverage, 50% max position, block funding >0.03%)

Usage:
    python -m training.train --profile shield --config training/configs/shield_config.yaml
    python -m training.train --profile builder --config training/configs/builder_config.yaml
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
    """Build an RLlib PPOConfig from YAML config dict (new API stack)."""
    env_name = ENV_MAP[profile]

    ppo_config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config=env_config,
        )
        .env_runners(
            num_env_runners=config.get("num_env_runners", 4),
            num_envs_per_env_runner=config.get("num_envs_per_env_runner", 1),
        )
        .training(
            # LR: schedule (list of [timestep, value]) or flat float
            lr=config.get("lr_schedule", config.get("lr", 3e-4)),
            gamma=config.get("gamma", 0.99),
            lambda_=config.get("lambda_", 0.95),
            clip_param=config.get("clip_param", 0.2),
            num_epochs=config.get("num_epochs", 10),
            minibatch_size=config.get("minibatch_size", 256),
            train_batch_size_per_learner=config.get("train_batch_size_per_learner", 2048),
            # Entropy: schedule (list of [timestep, value]) or flat float
            entropy_coeff=config.get("entropy_coeff_schedule", config.get("entropy_coeff", 0.01)),
            vf_loss_coeff=config.get("vf_loss_coeff", 0.5),
            grad_clip=config.get("grad_clip", 0.5),
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": config.get("model", {}).get("fcnet_hiddens", [256, 256, 128]),
                "fcnet_activation": config.get("model", {}).get("fcnet_activation", "tanh"),
            },
        )
        .evaluation(
            evaluation_interval=config.get("evaluation", {}).get("evaluation_interval", 25),
            evaluation_duration=config.get("evaluation", {}).get("evaluation_duration", 5),
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=config.get("evaluation", {}).get("num_env_runners", 1),
            evaluation_sample_timeout_s=600,  # 10 min timeout for long episodes
            evaluation_force_reset_envs_before_iteration=False,
            evaluation_parallel_to_training=config.get("evaluation", {}).get(
                "evaluation_parallel_to_training", False
            ),
            evaluation_config=PPOConfig.overrides(explore=False),
        )
        .callbacks(
            callbacks_class=_get_callbacks_class(),
        )
    )

    # GPU support — new stack uses .learners()
    num_gpus = _resolve_num_gpus(config.get("num_gpus", 0))
    if num_gpus > 0:
        ppo_config.learners(
            num_learners=config.get("num_learners", 1),
            num_gpus_per_learner=config.get("num_gpus_per_learner", 1),
        )

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
        profile: "shield" or "builder"
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

    # Split train/eval (80/20) and save to shared files
    # (avoids serializing 200K+ MarketFeatures through Ray's object store)
    split_idx = int(len(market_data) * 0.8)
    train_market = market_data[:split_idx]
    train_features = feature_data[:split_idx]
    eval_market = market_data[split_idx:]
    eval_features = feature_data[split_idx:]

    cache_dir = Path("/tmp/rl_training_data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.save(cache_dir / "train_market.npy", train_market)
    with open(cache_dir / "train_features.pkl", "wb") as f:
        pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(cache_dir / "eval_market.npy", eval_market)
    with open(cache_dir / "eval_features.pkl", "wb") as f:
        pickle.dump(eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved training data to {cache_dir} (train={len(train_features)}, eval={len(eval_features)})")

    # Build env_config with file paths instead of raw data
    env_kwargs = config.get("env_config", config.get("env_kwargs", {}))
    env_config = {
        "market_data_path": str(cache_dir / "train_market.npy"),
        "feature_data_path": str(cache_dir / "train_features.pkl"),
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

        # Set eval env config with eval data file paths
        eval_env_config = {
            "market_data_path": str(cache_dir / "eval_market.npy"),
            "feature_data_path": str(cache_dir / "eval_features.pkl"),
            **env_kwargs,
        }
        ppo_config.evaluation(evaluation_config=PPOConfig.overrides(
            env_config=eval_env_config,
            explore=False,
        ))

        algo = ppo_config.build_algo()

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
        total_steps = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))

        # Log to W&B
        if wandb_run:
            env_runners = result.get("env_runners", {})
            log_data = {
                "train/timesteps": total_steps,
                "train/episode_reward_mean": env_runners.get("episode_reward_mean", 0),
                "train/episode_len_mean": env_runners.get("episode_len_mean", 0),
            }

            # Learner stats (new stack: result["learners"]["default_policy"])
            learner_info = result.get("learners", {}).get("default_policy", {})
            if learner_info:
                log_data["train/policy_loss"] = learner_info.get("policy_loss", 0)
                log_data["train/vf_loss"] = learner_info.get("vf_loss", 0)
                log_data["train/entropy"] = learner_info.get("entropy", 0)
                log_data["train/total_loss"] = learner_info.get("total_loss", 0)

            # Trading-specific metrics from TradingCallbacks (new stack: env_runners)
            for metric_key in ("total_return", "max_drawdown", "win_rate", "total_trades", "total_pnl"):
                val = env_runners.get(metric_key, None)
                if val is not None:
                    log_data[f"train/{metric_key}"] = val

            # Eval metrics
            eval_results = result.get("evaluation", {})
            if eval_results:
                eval_runners = eval_results.get("env_runners", {})
                log_data["eval/episode_reward_mean"] = eval_runners.get("episode_reward_mean", 0)
                for metric_key in ("total_return", "max_drawdown", "win_rate", "total_trades", "total_pnl"):
                    val = eval_runners.get(metric_key, None)
                    if val is not None:
                        log_data[f"eval/{metric_key}"] = val

            wandb.log(log_data, step=int(total_steps))

        # Console progress
        if iteration % 5 == 0:
            env_runners = result.get("env_runners", {})
            reward_mean = env_runners.get("episode_reward_mean", 0)
            win_rate = env_runners.get("win_rate", 0)
            total_return = env_runners.get("total_return", 0)
            logger.info(
                f"[Iter {iteration}] steps={total_steps:,} "
                f"reward={reward_mean:.2f} win_rate={win_rate:.2f} return={total_return:.3f}"
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
    """
    Extract observation normalization stats and save for production inference.

    New API stack: MeanStdFilter is no longer available as an env_runner option.
    Instead, we collect running obs stats from the env's observation wrapper
    or compute them from a sample of observations via the env runner.
    """
    try:
        # Try to get stats from env runner's env (if using NormalizeObservation wrapper)
        env_runner = algo.env_runner
        env = env_runner.env
        if hasattr(env, "obs_rms"):
            # gym.wrappers.NormalizeObservation stores running stats
            norm_stats = {
                "obs_rms_mean": np.array(env.obs_rms.mean, dtype=np.float64),
                "obs_rms_var": np.array(env.obs_rms.var, dtype=np.float64),
                "obs_rms_count": float(env.obs_rms.count),
                "clip_obs": 10.0,
            }
        else:
            # Fallback: sample observations to compute stats
            logger.info("No obs_rms found, computing norm stats from training data...")
            sample_obs = []
            test_env = env_runner.env
            obs, _ = test_env.reset()
            sample_obs.append(obs)
            for _ in range(min(1000, len(test_env.market_data) - 1)):
                action = test_env.action_space.sample()
                obs, _, terminated, truncated, _ = test_env.step(action)
                sample_obs.append(obs)
                if terminated or truncated:
                    obs, _ = test_env.reset()
                    sample_obs.append(obs)
            obs_arr = np.array(sample_obs)
            norm_stats = {
                "obs_rms_mean": obs_arr.mean(axis=0).astype(np.float64),
                "obs_rms_var": obs_arr.var(axis=0).astype(np.float64),
                "obs_rms_count": float(len(obs_arr)),
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
        choices=["shield", "builder"],
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
