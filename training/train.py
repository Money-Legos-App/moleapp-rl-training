"""
Unified Training Script for Shield, Builder, and Hunter Models
================================================================
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
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import BuilderTradingEnv, HunterTradingEnv, ShieldTradingEnv

logger = logging.getLogger(__name__)

ENV_MAP = {
    "shield": ShieldTradingEnv,
    "builder": BuilderTradingEnv,
    "hunter": HunterTradingEnv,
}


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_env(profile: str, config: dict, market_data: np.ndarray, feature_data: list):
    """Build a VecNormalize-wrapped environment."""
    env_cls = ENV_MAP[profile]
    env_kwargs = config.get("env_kwargs", {})

    def make_env():
        env = env_cls(
            market_data=market_data,
            feature_data=feature_data,
            **env_kwargs,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    if config.get("normalize_observations", True):
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=config.get("normalize_rewards", True),
            clip_obs=config.get("norm_obs_clip", 10.0),
            clip_reward=config.get("norm_reward_clip", 10.0),
        )

    return vec_env


def train(
    profile: str,
    config_path: str,
    data_dir: str = "data/datasets",
    episode_dir: str = "data/episodes",
    output_dir: str = "models",
    resume_from: str | None = None,
):
    """
    Train a PPO model for the specified risk profile.

    Args:
        profile: "shield", "builder", or "hunter"
        config_path: Path to YAML config file
        data_dir: Directory containing .parquet market data files
        episode_dir: Directory with cached EpisodeBuilder output
        output_dir: Directory for model artifacts
        resume_from: Path to checkpoint to resume from (optional)
    """
    config = load_config(config_path)
    output_path = Path(output_dir) / profile
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {profile} model with config: {config_path}")
    logger.info(f"Output directory: {output_path}")

    # Load market data
    market_data, feature_data = _load_training_data(data_dir, episode_dir)
    logger.info(f"Loaded {len(market_data)} timesteps of market data")

    # Split train/eval (80/20)
    split_idx = int(len(market_data) * 0.8)
    train_market = market_data[:split_idx]
    train_features = feature_data[:split_idx]
    eval_market = market_data[split_idx:]
    eval_features = feature_data[split_idx:]

    # Build environments
    train_env = build_env(profile, config, train_market, train_features)
    eval_env = build_env(profile, config, eval_market, eval_features)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 200000),
        save_path=str(output_path / "checkpoints"),
        name_prefix=profile,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best"),
        log_path=str(output_path / "eval_logs"),
        eval_freq=config.get("eval_freq", 50000),
        n_eval_episodes=config.get("n_eval_episodes", 20),
        deterministic=config.get("deterministic_eval", True),
    )

    # Build or load model
    policy_kwargs = config.get("policy_kwargs", {})
    # Convert activation_fn string to actual function
    if "activation_fn" in policy_kwargs:
        import torch.nn as nn
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
        policy_kwargs["activation_fn"] = activation_map.get(
            policy_kwargs["activation_fn"], nn.Tanh
        )

    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=train_env)
    else:
        model = PPO(
            policy=config.get("policy", "MlpPolicy"),
            env=train_env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.01),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(output_path / "tb_logs"),
        )

    # Optional: W&B integration
    try:
        import wandb
        wandb_config = {
            "project": config.get("wandb_project", "moleapp-rl"),
            "name": config.get("wandb_run_name", f"{profile}-v1"),
            "tags": config.get("wandb_tags", [profile]),
            "config": config,
        }
        wandb.init(**wandb_config)
        logger.info("W&B initialized")
    except ImportError:
        logger.info("W&B not installed, skipping")

    # Train
    total_timesteps = config.get("total_timesteps", 10_000_000)
    logger.info(f"Starting training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model
    final_path = output_path / f"{profile}_final"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Save VecNormalize statistics (CRITICAL for production inference)
    if isinstance(train_env, VecNormalize):
        vecnorm_path = output_path / f"{profile}_vecnorm.pkl"
        train_env.save(str(vecnorm_path))
        logger.info(f"Saved VecNormalize stats to {vecnorm_path}")

        # Also export as raw numpy for production use
        vecnorm_data = {
            "obs_rms_mean": train_env.obs_rms.mean.copy(),
            "obs_rms_var": train_env.obs_rms.var.copy(),
            "obs_rms_count": train_env.obs_rms.count,
            "clip_obs": train_env.clip_obs,
        }
        raw_stats_path = output_path / f"{profile}_norm_stats.pkl"
        with open(raw_stats_path, "wb") as f:
            pickle.dump(vecnorm_data, f)
        logger.info(f"Saved raw norm stats to {raw_stats_path}")

    logger.info(f"Training complete for {profile}")
    return model


def _load_training_data(data_dir: str, episode_dir: str = "data/episodes"):
    """
    Load market data and features for training.

    Prefers cached EpisodeBuilder output (full 47-dim features with proper
    technical indicators). Falls back to simplified inline computation
    if episodes haven't been built yet.

    Returns:
        market_data: np.ndarray of shape (timesteps, 7) — OHLCV + funding + OI
        feature_data: list of MarketFeatures (one per timestep)
    """
    import pickle

    # Try cached EpisodeBuilder output first
    ep_path = Path(episode_dir) / "BTC"
    market_path = ep_path / "market_data.npy"
    features_path = ep_path / "features.pkl"

    if market_path.exists() and features_path.exists():
        market_data = np.load(market_path)
        with open(features_path, "rb") as f:
            feature_data = pickle.load(f)
        logger.info(
            f"Loaded cached episodes for BTC: {len(feature_data)} timesteps, "
            f"market_data shape={market_data.shape}"
        )
        return market_data, feature_data

    # Fall back: build from parquets using EpisodeBuilder
    logger.info("No cached episodes found, building from parquets...")
    from data.preprocessors.episode_builder import EpisodeBuilder

    builder = EpisodeBuilder(data_dir=data_dir)
    market_data, feature_data = builder.build_episodes("BTC")
    logger.info(f"Built {len(feature_data)} timesteps from {data_dir}")
    return market_data, feature_data


def main():
    parser = argparse.ArgumentParser(description="Train MoleApp RL models")
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

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    train(
        profile=args.profile,
        config_path=args.config,
        data_dir=args.data_dir,
        episode_dir=args.episode_dir,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
