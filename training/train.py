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
    output_dir: str = "models",
    resume_from: str | None = None,
):
    """
    Train a PPO model for the specified risk profile.

    Args:
        profile: "shield", "builder", or "hunter"
        config_path: Path to YAML config file
        data_dir: Directory containing .parquet market data files
        output_dir: Directory for model artifacts
        resume_from: Path to checkpoint to resume from (optional)
    """
    config = load_config(config_path)
    output_path = Path(output_dir) / profile
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {profile} model with config: {config_path}")
    logger.info(f"Output directory: {output_path}")

    # Load market data
    market_data, feature_data = _load_training_data(data_dir)
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


def _load_training_data(data_dir: str):
    """
    Load and merge market data from parquet files.

    Returns:
        market_data: np.ndarray of shape (timesteps, 6) — OHLCV + funding
        feature_data: list of MarketFeatures (one per timestep)
    """
    import pandas as pd
    from data.preprocessors.feature_engineer import MarketFeatures

    data_path = Path(data_dir)
    # For now, load BTC as the primary training asset
    # TODO: Multi-asset training with asset rotation
    btc_file = list(data_path.glob("BTC_1h_*.parquet"))
    if not btc_file:
        raise FileNotFoundError(f"No BTC data found in {data_dir}")

    df = pd.read_parquet(btc_file[0])
    logger.info(f"Loaded {len(df)} candles from {btc_file[0]}")

    # Build market_data array: open, high, low, close, volume, funding_rate
    market_data = np.zeros((len(df), 6), dtype=np.float32)
    market_data[:, 0] = df["open"].values
    market_data[:, 1] = df["high"].values
    market_data[:, 2] = df["low"].values
    market_data[:, 3] = df["close"].values
    market_data[:, 4] = df["volume"].values
    # funding_rate column 5 defaults to 0 if not available

    # Build MarketFeatures list (simplified — full version computes all 47 features)
    feature_data = []
    for i in range(len(df)):
        # Compute rolling stats with sufficient lookback
        lookback_30d = max(0, i - 720)  # 720 hours = 30 days
        close_prices = market_data[lookback_30d:i+1, 3]
        volumes = market_data[lookback_30d:i+1, 4]

        rolling_mean = float(np.mean(close_prices)) if len(close_prices) > 0 else market_data[i, 3]
        rolling_vol = float(np.mean(volumes)) if len(volumes) > 0 else market_data[i, 4]

        features = MarketFeatures(
            price=float(market_data[i, 3]),
            price_1h_ago=float(market_data[max(0, i-1), 3]),
            price_4h_ago=float(market_data[max(0, i-4), 3]),
            price_24h_ago=float(market_data[max(0, i-24), 3]),
            vwap_24h=float(np.mean(market_data[max(0, i-24):i+1, 3])),
            rolling_mean_30d=rolling_mean,
            volume_24h=float(np.sum(market_data[max(0, i-24):i+1, 4])),
            rolling_avg_vol_30d=rolling_vol,
            bid_imbalance_pct=0.0,  # Not available in historical candles
            spread_bps=1.0,  # Assume 1 bps spread
            open_interest=0.0,  # Load from OI data if available
            rolling_avg_oi_30d=0.0,
            oi_1h_ago=0.0,
            oi_4h_ago=0.0,
            funding_rate=float(market_data[i, 5]),
            funding_8h_cumulative=float(np.sum(market_data[max(0, i-8):i+1, 5])),
            prev_funding_rate=float(market_data[max(0, i-1), 5]),
            rsi_1h=50.0,  # TODO: Compute from ta library
            rsi_4h=50.0,
            macd_hist_1h=0.0,
            bb_position_1h=0.5,
            atr_1h=float(np.std(close_prices[-24:])) if len(close_prices) >= 24 else 0.0,
            ema_20=float(np.mean(close_prices[-20:])) if len(close_prices) >= 20 else rolling_mean,
            ema_50=float(np.mean(close_prices[-50:])) if len(close_prices) >= 50 else rolling_mean,
            ema_200=float(np.mean(close_prices[-200:])) if len(close_prices) >= 200 else rolling_mean,
            sma_4h=float(np.mean(close_prices[-4:])) if len(close_prices) >= 4 else rolling_mean,
            volume_trend_1h=0.0,
            roc_1h=0.0,
            roc_4h=0.0,
            account_value=1000.0,  # Overwritten by env
            initial_capital=1000.0,
            peak_account_value=1000.0,
            open_position_count=0,
            max_positions=5,
            margin_utilization=0.0,
            unrealized_pnl=0.0,
            mission_start_timestamp=0.0,
            current_timestamp=float(df.iloc[i]["timestamp"].timestamp()) if hasattr(df.iloc[i]["timestamp"], "timestamp") else 0.0,
            days_since_last_trade=0.0,
            has_open_position_this_asset=False,
            existing_direction=0,
            btc_dominance=50.0,
            fear_greed_index=50.0,
            market_regime=0,
            cross_asset_momentum=0.0,
        )
        feature_data.append(features)

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
        help="Directory containing market data",
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
        output_dir=args.output_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
