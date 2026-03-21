"""
W&B Sweep Agent for Shield PPO Hyperparameter Optimization
============================================================
Launched by `wandb agent`. Receives hyperparams from W&B, trains a shortened
PPO run (2M steps), and logs Shield-specific trading metrics.

Usage:
    # Normal: launched by W&B sweep controller
    wandb agent dapps4africa/MoleApp-RL/<SWEEP_ID>

    # Dry-run: test locally without W&B (100 steps, random params)
    python training/sweep_agent.py --dry-run

    # Dry-run with specific episode dir
    python training/sweep_agent.py --dry-run --episode-dir data/episodes
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import ShieldTradingEnv
from training.callbacks.sweep_eval_callback import SweepEvalCallback

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

# Fixed PPO params (not swept)
FIXED_PPO = {
    "n_steps": 2048,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# Sweep run length (shorter than production 10M)
SWEEP_TOTAL_TIMESTEPS = 2_000_000
EVAL_FREQ = 100_000
N_EVAL_EPISODES = 10


def load_episode_data(
    episode_dir: str = "data/episodes",
    asset: str = "BTC",
) -> tuple[np.ndarray, list]:
    """
    Load pre-built episode data (market_data + features) from cache.

    Falls back to building on-the-fly if cache doesn't exist.
    """
    ep_path = Path(episode_dir) / asset
    market_path = ep_path / "market_data.npy"
    features_path = ep_path / "features.pkl"

    if market_path.exists() and features_path.exists():
        market_data = np.load(market_path)
        with open(features_path, "rb") as f:
            feature_data = pickle.load(f)
        logger.info(f"Loaded cached episodes for {asset}: {len(feature_data)} timesteps")
        return market_data, feature_data

    # Fall back: build from parquets
    logger.info(f"No cached episodes for {asset}, building from parquets...")
    from data.preprocessors.episode_builder import EpisodeBuilder

    builder = EpisodeBuilder(data_dir="data/datasets")
    market_data, feature_data = builder.build_episodes(asset)
    return market_data, feature_data


def make_env(
    market_data: np.ndarray,
    feature_data: list,
    env_kwargs: dict | None = None,
) -> VecNormalize:
    """Build a VecNormalize-wrapped Shield environment."""
    kwargs = {**SHIELD_ENV_KWARGS, **(env_kwargs or {})}

    def _make():
        env = ShieldTradingEnv(
            market_data=market_data,
            feature_data=feature_data,
            **kwargs,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_make])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    return vec_env


def run_sweep():
    """Main sweep agent entry point — called by `wandb agent`."""
    import wandb

    wandb.init()
    config = wandb.config

    # Extract swept hyperparams
    learning_rate = config.learning_rate
    ent_coef = config.ent_coef
    batch_size = config.batch_size
    gamma = config.gamma

    logger.info(
        f"Sweep run: lr={learning_rate:.2e} ent={ent_coef:.4f} "
        f"batch={batch_size} gamma={gamma:.4f}"
    )

    # Load data (BTC primary, most liquid)
    market_data, feature_data = load_episode_data()

    # Split train/eval (80/20)
    split_idx = int(len(market_data) * 0.8)
    train_md, train_fd = market_data[:split_idx], feature_data[:split_idx]
    eval_md, eval_fd = market_data[split_idx:], feature_data[split_idx:]

    # Build envs
    train_env = make_env(train_md, train_fd)
    eval_env = make_env(eval_md, eval_fd)

    # Build PPO with swept + fixed params
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs={
            "net_arch": [256, 256, 128],
            "activation_fn": nn.Tanh,
        },
        verbose=0,
        **FIXED_PPO,
    )

    # Sweep eval callback
    eval_cb = SweepEvalCallback(
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        initial_capital=SHIELD_ENV_KWARGS["initial_capital"],
        verbose=1,
    )

    # Train
    model.learn(
        total_timesteps=SWEEP_TOTAL_TIMESTEPS,
        callback=[eval_cb],
        progress_bar=False,
    )

    # Log final summary
    final_metrics = eval_cb._evaluate()
    wandb.log(final_metrics, step=SWEEP_TOTAL_TIMESTEPS)
    wandb.summary.update(final_metrics)

    logger.info(
        f"Sweep run complete: Sharpe={final_metrics['eval/sharpe_ratio']:.3f} "
        f"DD={final_metrics['eval/max_drawdown']:.3f}"
    )

    # Cleanup
    train_env.close()
    eval_env.close()
    wandb.finish()


def run_dry_run(episode_dir: str = "data/episodes", steps: int = 100):
    """
    Local dry-run: quick sanity check without W&B.

    Creates env, model, runs a few steps, computes metrics.
    """
    logger.info(f"Dry-run mode: {steps} steps, no W&B")

    # Load data
    market_data, feature_data = load_episode_data(episode_dir=episode_dir)

    split_idx = int(len(market_data) * 0.8)
    train_md, train_fd = market_data[:split_idx], feature_data[:split_idx]
    eval_md, eval_fd = market_data[split_idx:], feature_data[split_idx:]

    # Build envs
    train_env = make_env(train_md, train_fd)
    eval_env = make_env(eval_md, eval_fd)

    # Random hyperparams (mid-range)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        batch_size=512,
        gamma=0.99,
        ent_coef=0.01,
        n_steps=min(steps, 2048),
        policy_kwargs={
            "net_arch": [256, 256, 128],
            "activation_fn": nn.Tanh,
        },
        verbose=0,
    )

    eval_cb = SweepEvalCallback(
        eval_env=eval_env,
        eval_freq=steps,
        n_eval_episodes=2,
        initial_capital=SHIELD_ENV_KWARGS["initial_capital"],
        verbose=1,
    )

    model.learn(total_timesteps=steps, callback=[eval_cb], progress_bar=False)

    # Run one final eval
    metrics = eval_cb._evaluate()
    logger.info("Dry-run metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    train_env.close()
    eval_env.close()
    logger.info("Dry-run complete")


def main():
    parser = argparse.ArgumentParser(description="W&B Sweep Agent for Shield PPO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run locally without W&B (100 steps, random params)",
    )
    parser.add_argument(
        "--episode-dir",
        default="data/episodes",
        help="Directory with cached episode data",
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
        run_sweep()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
