"""
Sweep Evaluation Callback
==========================
Custom SB3 callback that runs deterministic eval episodes and logs
trading-specific metrics (Sharpe, drawdown, win rate) to W&B.

Used by sweep_agent.py during hyperparameter sweeps.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


def compute_sharpe(
    episode_returns: list[float],
    annualize_factor: float = 3.49,  # sqrt(365/30) for 30-day episodes
) -> float:
    """
    Compute annualized Sharpe ratio from episode returns.

    Args:
        episode_returns: List of total returns (e.g., PnL / initial_capital)
        annualize_factor: sqrt(periods_per_year). Default = sqrt(365/30) ≈ 3.49

    Returns:
        Sharpe ratio (0.0 if insufficient data or zero variance)
    """
    if len(episode_returns) < 2:
        return 0.0
    arr = np.array(episode_returns)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float(arr.mean() / std * annualize_factor)


class SweepEvalCallback(BaseCallback):
    """
    Evaluate the agent every `eval_freq` steps and log trading metrics.

    Runs `n_eval_episodes` deterministic rollouts on the eval environment,
    extracts metrics from env info dicts, computes Sharpe ratio, and logs
    everything to W&B (if available) and stdout.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 100_000,
        n_eval_episodes: int = 10,
        initial_capital: float = 1000.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.initial_capital = initial_capital
        self._wandb = None

    def _init_callback(self) -> None:
        try:
            import wandb
            if wandb.run is not None:
                self._wandb = wandb
        except ImportError:
            pass

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = self._evaluate()
        self._log_metrics(metrics)
        return True

    def _evaluate(self) -> dict[str, float]:
        """Run eval episodes and compute aggregate metrics."""
        episode_returns = []
        episode_drawdowns = []
        total_trades = 0
        winning_trades = 0
        episode_rewards = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                ep_reward += float(reward[0])
                done = dones[0]

                if done:
                    info = infos[0]
                    # Extract terminal info (SB3 wraps it for VecEnv)
                    if "terminal_info" in info:
                        info = info["terminal_info"]
                    elif "episode" in info:
                        # Monitor wrapper
                        pass

                    pnl = info.get("total_pnl", 0.0)
                    ret = pnl / max(self.initial_capital, 1.0)
                    episode_returns.append(ret)
                    episode_drawdowns.append(info.get("max_drawdown", 0.0))
                    total_trades += info.get("total_trades", 0)
                    winning_trades += info.get("winning_trades", 0)

            episode_rewards.append(ep_reward)

        # Compute aggregate metrics
        sharpe = compute_sharpe(episode_returns)
        max_dd = max(episode_drawdowns) if episode_drawdowns else 0.0
        win_rate = winning_trades / max(total_trades, 1)
        avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        avg_pnl_per_trade = (
            sum(episode_returns) * self.initial_capital / max(total_trades, 1)
        )

        return {
            "eval/sharpe_ratio": sharpe,
            "eval/max_drawdown": max_dd,
            "eval/win_rate": win_rate,
            "eval/total_pnl": avg_return * self.initial_capital,
            "eval/avg_trade_pnl": avg_pnl_per_trade,
            "eval/num_trades": total_trades / max(self.n_eval_episodes, 1),
            "eval/episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "eval/avg_return": avg_return,
            "train/timesteps": self.num_timesteps,
        }

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to W&B and stdout."""
        if self._wandb is not None:
            self._wandb.log(metrics, step=self.num_timesteps)

        if self.verbose > 0:
            logger.info(
                f"[Step {self.num_timesteps}] "
                f"Sharpe={metrics['eval/sharpe_ratio']:.3f} "
                f"DD={metrics['eval/max_drawdown']:.3f} "
                f"WR={metrics['eval/win_rate']:.1%} "
                f"Trades={metrics['eval/num_trades']:.1f} "
                f"PnL=${metrics['eval/total_pnl']:.2f}"
            )
