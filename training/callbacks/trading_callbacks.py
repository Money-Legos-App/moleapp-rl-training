"""
RLlib Trading Callbacks
========================
Custom RLlib callbacks that extract trading-specific metrics (Sharpe,
drawdown, win rate) from episode info dicts and log them as custom metrics.

These metrics are automatically included in RLlib's result dict and can
be logged to W&B via the training loop or Ray Tune's WandbLoggerCallback.
"""

from __future__ import annotations

import logging

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


def compute_sharpe(
    episode_returns: list[float],
    annualize_factor: float = 3.49,  # sqrt(365/30) for 30-day episodes
) -> float:
    """
    Compute annualized Sharpe ratio from episode returns.

    Args:
        episode_returns: List of total returns (e.g., PnL / initial_capital)
        annualize_factor: sqrt(periods_per_year). Default = sqrt(365/30) ~ 3.49

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


class TradingCallbacks(DefaultCallbacks):
    """
    Extract trading metrics from env info at episode end.

    RLlib automatically aggregates custom_metrics across episodes and
    reports mean/min/max in the training result dict.
    """

    def on_episode_end(self, *, episode, env_runner, base_env, env_index, rl_module, metrics_logger, **kwargs):
        """Called at the end of each episode. Extract trading metrics from info."""
        # Get the last info dict from the episode
        info = episode.last_info_for()
        if info is None:
            return

        # Extract metrics from BaseTradingEnv._get_info()
        total_pnl = info.get("total_pnl", 0.0)
        initial_capital = info.get("account_value", 1000.0) - total_pnl
        if initial_capital <= 0:
            initial_capital = 1000.0

        total_return = total_pnl / initial_capital
        total_trades = info.get("total_trades", 0)
        winning_trades = info.get("winning_trades", 0)
        win_rate = winning_trades / max(total_trades, 1)
        max_drawdown = info.get("max_drawdown", 0.0)

        # Log as custom metrics (RLlib aggregates these automatically)
        metrics_logger.log_value("total_return", total_return)
        metrics_logger.log_value("max_drawdown", max_drawdown)
        metrics_logger.log_value("win_rate", win_rate)
        metrics_logger.log_value("total_trades", total_trades)
        metrics_logger.log_value("total_pnl", total_pnl)
        metrics_logger.log_value("winning_trades", winning_trades)
