"""
RLlib Trading Callbacks (New API Stack)
========================================
Custom RLlib callbacks that extract trading-specific metrics (Sharpe,
drawdown, win rate) from episode info dicts and log them via MetricsLogger.

Uses the new API stack (RLModule + Learner + ConnectorV2 + EnvRunner).
Metrics are reported under result["env_runners"]["<key>"] and can be
logged to W&B via the training loop.

V10: All metrics sanitized with nan_to_num before logging to prevent
NaN propagation into RLlib aggregation and W&B summaries.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


def _safe_float(val: float, default: float = 0.0) -> float:
    """Sanitize a float — replace NaN/Inf with default."""
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def compute_sharpe(
    episode_returns: list[float],
    annualize_factor: float = 3.49,  # sqrt(365/30) for 30-day episodes
) -> float:
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

    V10: every metric is sanitized through _safe_float before logging.
    Zero-trade episodes explicitly produce 0.0 for all rate metrics.
    """

    def on_episode_end(self, *, episode, env_runner, metrics_logger, env, env_index, **kwargs):
        """Called at the end of each episode. Extract trading metrics from info."""
        infos = episode.get_infos()
        if not infos:
            return
        info = infos[-1]

        # Extract raw values from BaseTradingEnv._get_info()
        total_pnl = float(info.get("total_pnl", 0.0))
        account_value = float(info.get("account_value", 1000.0))
        total_trades = int(info.get("total_trades", 0))
        winning_trades = int(info.get("winning_trades", 0))
        max_drawdown = float(info.get("max_drawdown", 0.0))

        # Compute initial capital (avoid negative/zero)
        initial_capital = max(account_value - total_pnl, 1.0)

        # V10: explicit zero-trade guard — prevent all division-by-zero NaNs
        if total_trades == 0:
            win_rate = 0.0
            total_pnl = 0.0
            total_return = 0.0
            risk_adjusted_return = 0.0
        else:
            win_rate = winning_trades / total_trades
            total_return = total_pnl / initial_capital
            risk_adjusted_return = total_return / max(max_drawdown, 0.01)

        # V10: sanitize every value through nan_to_num before logging
        metrics_logger.log_value("total_return", _safe_float(total_return))
        metrics_logger.log_value("max_drawdown", _safe_float(max_drawdown))
        metrics_logger.log_value("win_rate", _safe_float(win_rate))
        metrics_logger.log_value("total_trades", total_trades)
        metrics_logger.log_value("total_pnl", _safe_float(total_pnl))
        metrics_logger.log_value("winning_trades", winning_trades)
        metrics_logger.log_value("risk_adjusted_return", _safe_float(risk_adjusted_return))
