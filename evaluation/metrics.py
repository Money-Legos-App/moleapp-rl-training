"""
RL Model Evaluation Metrics
=============================
Sharpe ratio, Calmar ratio, max drawdown, win rate, and more.
Used for both training evaluation and shadow-mode comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalMetrics:
    """Complete evaluation metrics for a model/period."""
    total_return_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    avg_trade_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_hold_time_hours: float
    trade_frequency_per_day: float


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: float = 365 * 24 / 15) -> float:
    """
    Annualized Sharpe ratio.

    Args:
        returns: Array of per-step returns (not cumulative)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year (15-min steps)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    if std_return < 1e-10:
        return 0.0

    return float(mean_return / std_return * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    if len(equity_curve) < 2:
        return 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.maximum(peak, 1e-10)
    return float(np.max(drawdown))


def calculate_calmar(total_return: float, max_drawdown: float, period_years: float = 1.0) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    if max_drawdown < 1e-10:
        return 0.0
    annualized_return = total_return / max(period_years, 1e-10)
    return annualized_return / max_drawdown


def calculate_profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    """Profit factor: gross profit / gross loss."""
    gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
    if gross_loss < 1e-10:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def evaluate_episode(info_history: list[dict]) -> EvalMetrics:
    """
    Calculate full evaluation metrics from episode info history.

    Args:
        info_history: List of info dicts from env.step() calls
    """
    if not info_history:
        return EvalMetrics(
            total_return_pct=0.0, sharpe_ratio=0.0, calmar_ratio=0.0,
            max_drawdown_pct=0.0, win_rate=0.0, total_trades=0,
            avg_trade_pnl_pct=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            profit_factor=0.0, avg_hold_time_hours=0.0, trade_frequency_per_day=0.0,
        )

    # Build equity curve
    equity = np.array([info["account_value"] for info in info_history])
    initial = info_history[0].get("account_value", 1000.0)

    # Per-step returns
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-10)

    total_return = (equity[-1] - initial) / initial
    max_dd = calculate_max_drawdown(equity)
    sharpe = calculate_sharpe(returns)

    total_steps = len(info_history)
    period_days = total_steps * 15 / 1440  # 15-min steps
    period_years = period_days / 365

    calmar = calculate_calmar(total_return, max_dd, period_years)

    final_info = info_history[-1]
    total_trades = final_info.get("total_trades", 0)
    winning_trades = final_info.get("winning_trades", 0)
    win_rate = winning_trades / max(total_trades, 1)

    return EvalMetrics(
        total_return_pct=total_return * 100,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd * 100,
        win_rate=win_rate,
        total_trades=total_trades,
        avg_trade_pnl_pct=total_return / max(total_trades, 1) * 100,
        avg_win_pct=0.0,  # TODO: Track per-trade PnL
        avg_loss_pct=0.0,
        profit_factor=0.0,  # TODO: Track per-trade wins/losses
        avg_hold_time_hours=0.0,
        trade_frequency_per_day=total_trades / max(period_days, 1),
    )
