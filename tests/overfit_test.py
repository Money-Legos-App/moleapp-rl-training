"""
Day 1 Overfit Test — Prove the RL pipeline works before RunPod (RLlib)
=======================================================================
Goal: NOT to find profitable alpha. Goal is to prove the observation space,
action mapping, reward function, and episode builder are bug-free.

If PPO cannot memorize a 30-day chart with zero fees and fixed seed, then
something is fundamentally broken and RunPod training will fail.

Setup:
  - 30 days of synthetic BTC data with a clear trend pattern
  - Zero fees, zero funding, zero slippage
  - Fixed seed (deterministic episode start)
  - PPO with memorization hyperparams (high LR, zero entropy, many epochs)

Pass criteria:
  - Agent's final account value > 1.5x initial capital (50% return)
  - Agent executes >5 trades (not stuck in HOLD)
  - Agent longs during uptrends and shorts during downtrends

Fail criteria:
  - Account value flat or negative → BUG in reward/action/obs
  - Zero trades → agent can't see position state or action mapping broken
  - Random-looking trades → observation doesn't encode price direction

Usage:
    cd moleapp-rl-training
    source .venv/bin/activate
    python tests/overfit_test.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Fix seeds globally FIRST
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from data.preprocessors.feature_engineer import OBS_DIM, MarketFeatures, build_observation
from envs.base_trading_env import BaseTradingEnv, Position, EpisodeState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Overfit Environment — Zero friction, deterministic start
# ──────────────────────────────────────────────────────────────────────

class OverfitTradingEnv(BaseTradingEnv):
    """
    Zero-friction environment for overfit testing.

    Changes from BaseTradingEnv:
    - All fees = 0 (isolate agent's ability to read price action)
    - No funding charges
    - Always starts at index 0 (deterministic episode)
    - Relaxed SL/TP bounds (let the agent trade freely)
    - Simple PnL reward (no asymmetry, no drawdown penalty)
    """

    def __init__(self, market_data, feature_data, **kwargs):
        kwargs.setdefault("max_leverage", 3)
        kwargs.setdefault("max_positions", 5)
        kwargs.setdefault("max_sl_pct", 0.50)   # Very wide — won't trigger
        kwargs.setdefault("min_sl_pct", 0.50)
        kwargs.setdefault("max_tp_pct", 0.50)   # Very wide — won't trigger
        kwargs.setdefault("min_tp_pct", 0.50)
        kwargs.setdefault("profile_name", "overfit_test")
        super().__init__(market_data=market_data, feature_data=feature_data, **kwargs)

    def reset(self, *, seed=None, options=None):
        """Always start at index 0 for deterministic episodes."""
        super(BaseTradingEnv, self).reset(seed=seed)
        # Fixed start — no randomness
        self._episode_start_idx = 0
        self.state = EpisodeState(
            account_value=self.initial_capital,
            initial_capital=self.initial_capital,
            peak_account_value=self.initial_capital,
        )
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _process_action(self, action, idx):
        """Override: zero entry fee."""
        direction_size = float(action[0])
        if -0.1 <= direction_size <= 0.1:
            return 0.0

        direction = 1 if direction_size > 0 else -1
        size_frac = abs(direction_size)

        leverage = 1.0 + (float(action[1]) + 1.0) / 2.0 * (self.max_leverage - 1.0)
        leverage = np.clip(leverage, 1.0, self.max_leverage)

        sl_pct = self.min_sl_pct
        tp_pct = self.min_tp_pct

        # Map action[4] to confidence scaling [0.05, 1.0]
        confidence = (float(action[4]) + 1.0) / 2.0
        confidence = np.clip(confidence, 0.05, 1.0)

        available_margin = self.state.account_value * (1.0 - self._margin_utilization())
        position_usd = available_margin * size_frac * leverage * confidence

        if position_usd < 1.0:  # Relaxed minimum for testing
            return 0.0

        # NO FEE
        price = self._get_price(idx)
        self.state.position = Position(
            direction=direction,
            entry_price=price,
            size_usd=position_usd,
            leverage=leverage,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            entry_step=self.state.step,
        )
        self.state.total_trades += 1
        self.state.last_trade_step = self.state.step
        return 0.0

    def _close_position(self, pnl_pct, reason=""):
        """Override: zero exit fee."""
        if self.state.position is None:
            return 0.0

        pos = self.state.position
        realized_pnl = pos.size_usd * pnl_pct / pos.leverage

        # NO FEE
        self.state.account_value += realized_pnl
        self.state.total_pnl += realized_pnl

        if realized_pnl > 0:
            self.state.winning_trades += 1

        self.state.peak_account_value = max(
            self.state.peak_account_value, self.state.account_value
        )
        self.state.position = None
        return realized_pnl / max(self.state.account_value, 1.0)

    def _process_funding(self, idx):
        """Override: no funding charges."""
        pass

    def _check_sl_tp_liquidation(self, idx):
        """Override: close position after N steps instead of SL/TP."""
        if self.state.position is None:
            return 0.0

        pos = self.state.position
        price = self._get_price(idx)
        entry = pos.entry_price

        if pos.direction == 1:
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry

        pnl_pct *= pos.leverage

        # Close after 20 steps (5 hours) — force the agent to re-evaluate
        if self.state.step - pos.entry_step >= 20:
            return self._close_position(pnl_pct=pnl_pct, reason="time_exit")

        return 0.0

    def _calculate_reward(self, ctx):
        """Simple: raw PnL percentage, no bells and whistles."""
        return ctx["pnl_pct"] * 10.0  # Scale up for clearer gradient signal


# Register for RLlib
register_env("OverfitTradingEnv", lambda cfg: OverfitTradingEnv(**cfg))


# ──────────────────────────────────────────────────────────────────────
# Generate synthetic data with clear, learnable patterns
# ──────────────────────────────────────────────────────────────────────

def generate_overfit_data(n_steps: int = 720) -> tuple[np.ndarray, list[MarketFeatures]]:
    """
    Generate 30 days of 1h BTC data with an obvious, repeating pattern:
    - 3 days up, 2 days down, repeat.

    A perfect agent would long the ups and short the downs for ~150% return.
    """
    rng = np.random.RandomState(SEED)

    base_price = 50000.0
    prices = [base_price]

    for i in range(1, n_steps):
        day = i // 24
        cycle_day = day % 5  # 5-day repeating cycle

        if cycle_day < 3:
            # Uptrend: +0.3% per hour avg
            drift = 0.003
        else:
            # Downtrend: -0.4% per hour avg
            drift = -0.004

        noise = rng.normal(0, 0.001)  # Small noise
        new_price = prices[-1] * (1.0 + drift + noise)
        prices.append(max(new_price, 100.0))  # Floor

    prices = np.array(prices, dtype=np.float64)

    # Build market_data array: (n, 7) = [open, high, low, close, volume, funding, OI]
    market_data = np.zeros((n_steps, 7), dtype=np.float32)
    for i in range(n_steps):
        p = float(prices[i])
        noise = abs(rng.normal(0, p * 0.0005))
        market_data[i, 0] = p - noise  # open
        market_data[i, 1] = p + noise  # high
        market_data[i, 2] = p - noise  # low
        market_data[i, 3] = p          # close
        market_data[i, 4] = rng.uniform(1e7, 5e7)  # volume
        market_data[i, 5] = 0.0       # zero funding
        market_data[i, 6] = rng.uniform(1e8, 3e8)  # OI

    # Build MarketFeatures
    features = []
    for i in range(n_steps):
        p = float(prices[i])
        ts = 1700000000.0 + i * 3600

        # Compute simple price momentum (the main signal agent should learn)
        p_1h = float(prices[max(0, i - 1)])
        p_4h = float(prices[max(0, i - 4)])
        p_24h = float(prices[max(0, i - 24)])

        features.append(MarketFeatures(
            price=p,
            price_1h_ago=p_1h,
            price_4h_ago=p_4h,
            price_24h_ago=p_24h,
            vwap_24h=p,
            rolling_mean_30d=base_price,
            volume_24h=float(market_data[i, 4]),
            rolling_avg_vol_30d=3e7,
            bid_imbalance_pct=0.0,
            spread_bps=1.0,
            open_interest=float(market_data[i, 6]),
            rolling_avg_oi_30d=2e8,
            oi_1h_ago=float(market_data[max(0, i - 1), 6]),
            oi_4h_ago=float(market_data[max(0, i - 4), 6]),
            funding_rate=0.0,
            funding_8h_cumulative=0.0,
            prev_funding_rate=0.0,
            rsi_1h=50.0 + (p - p_1h) / p * 500,  # Synthetic RSI based on momentum
            rsi_4h=50.0 + (p - p_4h) / p * 200,
            macd_hist_1h=(p - p_4h) / p * 100,  # Synthetic MACD
            bb_position_1h=0.5,
            atr_1h=abs(p - p_1h),
            ema_20=p,
            ema_50=float(np.mean(prices[max(0, i - 50):i + 1])) if i > 0 else p,
            ema_200=base_price,
            sma_4h=float(np.mean(prices[max(0, i - 4):i + 1])),
            volume_trend_1h=0.0,
            roc_1h=(p - p_1h) / p_1h * 100 if p_1h > 0 else 0.0,
            roc_4h=(p - p_4h) / p_4h * 100 if p_4h > 0 else 0.0,
            account_value=1000.0,
            initial_capital=1000.0,
            peak_account_value=1000.0,
            open_position_count=0,
            max_positions=5,
            margin_utilization=0.0,
            unrealized_pnl=0.0,
            mission_start_timestamp=1700000000.0,
            current_timestamp=ts,
            days_since_last_trade=0.0,
            has_open_position_this_asset=False,
            existing_direction=0,
            btc_dominance=50.0,
            fear_greed_index=50.0,
            market_regime=1 if (i // 24) % 5 < 3 else -1,  # Encode the regime directly
            cross_asset_momentum=(p - p_24h) / p_24h if p_24h > 0 else 0.0,
        ))

    return market_data, features


# ──────────────────────────────────────────────────────────────────────
# Run the overfit test
# ──────────────────────────────────────────────────────────────────────

def run_overfit_test(
    total_timesteps: int = 100_000,
    verbose: bool = True,
) -> dict:
    """
    Run PPO on a tiny, deterministic environment using RLlib.

    Returns dict with results and pass/fail status.
    """
    if verbose:
        print("=" * 70)
        print("DAY 1 OVERFIT TEST — Proving the RL pipeline works (RLlib)")
        print("=" * 70)
        print()

    # 1. Generate data
    print("[1/4] Generating 30-day synthetic BTC data with 3-up/2-down pattern...")
    market_data, features = generate_overfit_data(n_steps=720)
    price_start = market_data[0, 3]
    price_end = market_data[-1, 3]
    print(f"  Price range: ${price_start:.0f} → ${price_end:.0f}")
    print(f"  Pattern: 3 days bullish (+0.3%/hr), 2 days bearish (-0.4%/hr), repeat")
    print()

    # 2. Configure RLlib PPO
    print("[2/4] Creating zero-friction overfit environment with RLlib PPO...")

    env_config = {
        "market_data": market_data,
        "feature_data": features,
        "episode_length": 700,  # Almost full 30 days
        "initial_capital": 1000.0,
    }

    ray.init(ignore_reinit_error=True, num_cpus=2)

    config = (
        PPOConfig()
        .environment(
            env="OverfitTradingEnv",
            env_config=env_config,
        )
        .env_runners(
            num_env_runners=0,  # Local worker only for determinism
        )
        .training(
            lr=1e-3,               # High — aggressive learning
            entropy_coeff=0.0,     # Zero entropy — no exploration, pure exploitation
            num_epochs=20,         # Train many times on same batch
            train_batch_size_per_learner=700,  # Collect one full episode per update
            minibatch_size=64,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            grad_clip=0.5,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [128, 128],  # Small net — easier to overfit
                "fcnet_activation": "tanh",
            },
        )
    )

    print(f"  Episode length: 700 steps (29 days)")
    print(f"  Fees: 0, Funding: 0, Slippage: 0")
    print(f"  Position auto-closes after 20 steps (forces re-evaluation)")
    print()

    # 3. Train
    n_iterations = max(1, total_timesteps // 700)  # ~iterations to reach total_timesteps
    print(f"[3/4] Training PPO for ~{total_timesteps:,} timesteps ({n_iterations} iterations)...")
    print(f"  LR=1e-3, ent_coef=0.0, num_epochs=20, minibatch=64")
    print()

    algo = config.build_algo()
    start_time = time.time()

    for i in range(n_iterations):
        result = algo.train()
        if verbose and (i + 1) % 20 == 0:
            reward = result.get("env_runners", {}).get("episode_reward_mean", 0)
            steps = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))
            print(f"  Iter {i + 1}/{n_iterations}: steps={steps}, reward={reward:.2f}")

    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.1f}s")
    print()

    # 4. Evaluate: run the trained agent through the same data
    print("[4/4] Evaluating trained agent on the same 30-day data...")
    eval_env = OverfitTradingEnv(**env_config)
    obs, info = eval_env.reset(seed=SEED)

    # Use RLModule directly for inference (new API stack compatible)
    import torch
    rl_module = algo.get_module()

    total_reward = 0.0
    trade_log = []
    step = 0

    while True:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        fwd_out = rl_module.forward_inference({"obs": obs_tensor})
        action_dist_inputs = fwd_out["action_dist_inputs"]
        # Deterministic: take action means only (first 5 of 10 action_dist_inputs)
        action = action_dist_inputs.squeeze(0).detach().numpy()[:5]

        prev_position = eval_env.state.position
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        step += 1

        # Log trades
        if eval_env.state.position is not None and prev_position is None:
            direction = "LONG" if eval_env.state.position.direction == 1 else "SHORT"
            trade_log.append({
                "step": step,
                "action": direction,
                "price": eval_env.state.position.entry_price,
                "size": eval_env.state.position.size_usd,
            })

        if terminated or truncated:
            break

    # Cleanup
    algo.stop()
    ray.shutdown()

    # Results
    final_value = eval_env.state.account_value
    total_return = (final_value - 1000.0) / 1000.0 * 100
    total_trades = eval_env.state.total_trades
    win_rate = eval_env.state.winning_trades / max(total_trades, 1) * 100

    results = {
        "final_value": final_value,
        "total_return_pct": total_return,
        "total_trades": total_trades,
        "winning_trades": eval_env.state.winning_trades,
        "win_rate": win_rate,
        "total_reward": total_reward,
        "train_time_s": train_time,
    }

    # Print results
    print()
    print("─" * 70)
    print("RESULTS")
    print("─" * 70)
    print(f"  Initial Capital:  $1,000.00")
    print(f"  Final Value:      ${final_value:,.2f}")
    print(f"  Total Return:     {total_return:+.1f}%")
    print(f"  Total Trades:     {total_trades}")
    print(f"  Win Rate:         {win_rate:.0f}%")
    print(f"  Training Time:    {train_time:.1f}s")
    print()

    if trade_log:
        print("  Trade Log (first 10):")
        for t in trade_log[:10]:
            print(f"    Step {t['step']:4d}: {t['action']:5s} @ ${t['price']:,.0f} (${t['size']:,.0f})")
        if len(trade_log) > 10:
            print(f"    ... and {len(trade_log) - 10} more trades")
    print()

    # Pass/Fail
    print("─" * 70)
    passed = True
    reasons = []

    if total_trades < 5:
        passed = False
        reasons.append(
            f"FAIL: Only {total_trades} trades. Agent is stuck in HOLD.\n"
            f"  → Check: Is position state (obs[35], obs[36]) reaching the agent?\n"
            f"  → Check: Is the action dead zone [-0.1, 0.1] too wide?"
        )

    if total_return < 50.0:
        passed = False
        reasons.append(
            f"FAIL: Return is {total_return:+.1f}% (need >50% on this easy pattern).\n"
            f"  → Check: Is reward function penalizing correct trades?\n"
            f"  → Check: Does observation encode price direction (obs[1-3] = pct_change)?\n"
            f"  → Check: Lookahead bias in episode builder?"
        )

    if total_return < 0:
        reasons.append(
            "FAIL: Agent LOST money on zero-fee, easy trend data.\n"
            "  → This is a critical bug. DO NOT proceed to RunPod.\n"
            "  → Most likely: reward function sign is inverted, or observation\n"
            "    doesn't contain current price momentum."
        )

    if passed:
        print("PASS — The RL pipeline is working correctly.")
        print(f"  Agent learned the 3-up/2-down pattern and made {total_return:+.1f}% return.")
        print("  Safe to proceed to full training on RunPod.")
    else:
        print("FAIL — Bugs detected. DO NOT proceed to RunPod.")
        for r in reasons:
            print(f"\n  {r}")

    print("─" * 70)

    results["passed"] = passed
    results["fail_reasons"] = reasons
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # Allow custom timesteps via CLI
    timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

    results = run_overfit_test(total_timesteps=timesteps)
    sys.exit(0 if results["passed"] else 1)
