"""
Shared Feature Engineering Module
==================================
THIS FILE IS THE SINGLE SOURCE OF TRUTH for observation vector construction.
It is used IDENTICALLY in:
  1. Training: moleapp-rl-training/data/preprocessors/feature_engineer.py
  2. Production: agent-service/app/services/rl/rl_feature_builder.py (symlinked or copied)

ANY change here MUST be reflected in both locations and validated via
cross-validation tests (tests/test_feature_parity.py).

Feature version hash is embedded in every observation for skew detection.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Feature version — bump on ANY change to feature computation
FEATURE_VERSION = "1.1.0"
FEATURE_HASH = hashlib.md5(f"moleapp-features-{FEATURE_VERSION}".encode()).hexdigest()[:8]

# Total observation dimensions
OBS_DIM = 47

# Asset identity map — sorted alphabetically, normalized to [0, 1]
_ASSETS_SORTED = sorted([
    "ARB", "AVAX", "BTC", "DOGE", "ETH", "FET", "kPEPE", "NEAR",
    "PENDLE", "POPCAT", "SEI", "SOL", "SUI", "TAO", "WIF",
])
ASSET_ID_MAP = {a: i / max(len(_ASSETS_SORTED) - 1, 1) for i, a in enumerate(_ASSETS_SORTED)}


@dataclass
class MarketFeatures:
    """Raw market data inputs for a single asset at a single timestep."""

    # Price
    price: float
    price_1h_ago: float
    price_4h_ago: float
    price_24h_ago: float
    vwap_24h: float
    rolling_mean_30d: float

    # Volume & Liquidity
    volume_24h: float
    rolling_avg_vol_30d: float
    bid_imbalance_pct: float  # -100 to +100
    spread_bps: float

    # Open Interest
    open_interest: float
    rolling_avg_oi_30d: float
    oi_1h_ago: float
    oi_4h_ago: float

    # Funding Rate
    funding_rate: float  # per 8h period
    funding_8h_cumulative: float  # sum of last 3 periods
    prev_funding_rate: float

    # Technical Indicators (pre-computed from candles)
    rsi_1h: float  # 0-100
    rsi_4h: float
    macd_hist_1h: float
    bb_position_1h: float  # 0-1 (position within Bollinger Bands)
    atr_1h: float
    ema_20: float
    ema_50: float
    ema_200: float
    sma_4h: float
    volume_trend_1h: float  # normalized volume direction
    roc_1h: float  # rate of change
    roc_4h: float

    # Portfolio State
    account_value: float
    initial_capital: float
    peak_account_value: float
    open_position_count: int
    max_positions: int
    margin_utilization: float  # 0-1
    unrealized_pnl: float
    mission_start_timestamp: float  # unix epoch
    current_timestamp: float
    days_since_last_trade: float
    has_open_position_this_asset: bool
    existing_direction: int  # -1=short, 0=flat, 1=long

    # Market Regime (cross-asset)
    btc_dominance: float  # 0-100
    fear_greed_index: float  # 0-100
    market_regime: int  # -1=bear, 0=neutral, 1=bull
    cross_asset_momentum: float  # average momentum across all assets

    # v1.1.0: Asset identity & liquidation distance (formerly reserved slots)
    asset_id_normalized: float = 0.0  # [0, 1] from ASSET_ID_MAP
    distance_to_liquidation: float = 1.0  # [0, 1]: 0=liquidated, 1=safe/no position


def build_observation(features: MarketFeatures) -> np.ndarray:
    """
    Build the 47-dimensional observation vector from raw market features.

    All outputs are raw (un-normalized). VecNormalize handles scaling during
    training; production inference applies saved VecNormalize stats.

    Returns:
        np.ndarray of shape (47,) with dtype float32
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # --- Price features (5) ---
    obs[0] = _safe_div(features.price, features.rolling_mean_30d)
    obs[1] = _pct_change(features.price, features.price_1h_ago)
    obs[2] = _pct_change(features.price, features.price_4h_ago)
    obs[3] = _pct_change(features.price, features.price_24h_ago)
    obs[4] = _safe_div(features.price, features.vwap_24h) - 1.0

    # --- Volume & Liquidity (4) ---
    obs[5] = _safe_div(features.volume_24h, features.rolling_avg_vol_30d)
    obs[6] = _safe_div(features.volume_24h, features.rolling_avg_vol_30d)  # vol_vs_avg
    obs[7] = features.bid_imbalance_pct / 100.0  # normalize to [-1, 1]
    obs[8] = features.spread_bps / 100.0  # normalize basis points

    # --- Open Interest (3) ---
    obs[9] = _safe_div(features.open_interest, features.rolling_avg_oi_30d)
    obs[10] = _pct_change(features.open_interest, features.oi_1h_ago)
    oi_momentum_1h = _pct_change(features.open_interest, features.oi_1h_ago)
    oi_momentum_4h = _pct_change(features.open_interest, features.oi_4h_ago)
    obs[11] = oi_momentum_1h - oi_momentum_4h  # OI acceleration

    # --- Funding Rate (4) ---
    obs[12] = features.funding_rate * 10000  # scale up tiny values
    obs[13] = features.funding_8h_cumulative * 10000
    obs[14] = features.funding_rate * 3 * 365 * 100  # annualized %
    obs[15] = (features.funding_rate - features.prev_funding_rate) * 10000  # trend

    # --- Technical Indicators (12) ---
    obs[16] = (features.rsi_1h - 50.0) / 50.0  # center around 0, range [-1, 1]
    obs[17] = (features.rsi_4h - 50.0) / 50.0
    obs[18] = features.macd_hist_1h  # already centered around 0
    obs[19] = features.bb_position_1h * 2.0 - 1.0  # [0,1] -> [-1,1]
    obs[20] = _safe_div(features.atr_1h, features.price)  # normalized volatility
    obs[21] = _ema_cross_signal(features.ema_20, features.ema_50)
    obs[22] = _ema_cross_signal(features.ema_50, features.ema_200)
    obs[23] = 1.0 if features.price > features.sma_4h else -1.0
    obs[24] = features.volume_trend_1h
    obs[25] = features.roc_1h / 100.0  # normalize percentage
    obs[26] = features.roc_4h / 100.0
    obs[27] = _volatility_regime(features.atr_1h, features.price)

    # --- Portfolio State (9) ---
    obs[28] = _safe_div(features.account_value, features.initial_capital)
    obs[29] = _drawdown(features.account_value, features.peak_account_value)
    obs[30] = features.open_position_count / max(features.max_positions, 1)
    obs[31] = features.margin_utilization
    obs[32] = _safe_div(features.unrealized_pnl, features.account_value)
    obs[33] = _time_in_mission(features.mission_start_timestamp, features.current_timestamp)
    obs[34] = min(features.days_since_last_trade / 7.0, 1.0)  # cap at 1 week
    obs[35] = 1.0 if features.has_open_position_this_asset else 0.0
    obs[36] = float(features.existing_direction)

    # --- Market Regime (4) ---
    obs[37] = features.btc_dominance / 100.0
    obs[38] = features.fear_greed_index / 100.0
    obs[39] = float(features.market_regime)
    obs[40] = features.cross_asset_momentum

    # --- Time Features (4, cyclical encoding) ---
    hour = _hour_from_timestamp(features.current_timestamp)
    day_of_week = _day_of_week_from_timestamp(features.current_timestamp)
    obs[41] = math.sin(2 * math.pi * hour / 24.0)
    obs[42] = math.cos(2 * math.pi * hour / 24.0)
    obs[43] = math.sin(2 * math.pi * day_of_week / 7.0)
    obs[44] = math.cos(2 * math.pi * day_of_week / 7.0)

    # --- Asset Identity & Liquidation Distance (2, v1.1.0) ---
    obs[45] = features.asset_id_normalized
    obs[46] = features.distance_to_liquidation

    return obs


# ──────────────────────────────────────────────────────────────────────
# Helper functions (deterministic, no external dependencies)
# ──────────────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float, default: float = 1.0) -> float:
    """Division with zero protection."""
    if abs(b) < 1e-10:
        return default
    return a / b


def _pct_change(current: float, previous: float) -> float:
    """Percentage change from previous to current."""
    if abs(previous) < 1e-10:
        return 0.0
    return (current - previous) / previous


def _ema_cross_signal(fast_ema: float, slow_ema: float) -> float:
    """Cross signal: positive = fast above slow, magnitude = distance."""
    if abs(slow_ema) < 1e-10:
        return 0.0
    return (fast_ema - slow_ema) / slow_ema


def _volatility_regime(atr: float, price: float) -> float:
    """Map ATR/price ratio to regime: low (<0.01), medium (0.01-0.03), high (>0.03)."""
    ratio = _safe_div(atr, price, default=0.0)
    if ratio < 0.01:
        return -1.0  # low vol
    elif ratio < 0.03:
        return 0.0  # medium vol
    else:
        return 1.0  # high vol


def _drawdown(current_value: float, peak_value: float) -> float:
    """Current drawdown as a fraction (0 = no drawdown, 1 = total loss)."""
    if peak_value <= 0:
        return 0.0
    dd = (peak_value - current_value) / peak_value
    return max(0.0, min(dd, 1.0))


def _time_in_mission(start_ts: float, current_ts: float, max_days: float = 45.0) -> float:
    """Fraction of mission elapsed (0-1, capped at max_days)."""
    if start_ts <= 0 or current_ts <= start_ts:
        return 0.0
    elapsed_days = (current_ts - start_ts) / 86400.0
    return min(elapsed_days / max_days, 1.0)


def _hour_from_timestamp(ts: float) -> float:
    """Extract hour (0-23) from unix timestamp (UTC)."""
    import time
    return time.gmtime(ts).tm_hour


def _day_of_week_from_timestamp(ts: float) -> float:
    """Extract day of week (0=Monday, 6=Sunday) from unix timestamp."""
    import time
    return time.gmtime(ts).tm_wday


def get_feature_version() -> str:
    """Return the current feature version string."""
    return FEATURE_VERSION


def get_feature_hash() -> str:
    """Return the feature version hash for skew detection."""
    return FEATURE_HASH
