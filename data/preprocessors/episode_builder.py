"""
Episode Builder — Transforms raw data into Gymnasium-ready episodes
====================================================================
Takes OHLCV, funding, and OI parquet files and produces:
1. market_data array: (timesteps, n_features) with OHLCV + funding + OI
2. feature_data list: List[MarketFeatures] for observation building

Also computes all technical indicators (RSI, MACD, Bollinger Bands, EMA,
ATR, ROC) using the `ta` library, ensuring consistent computation between
training and production.

Usage:
    builder = EpisodeBuilder(data_dir="data/datasets")
    market_data, feature_list = builder.build_episodes("BTC")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta

from data.preprocessors.feature_engineer import MarketFeatures

logger = logging.getLogger(__name__)

# Market data array column layout (for BaseTradingEnv)
# Columns: open, high, low, close, volume, funding_rate, open_interest
MARKET_DATA_COLS = ["open", "high", "low", "close", "volume", "funding_rate", "open_interest"]


class EpisodeBuilder:
    """Builds training episodes from raw collected data."""

    def __init__(
        self,
        data_dir: str = "data/datasets",
        lookback_days: int = 730,
        timeframe: str = "1h",
    ):
        self.data_dir = Path(data_dir)
        self.lookback_days = lookback_days
        self.timeframe = timeframe

    def build_episodes(
        self,
        asset: str,
        initial_capital: float = 1000.0,
    ) -> tuple[np.ndarray, list[MarketFeatures]]:
        """
        Build complete market_data array and MarketFeatures list for an asset.

        Args:
            asset: Asset name (e.g., "BTC")
            initial_capital: Default starting capital for portfolio features

        Returns:
            (market_data, feature_list) tuple ready for BaseTradingEnv
        """
        # Load raw data
        df = self._load_and_merge(asset)
        if df is None or df.empty:
            raise ValueError(f"No data available for {asset}")

        # Compute technical indicators
        df = self._compute_technicals(df)

        # Compute cross-timeframe summaries
        df = self._compute_multi_timeframe(df)

        # Drop NaN rows from indicator warmup
        warmup = 200  # EMA-200 needs ~200 candles to be meaningful
        df = df.iloc[warmup:].reset_index(drop=True)

        if len(df) < 500:
            raise ValueError(f"Insufficient data after warmup for {asset}: {len(df)} rows")

        # Build market_data array
        market_data = self._build_market_array(df)

        # Build MarketFeatures list
        feature_list = self._build_feature_list(df, initial_capital)

        logger.info(
            f"Built episodes for {asset}: {len(df)} timesteps, "
            f"market_data shape={market_data.shape}"
        )

        return market_data, feature_list

    def _load_and_merge(self, asset: str) -> Optional[pd.DataFrame]:
        """Load OHLCV, funding, and OI data, merge on timestamp."""
        ohlcv_path = self.data_dir / f"{asset}_{self.timeframe}_{self.lookback_days}d.parquet"
        funding_path = self.data_dir / f"{asset}_funding_{self.lookback_days}d.parquet"
        oi_path = self.data_dir / f"{asset}_oi_derived.parquet"

        if not ohlcv_path.exists():
            logger.error(f"OHLCV file not found: {ohlcv_path}")
            return None

        # Load OHLCV (required)
        ohlcv = pd.read_parquet(ohlcv_path)
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])
        ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)

        # Load funding (merge as-of)
        if funding_path.exists():
            funding = pd.read_parquet(funding_path)
            funding["timestamp"] = pd.to_datetime(funding["timestamp"])
            funding = funding.sort_values("timestamp")
            ohlcv = pd.merge_asof(
                ohlcv, funding[["timestamp", "funding_rate"]], on="timestamp", direction="backward"
            )
        else:
            logger.warning(f"No funding data for {asset}, using zeros")
            ohlcv["funding_rate"] = 0.0

        ohlcv["funding_rate"] = ohlcv["funding_rate"].fillna(0.0)

        # Load OI (merge as-of)
        if oi_path.exists():
            oi = pd.read_parquet(oi_path)
            oi["timestamp"] = pd.to_datetime(oi["timestamp"])
            oi = oi.sort_values("timestamp")
            ohlcv = pd.merge_asof(
                ohlcv, oi[["timestamp", "open_interest"]], on="timestamp", direction="backward"
            )
        else:
            logger.warning(f"No OI data for {asset}, deriving from volume")
            ohlcv["open_interest"] = ohlcv["volume"].ewm(span=24, adjust=False).mean()

        ohlcv["open_interest"] = ohlcv["open_interest"].ffill().fillna(0.0)

        return ohlcv

    def _compute_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators using the `ta` library."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI (14-period)
        df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # MACD (12, 26, 9)
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Bollinger Bands (20, 2)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_position"] = (close - df["bb_lower"]) / (
            (df["bb_upper"] - df["bb_lower"]).clip(lower=1e-10)
        )
        df["bb_position"] = df["bb_position"].clip(0.0, 1.0)

        # ATR (14-period)
        df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        # EMAs
        df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
        df["ema_200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()

        # SMA (for 4h equivalent on 1h data, use 4-period)
        df["sma_4h"] = close.rolling(window=4).mean()

        # Rate of Change
        df["roc_1"] = ta.momentum.ROCIndicator(close, window=1).roc()  # 1h ROC
        df["roc_4"] = ta.momentum.ROCIndicator(close, window=4).roc()  # 4h ROC

        # Volume trend (normalized)
        vol_sma = volume.rolling(window=20).mean()
        df["volume_trend"] = (volume / vol_sma.clip(lower=1.0)) - 1.0

        # VWAP approximation (24h rolling)
        df["vwap_24h"] = (
            (close * volume).rolling(window=24).sum()
            / volume.rolling(window=24).sum().clip(lower=1.0)
        )

        # Rolling averages for normalization
        df["rolling_mean_30d"] = close.rolling(window=720, min_periods=1).mean()
        df["rolling_avg_vol_30d"] = volume.rolling(window=720, min_periods=1).mean()

        # OI features
        df["rolling_avg_oi_30d"] = df["open_interest"].rolling(window=720, min_periods=1).mean()

        # Funding cumulative (last 3 periods = 24h of 8h funding)
        df["funding_8h_cum"] = df["funding_rate"].rolling(window=3, min_periods=1).sum()
        df["prev_funding"] = df["funding_rate"].shift(1).fillna(0.0)

        # Bid imbalance and spread (not available from OHLCV, use proxies)
        # Proxy: (close - open) / (high - low) captures buying vs selling pressure
        hl_range = (high - low).clip(lower=1e-10)
        df["bid_imbalance_proxy"] = ((close - low) - (high - close)) / hl_range * 100.0
        df["spread_proxy"] = (hl_range / close) * 10000.0  # in basis points

        return df

    def _compute_multi_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 4h-equivalent RSI from 1h data."""
        # Resample-like: 4h RSI = RSI over 4*14=56 1h candles
        df["rsi_4h"] = ta.momentum.RSIIndicator(df["close"], window=56).rsi()

        # Cross-asset momentum placeholder (single asset, so use ROC average)
        df["cross_asset_momentum"] = (
            df["roc_1"].rolling(window=24, min_periods=1).mean() / 100.0
        )

        return df

    def _build_market_array(self, df: pd.DataFrame) -> np.ndarray:
        """Build (timesteps, 7) numpy array for BaseTradingEnv."""
        cols = ["open", "high", "low", "close", "volume", "funding_rate", "open_interest"]
        return df[cols].values.astype(np.float32)

    def _build_feature_list(
        self,
        df: pd.DataFrame,
        initial_capital: float,
    ) -> list[MarketFeatures]:
        """
        Build list of MarketFeatures, one per timestep.

        Portfolio state fields are set to defaults — BaseTradingEnv._get_observation()
        will override them with actual episode state at runtime.
        """
        features = []
        timestamps = df["timestamp"].values

        for i in range(len(df)):
            row = df.iloc[i]
            ts = float(pd.Timestamp(timestamps[i]).timestamp())

            # Lagged prices
            price_1h = float(df.iloc[max(0, i - 1)]["close"])
            price_4h = float(df.iloc[max(0, i - 4)]["close"])
            price_24h = float(df.iloc[max(0, i - 24)]["close"])

            # Lagged OI
            oi_1h = float(df.iloc[max(0, i - 1)]["open_interest"])
            oi_4h = float(df.iloc[max(0, i - 4)]["open_interest"])

            mf = MarketFeatures(
                # Price
                price=float(row["close"]),
                price_1h_ago=price_1h,
                price_4h_ago=price_4h,
                price_24h_ago=price_24h,
                vwap_24h=float(row.get("vwap_24h", row["close"])),
                rolling_mean_30d=float(row.get("rolling_mean_30d", row["close"])),

                # Volume & Liquidity
                volume_24h=float(row["volume"]),
                rolling_avg_vol_30d=float(row.get("rolling_avg_vol_30d", row["volume"])),
                bid_imbalance_pct=float(row.get("bid_imbalance_proxy", 0.0)),
                spread_bps=float(row.get("spread_proxy", 1.0)),

                # Open Interest
                open_interest=float(row["open_interest"]),
                rolling_avg_oi_30d=float(row.get("rolling_avg_oi_30d", row["open_interest"])),
                oi_1h_ago=oi_1h,
                oi_4h_ago=oi_4h,

                # Funding Rate
                funding_rate=float(row["funding_rate"]),
                funding_8h_cumulative=float(row.get("funding_8h_cum", 0.0)),
                prev_funding_rate=float(row.get("prev_funding", 0.0)),

                # Technical Indicators
                rsi_1h=float(row.get("rsi_14", 50.0)),
                rsi_4h=float(row.get("rsi_4h", 50.0)),
                macd_hist_1h=float(row.get("macd_hist", 0.0)),
                bb_position_1h=float(row.get("bb_position", 0.5)),
                atr_1h=float(row.get("atr_14", 0.0)),
                ema_20=float(row.get("ema_20", row["close"])),
                ema_50=float(row.get("ema_50", row["close"])),
                ema_200=float(row.get("ema_200", row["close"])),
                sma_4h=float(row.get("sma_4h", row["close"])),
                volume_trend_1h=float(row.get("volume_trend", 0.0)),
                roc_1h=float(row.get("roc_1", 0.0)),
                roc_4h=float(row.get("roc_4", 0.0)),

                # Portfolio State (defaults — overridden by env at runtime)
                account_value=initial_capital,
                initial_capital=initial_capital,
                peak_account_value=initial_capital,
                open_position_count=0,
                max_positions=5,
                margin_utilization=0.0,
                unrealized_pnl=0.0,
                mission_start_timestamp=ts,
                current_timestamp=ts,
                days_since_last_trade=0.0,
                has_open_position_this_asset=False,
                existing_direction=0,

                # Market Regime (defaults, single-asset context)
                btc_dominance=50.0,
                fear_greed_index=50.0,
                market_regime=0,
                cross_asset_momentum=float(row.get("cross_asset_momentum", 0.0)),
            )
            features.append(mf)

        return features


def build_all_assets(
    data_dir: str = "data/datasets",
    output_dir: str = "data/episodes",
    lookback_days: int = 730,
) -> dict[str, Path]:
    """
    Build and cache episode data for all assets.

    Saves (market_data.npy, feature_count.txt) per asset for quick loading.
    """
    from data.collectors.hl_ohlcv_collector import ALLOWED_ASSETS

    builder = EpisodeBuilder(data_dir=data_dir, lookback_days=lookback_days)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = {}

    for asset in ALLOWED_ASSETS:
        try:
            market_data, feature_list = builder.build_episodes(asset)

            asset_dir = output_path / asset
            asset_dir.mkdir(exist_ok=True)

            # Save market data as numpy
            np.save(asset_dir / "market_data.npy", market_data)

            # Save features as pickle (MarketFeatures are dataclasses)
            import pickle
            with open(asset_dir / "features.pkl", "wb") as f:
                pickle.dump(feature_list, f)

            results[asset] = asset_dir
            logger.info(f"Cached episodes for {asset}: {len(feature_list)} timesteps")

        except Exception as e:
            logger.warning(f"Failed to build episodes for {asset}: {e}")

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build training episodes")
    parser.add_argument("--asset", default=None, help="Single asset to build (or all)")
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--output-dir", default="data/episodes")
    args = parser.parse_args()

    if args.asset:
        builder = EpisodeBuilder(data_dir=args.data_dir)
        market_data, features = builder.build_episodes(args.asset)
        print(f"Built {args.asset}: {market_data.shape[0]} timesteps, {market_data.shape[1]} features")
    else:
        results = build_all_assets(data_dir=args.data_dir, output_dir=args.output_dir)
        print(f"Built episodes for {len(results)} assets")
