"""
Hyperliquid OHLCV Historical Data Collector
=============================================
Pulls candle data via HL REST API for training dataset construction.
Supports 1h, 4h, 1d timeframes for all allowed assets.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# MoleApp's 15 allowed assets
ALLOWED_ASSETS = [
    "BTC", "ETH", "SOL", "SUI", "SEI", "AVAX", "TAO", "FET",
    "NEAR", "WIF", "POPCAT", "kPEPE", "DOGE", "PENDLE", "ARB",
]

TIMEFRAMES = {
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def fetch_candles(
    asset: str,
    timeframe: str = "1h",
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
    max_candles: int = 5000,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Hyperliquid.

    Args:
        asset: Asset name (e.g., "BTC")
        timeframe: "1h", "4h", or "1d"
        start_time_ms: Start timestamp in milliseconds
        end_time_ms: End timestamp in milliseconds
        max_candles: Maximum candles per request (HL limit: 5000)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)

    interval = TIMEFRAMES.get(timeframe, 3600)
    if start_time_ms is None:
        start_time_ms = end_time_ms - (max_candles * interval * 1000)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": asset,
            "interval": timeframe,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
        },
    }

    try:
        resp = requests.post(HL_INFO_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            logger.warning(f"No candle data for {asset} {timeframe}")
            return pd.DataFrame()

        rows = []
        for candle in data:
            rows.append({
                "timestamp": candle["t"],
                "open": float(candle["o"]),
                "high": float(candle["h"]),
                "low": float(candle["l"]),
                "close": float(candle["c"]),
                "volume": float(candle["v"]),
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception as e:
        logger.error(f"Failed to fetch candles for {asset}: {e}")
        return pd.DataFrame()


def fetch_all_history(
    asset: str,
    timeframe: str = "1h",
    lookback_days: int = 730,  # 2 years
    output_dir: str = "data/datasets",
) -> Optional[Path]:
    """
    Fetch full history for an asset by paginating through time windows.

    Returns path to saved parquet file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (lookback_days * 86400 * 1000)

    interval_ms = TIMEFRAMES[timeframe] * 1000
    chunk_size = 5000  # HL max per request
    chunk_ms = chunk_size * interval_ms

    all_dfs = []
    current_start = start_ms

    while current_start < end_ms:
        current_end = min(current_start + chunk_ms, end_ms)
        logger.info(
            f"Fetching {asset} {timeframe}: "
            f"{pd.Timestamp(current_start, unit='ms')} → "
            f"{pd.Timestamp(current_end, unit='ms')}"
        )

        df = fetch_candles(
            asset=asset,
            timeframe=timeframe,
            start_time_ms=current_start,
            end_time_ms=current_end,
        )

        if not df.empty:
            all_dfs.append(df)

        current_start = current_end
        time.sleep(0.5)  # Rate limit courtesy

    if not all_dfs:
        logger.warning(f"No data collected for {asset} {timeframe}")
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    full_df = full_df.reset_index(drop=True)

    file_path = output_path / f"{asset}_{timeframe}_{lookback_days}d.parquet"
    full_df.to_parquet(file_path, index=False)
    logger.info(f"Saved {len(full_df)} candles to {file_path}")

    return file_path


def collect_all_assets(
    timeframe: str = "1h",
    lookback_days: int = 730,
    output_dir: str = "data/datasets",
) -> dict[str, Path]:
    """Collect history for all 15 MoleApp assets."""
    results = {}
    for asset in ALLOWED_ASSETS:
        path = fetch_all_history(
            asset=asset,
            timeframe=timeframe,
            lookback_days=lookback_days,
            output_dir=output_dir,
        )
        if path:
            results[asset] = path
        time.sleep(1.0)  # Be nice to HL API
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting full OHLCV collection for all assets...")
    results = collect_all_assets(timeframe="1h", lookback_days=730)
    logger.info(f"Collected data for {len(results)} assets")
