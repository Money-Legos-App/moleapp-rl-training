"""
Hyperliquid Funding Rate Historical Data Collector
====================================================
Pulls funding rate history for all allowed assets.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from data.collectors.asset_config import ALLOWED_ASSETS
from data.collectors.hl_ohlcv_collector import HL_INFO_URL

logger = logging.getLogger(__name__)


def fetch_funding_history(
    asset: str,
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch funding rate history from Hyperliquid.

    Returns DataFrame with columns: timestamp, funding_rate
    """
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)
    if start_time_ms is None:
        start_time_ms = end_time_ms - (730 * 86400 * 1000)  # 2 years

    payload = {
        "type": "fundingHistory",
        "coin": asset,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
    }

    try:
        resp = requests.post(HL_INFO_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame()

        rows = []
        for entry in data:
            rows.append({
                "timestamp": pd.Timestamp(entry["time"], unit="ms"),
                "funding_rate": float(entry["fundingRate"]),
                "premium": float(entry.get("premium", 0)),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception as e:
        logger.error(f"Failed to fetch funding for {asset}: {e}")
        return pd.DataFrame()


def collect_all_funding(
    lookback_days: int = 730,
    output_dir: str = "data/datasets",
) -> dict[str, Path]:
    """Collect funding rate history for all assets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (lookback_days * 86400 * 1000)

    results = {}
    for asset in ALLOWED_ASSETS:
        logger.info(f"Fetching funding history for {asset}...")
        df = fetch_funding_history(asset, start_ms, end_ms)

        if not df.empty:
            file_path = output_path / f"{asset}_funding_{lookback_days}d.parquet"
            df.to_parquet(file_path, index=False)
            results[asset] = file_path
            logger.info(f"Saved {len(df)} funding records for {asset}")

        time.sleep(0.5)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = collect_all_funding()
    logger.info(f"Collected funding data for {len(results)} assets")
