"""
Hyperliquid Open Interest Historical Data Collector
=====================================================
Pulls open interest snapshots via HL REST API.

HL does not expose a dedicated OI history endpoint — OI is available as a
point-in-time field in `meta` and `metaAndAssetCtxs`.  To build a historical
series we:
  1. Capture current OI snapshots on every run (intended as a cron job).
  2. Merge snapshots into a growing parquet file per asset.

For backfill (pre-cron), we derive approximate OI from candle volume and
funding rate magnitude as a proxy (clearly flagged as `source="derived"`).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from data.collectors.asset_config import ALLOWED_ASSETS
from data.collectors.hl_ohlcv_collector import HL_INFO_URL

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Live OI Snapshot (run as cron every 15 min)
# ──────────────────────────────────────────────────────────────────────

def fetch_current_oi() -> dict[str, dict]:
    """
    Fetch current open interest for all assets from HL meta endpoint.

    Returns:
        {asset: {"open_interest": float, "mark_px": float, "funding_rate": float}}
    """
    payload = {"type": "metaAndAssetCtxs"}

    try:
        resp = requests.post(HL_INFO_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # data is [meta_info, [asset_ctx, ...]]
        if not data or len(data) < 2:
            logger.warning("Unexpected metaAndAssetCtxs response format")
            return {}

        meta = data[0]
        asset_ctxs = data[1]
        universe = meta.get("universe", [])

        result = {}
        for i, ctx in enumerate(asset_ctxs):
            if i >= len(universe):
                break
            asset_name = universe[i].get("name", "")
            if asset_name not in ALLOWED_ASSETS:
                continue

            result[asset_name] = {
                "open_interest": float(ctx.get("openInterest", 0)),
                "mark_px": float(ctx.get("markPx", 0)),
                "funding_rate": float(ctx.get("funding", 0)),
                "premium": float(ctx.get("premium", 0)),
                "day_ntl_vlm": float(ctx.get("dayNtlVlm", 0)),
            }

        return result

    except Exception as e:
        logger.error(f"Failed to fetch current OI: {e}")
        return {}


def snapshot_and_append(output_dir: str = "data/datasets") -> int:
    """
    Take an OI snapshot for all assets and append to per-asset parquet files.

    Returns:
        Number of assets successfully recorded.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    oi_data = fetch_current_oi()
    if not oi_data:
        logger.warning("No OI data fetched")
        return 0

    now = pd.Timestamp.utcnow()
    count = 0

    for asset, data in oi_data.items():
        row = {
            "timestamp": now,
            "open_interest": data["open_interest"],
            "mark_px": data["mark_px"],
            "funding_rate": data["funding_rate"],
            "premium": data.get("premium", 0.0),
            "day_ntl_vlm": data.get("day_ntl_vlm", 0.0),
            "source": "live",
        }

        file_path = output_path / f"{asset}_oi_history.parquet"

        new_df = pd.DataFrame([row])

        if file_path.exists():
            existing = pd.read_parquet(file_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            combined.to_parquet(file_path, index=False)
        else:
            new_df.to_parquet(file_path, index=False)

        count += 1

    logger.info(f"OI snapshot recorded for {count} assets at {now}")
    return count


# ──────────────────────────────────────────────────────────────────────
# Derived OI Backfill (approximation from volume + funding)
# ──────────────────────────────────────────────────────────────────────

def derive_oi_from_candles_and_funding(
    asset: str,
    ohlcv_path: str | Path,
    funding_path: str | Path,
    output_dir: str = "data/datasets",
) -> Optional[Path]:
    """
    Derive approximate OI history from OHLCV volume and funding rates.

    Logic:
    - High volume + high abs(funding) → high OI
    - Uses exponential smoothing of volume as OI proxy
    - Clearly tagged as source="derived" (not live)

    This is used for training when we don't have real historical OI snapshots.
    The model learns relative OI patterns, not absolute values (VecNormalize
    handles the scaling anyway).

    Args:
        asset: Asset name (e.g., "BTC")
        ohlcv_path: Path to OHLCV parquet file
        funding_path: Path to funding history parquet file
        output_dir: Output directory for derived OI file

    Returns:
        Path to saved parquet file, or None on error.
    """
    try:
        ohlcv_df = pd.read_parquet(ohlcv_path)
        funding_df = pd.read_parquet(funding_path)
    except Exception as e:
        logger.error(f"Failed to read input files for {asset}: {e}")
        return None

    if ohlcv_df.empty:
        logger.warning(f"Empty OHLCV data for {asset}")
        return None

    # Ensure timestamp columns
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
    funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"])

    # Merge on nearest timestamp (funding is 8h, OHLCV is 1h)
    ohlcv_df = ohlcv_df.sort_values("timestamp")
    funding_df = funding_df.sort_values("timestamp")

    # Forward-fill funding rate to match OHLCV timestamps
    merged = pd.merge_asof(
        ohlcv_df,
        funding_df[["timestamp", "funding_rate"]],
        on="timestamp",
        direction="backward",
    )
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)

    # Derive OI proxy:
    # 1. Exponential moving average of volume (captures sustained activity)
    merged["vol_ema"] = merged["volume"].ewm(span=24, adjust=False).mean()

    # 2. Combine with funding magnitude (high funding = crowded positioning)
    merged["funding_intensity"] = merged["funding_rate"].abs() * 10000  # scale up

    # 3. OI proxy = volume_ema * (1 + funding_intensity)
    # This captures: high volume with strong funding = likely high OI
    merged["open_interest"] = merged["vol_ema"] * (1.0 + merged["funding_intensity"])

    # 4. Also compute rolling stats for the feature builder
    merged["oi_rolling_30d"] = merged["open_interest"].rolling(
        window=720, min_periods=1  # 30 days of 1h candles
    ).mean()

    result_df = merged[["timestamp", "open_interest", "funding_rate"]].copy()
    result_df["mark_px"] = merged["close"]
    result_df["source"] = "derived"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{asset}_oi_derived.parquet"
    result_df.to_parquet(file_path, index=False)

    logger.info(f"Derived OI history for {asset}: {len(result_df)} rows → {file_path}")
    return file_path


def derive_all_assets(
    ohlcv_dir: str = "data/datasets",
    funding_dir: str = "data/datasets",
    output_dir: str = "data/datasets",
    lookback_days: int = 730,
) -> dict[str, Path]:
    """Derive OI for all 15 assets from existing OHLCV + funding data."""
    results = {}
    for asset in ALLOWED_ASSETS:
        ohlcv_path = Path(ohlcv_dir) / f"{asset}_1h_{lookback_days}d.parquet"
        funding_path = Path(funding_dir) / f"{asset}_funding_{lookback_days}d.parquet"

        if not ohlcv_path.exists():
            logger.warning(f"Missing OHLCV file for {asset}: {ohlcv_path}")
            continue
        if not funding_path.exists():
            logger.warning(f"Missing funding file for {asset}: {funding_path}")
            continue

        path = derive_oi_from_candles_and_funding(
            asset=asset,
            ohlcv_path=ohlcv_path,
            funding_path=funding_path,
            output_dir=output_dir,
        )
        if path:
            results[asset] = path

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Hyperliquid OI Collector")
    parser.add_argument(
        "--mode",
        choices=["snapshot", "derive"],
        default="snapshot",
        help="'snapshot' for live OI, 'derive' for backfill from OHLCV+funding",
    )
    parser.add_argument("--output-dir", default="data/datasets")
    args = parser.parse_args()

    if args.mode == "snapshot":
        count = snapshot_and_append(output_dir=args.output_dir)
        logger.info(f"Snapshot complete: {count} assets")
    else:
        results = derive_all_assets(output_dir=args.output_dir)
        logger.info(f"Derived OI for {len(results)} assets")
