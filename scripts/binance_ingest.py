"""
Binance Vision Full Ingestion Pipeline
========================================
One-command pipeline: Download → Validate → Save → Derive OI → R2 Upload → W&B Artifact

Usage:
    # Full pipeline for all 15 assets
    python scripts/binance_ingest.py --all

    # Specific assets, skip cloud uploads
    python scripts/binance_ingest.py --assets BTC ETH SOL --skip-r2 --skip-wandb

    # Just download + validate (no episodes, no cloud)
    python scripts/binance_ingest.py --assets BTC --skip-r2 --skip-wandb --skip-episodes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Binance Vision full ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Collect all 15 assets")
    group.add_argument("--assets", nargs="+", help="Specific assets to collect")

    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--raw-dir", default="data/raw/binance")
    parser.add_argument("--output-dir", default="data/datasets")
    parser.add_argument("--episode-dir", default="data/episodes")
    parser.add_argument("--skip-r2", action="store_true", help="Skip R2 upload")
    parser.add_argument("--skip-wandb", action="store_true", help="Skip W&B artifact push")
    parser.add_argument("--skip-episodes", action="store_true", help="Skip episode building")
    parser.add_argument("--dataset-version", default="1.0.0", help="Version for W&B artifact")

    args = parser.parse_args()

    assets = None if args.all else args.assets

    # ── Step 1: Download + Validate + Save ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading from Binance Vision")
    logger.info("=" * 60)

    from data.collectors.binance_vision_collector import collect_all_assets

    results = collect_all_assets(
        assets=assets,
        lookback_days=args.lookback_days,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )

    success_klines = sum(1 for r in results.values() if r["klines"])
    success_funding = sum(1 for r in results.values() if r["funding"])
    total = len(results)

    logger.info(f"Downloaded: {success_klines}/{total} klines, {success_funding}/{total} funding")

    if success_klines == 0:
        logger.error("No kline data collected. Aborting.")
        sys.exit(1)

    # ── Step 2: Derive OI from OHLCV + Funding ─────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Deriving Open Interest")
    logger.info("=" * 60)

    from data.collectors.hl_oi_collector import derive_all_assets

    derive_all_assets(
        ohlcv_dir=args.output_dir,
        funding_dir=args.output_dir,
        output_dir=args.output_dir,
        lookback_days=args.lookback_days,
    )

    # ── Step 3: Upload to R2 ────────────────────────────────────────
    if not args.skip_r2:
        logger.info("=" * 60)
        logger.info("STEP 3: Uploading to Cloudflare R2")
        logger.info("=" * 60)

        from scripts.r2_sync import R2_PREFIXES, upload_data

        # Upload raw CSV backups
        raw_path = Path(args.raw_dir)
        if raw_path.exists() and any(raw_path.rglob("*.zip")):
            logger.info("Uploading raw ZIPs to R2 raw/ prefix...")
            upload_data(
                data_dir=args.raw_dir,
                prefix=R2_PREFIXES["raw"],
                file_glob="*.zip",
            )

        # Upload validated parquets
        logger.info("Uploading processed parquets to R2 processed/1h/ prefix...")
        upload_data(
            data_dir=args.output_dir,
            prefix=R2_PREFIXES["processed"],
        )
    else:
        logger.info("STEP 3: Skipped R2 upload (--skip-r2)")

    # ── Step 4: Build Episodes ──────────────────────────────────────
    if not args.skip_episodes:
        logger.info("=" * 60)
        logger.info("STEP 4: Building episodes")
        logger.info("=" * 60)

        from data.preprocessors.episode_builder import build_all_assets

        episodes = build_all_assets(
            data_dir=args.output_dir,
            output_dir=args.episode_dir,
            lookback_days=args.lookback_days,
        )
        logger.info(f"Built episodes for {len(episodes)} assets")
    else:
        logger.info("STEP 4: Skipped episode building (--skip-episodes)")

    # ── Step 5: Push W&B Artifact ───────────────────────────────────
    if not args.skip_wandb and not args.skip_episodes:
        logger.info("=" * 60)
        logger.info("STEP 5: Pushing W&B artifact")
        logger.info("=" * 60)

        from scripts.wandb_dataset_push import push_episodes

        artifact_name = push_episodes(
            data_dir=args.episode_dir,
            version=args.dataset_version,
            description=f"Binance Vision {args.lookback_days}d dataset",
        )
        logger.info(f"W&B artifact: {artifact_name}")
    else:
        logger.info("STEP 5: Skipped W&B push")

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    for asset, r in sorted(results.items()):
        k = "OK" if r["klines"] else "FAIL"
        f = "OK" if r["funding"] else "FAIL"
        logger.info(f"  {asset:10s}  klines={k}  funding={f}")

    logger.info(f"\nTotal: {success_klines}/{total} klines, {success_funding}/{total} funding")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
