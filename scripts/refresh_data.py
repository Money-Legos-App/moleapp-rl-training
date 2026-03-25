"""
Incremental Data Refresh — Fetch Recent Months & Merge with R2 Archive
========================================================================
Downloads only the last N months of Binance data, merges with existing
parquets (from R2 or local), rebuilds episodes, and re-uploads.

Unlike binance_ingest.py (which re-downloads 2 years every time), this
script is designed to run periodically (weekly/monthly) and only fetches
what's new.

Usage:
    # Refresh last 1 month for all assets, merge with R2 data, re-upload
    python scripts/refresh_data.py --months 1

    # Refresh last 3 months, specific assets only
    python scripts/refresh_data.py --months 3 --assets BTC ETH SOL

    # Refresh locally only (no R2 download/upload)
    python scripts/refresh_data.py --months 1 --local-only

    # Dry run — show what would be fetched without downloading
    python scripts/refresh_data.py --months 1 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default lookback that matches existing parquet filenames
ARCHIVE_LOOKBACK_DAYS = 730


def _months_to_fetch(n_months: int) -> list[tuple[int, int]]:
    """Generate (year, month) tuples for the last n_months, most recent first."""
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    months = []

    for _ in range(n_months):
        months.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1

    return months


def _download_recent_months(
    asset: str,
    months: list[tuple[int, int]],
    raw_dir: str,
    output_dir: str,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch klines and funding for specific months only.

    Returns (klines_df, funding_df) — either can be None if no data.
    """
    from data.collectors.asset_config import (
        BINANCE_LISTING_DATES,
        BINANCE_SYMBOL_MAP,
    )
    from data.collectors.binance_vision_collector import (
        _build_funding_url,
        _build_kline_url,
        _clamp_extreme_wicks,
        _download_and_extract_zip,
        _fill_maintenance_gaps,
        _get_retry_session,
        _parse_funding_csv,
        _parse_kline_csv,
    )

    symbol = BINANCE_SYMBOL_MAP.get(asset)
    if not symbol:
        logger.error(f"No Binance symbol for {asset}")
        return None, None

    listing = BINANCE_LISTING_DATES.get(symbol, "2019-01")
    listing_year, listing_month = map(int, listing.split("-"))

    session = _get_retry_session()
    raw_path = Path(raw_dir)

    kline_dfs = []
    funding_dfs = []

    for year, month in months:
        if (year, month) < (listing_year, listing_month):
            continue

        # Klines
        url = _build_kline_url(symbol, "1h", year, month)
        logger.info(f"  {asset} klines {year}-{month:02d}")
        csv_bytes = _download_and_extract_zip(url, session, raw_path)
        if csv_bytes is not None:
            try:
                df = _parse_kline_csv(csv_bytes)
                if not df.empty:
                    kline_dfs.append(df)
            except Exception as e:
                logger.error(f"  Parse error {asset} klines {year}-{month:02d}: {e}")

        # Funding
        url = _build_funding_url(symbol, year, month)
        logger.info(f"  {asset} funding {year}-{month:02d}")
        csv_bytes = _download_and_extract_zip(url, session, raw_path)
        if csv_bytes is not None:
            try:
                df = _parse_funding_csv(csv_bytes)
                if not df.empty:
                    funding_dfs.append(df)
            except Exception as e:
                logger.error(f"  Parse error {asset} funding {year}-{month:02d}: {e}")

        time.sleep(0.3)

    klines = pd.concat(kline_dfs, ignore_index=True) if kline_dfs else None
    funding = pd.concat(funding_dfs, ignore_index=True) if funding_dfs else None

    # Clean klines
    if klines is not None:
        klines = klines.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        klines = _fill_maintenance_gaps(klines, max_gap_hours=3)
        klines = _clamp_extreme_wicks(klines, max_wick_pct=0.40)
        # Drop maintenance column
        klines = klines.drop(columns=["maintenance"], errors="ignore")

    if funding is not None:
        funding = funding.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return klines, funding


def _merge_with_archive(
    new_df: pd.DataFrame,
    archive_path: Path,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Merge new data with existing archive parquet.

    New data overwrites archive rows for overlapping timestamps
    (handles Binance revisions to recent months).
    """
    if not archive_path.exists():
        logger.info(f"  No archive at {archive_path.name}, using new data only")
        return new_df

    archive = pd.read_parquet(archive_path)
    logger.info(f"  Archive: {len(archive)} rows, new: {len(new_df)} rows")

    # Remove archive rows that overlap with new data's time range
    new_min = new_df[timestamp_col].min()
    new_max = new_df[timestamp_col].max()

    archive_keep = archive[
        (archive[timestamp_col] < new_min) | (archive[timestamp_col] > new_max)
    ]

    merged = pd.concat([archive_keep, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    added = len(merged) - len(archive)
    logger.info(f"  Merged: {len(merged)} rows ({'+' if added >= 0 else ''}{added} net)")

    return merged


def refresh_asset(
    asset: str,
    months: list[tuple[int, int]],
    raw_dir: str,
    data_dir: str,
    lookback_days: int = ARCHIVE_LOOKBACK_DAYS,
) -> dict[str, bool]:
    """Refresh a single asset: download recent months, merge with archive."""
    from data.validators.validate_hl_data import validate_dataset, validate_funding_data

    result = {"klines": False, "funding": False}
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Refreshing {asset}...")

    new_klines, new_funding = _download_recent_months(asset, months, raw_dir, data_dir)

    # Merge klines
    if new_klines is not None and not new_klines.empty:
        archive_path = data_path / f"{asset}_1h_{lookback_days}d.parquet"
        merged = _merge_with_archive(new_klines, archive_path)

        # Validate
        validation = validate_dataset(merged, asset=asset, timeframe="1h")
        if validation.passed:
            merged.to_parquet(archive_path, index=False)
            logger.info(f"  Saved {len(merged)} klines to {archive_path.name}")
            result["klines"] = True
        else:
            logger.error(f"  Validation FAILED for {asset} klines:\n{validation.summary()}")
    else:
        logger.warning(f"  No new kline data for {asset}")

    # Merge funding
    if new_funding is not None and not new_funding.empty:
        archive_path = data_path / f"{asset}_funding_{lookback_days}d.parquet"
        merged = _merge_with_archive(new_funding, archive_path)

        validation = validate_funding_data(merged, asset=asset)
        if validation.errors:
            logger.warning(f"  Funding warnings for {asset}: {len(validation.errors)} issues")

        merged.to_parquet(archive_path, index=False)
        logger.info(f"  Saved {len(merged)} funding records to {archive_path.name}")
        result["funding"] = True
    else:
        logger.warning(f"  No new funding data for {asset}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Incremental data refresh — fetch recent months and merge with archive"
    )
    parser.add_argument(
        "--months", type=int, default=1,
        help="Number of recent months to fetch (default: 1)",
    )
    parser.add_argument(
        "--assets", nargs="*", default=None,
        help="Specific assets to refresh (default: all 15)",
    )
    parser.add_argument("--raw-dir", default="data/raw/binance")
    parser.add_argument("--data-dir", default="data/datasets")
    parser.add_argument("--episode-dir", default="data/episodes")
    parser.add_argument(
        "--local-only", action="store_true",
        help="Skip R2 download/upload (use local parquets only)",
    )
    parser.add_argument(
        "--skip-episodes", action="store_true",
        help="Skip episode rebuilding after merge",
    )
    parser.add_argument(
        "--skip-r2-upload", action="store_true",
        help="Download from R2 but don't re-upload after merge",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be fetched without downloading",
    )

    args = parser.parse_args()

    from data.collectors.asset_config import ALLOWED_ASSETS

    target_assets = args.assets or ALLOWED_ASSETS
    months = _months_to_fetch(args.months)

    month_str = ", ".join(f"{y}-{m:02d}" for y, m in months)
    logger.info(f"Refresh plan: {len(target_assets)} assets × {len(months)} months ({month_str})")

    if args.dry_run:
        for asset in target_assets:
            logger.info(f"  Would fetch: {asset} for {month_str}")
        logger.info("Dry run complete. No data downloaded.")
        return

    # ── Step 1: Pull existing archive from R2 (if not local-only) ────
    if not args.local_only:
        logger.info("=" * 60)
        logger.info("STEP 1: Pulling current archive from R2")
        logger.info("=" * 60)

        from scripts.r2_sync import R2_PREFIXES, download_data

        download_data(data_dir=args.data_dir, prefix=R2_PREFIXES["processed"])
    else:
        logger.info("STEP 1: Skipped R2 download (--local-only)")

    # ── Step 2: Fetch recent months and merge ────────────────────────
    logger.info("=" * 60)
    logger.info(f"STEP 2: Fetching {args.months} recent month(s) from Binance")
    logger.info("=" * 60)

    results = {}
    for asset in target_assets:
        results[asset] = refresh_asset(
            asset=asset,
            months=months,
            raw_dir=args.raw_dir,
            data_dir=args.data_dir,
        )
        time.sleep(0.5)

    # ── Step 3: Re-derive OI ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Re-deriving Open Interest")
    logger.info("=" * 60)

    from data.collectors.hl_oi_collector import derive_all_assets

    derive_all_assets(
        ohlcv_dir=args.data_dir,
        funding_dir=args.data_dir,
        output_dir=args.data_dir,
        lookback_days=ARCHIVE_LOOKBACK_DAYS,
    )

    # ── Step 4: Rebuild episodes ─────────────────────────────────────
    if not args.skip_episodes:
        logger.info("=" * 60)
        logger.info("STEP 4: Rebuilding episodes")
        logger.info("=" * 60)

        from data.preprocessors.episode_builder import build_all_assets

        episodes = build_all_assets(
            data_dir=args.data_dir,
            output_dir=args.episode_dir,
            lookback_days=ARCHIVE_LOOKBACK_DAYS,
        )
        logger.info(f"Built episodes for {len(episodes)} assets")
    else:
        logger.info("STEP 4: Skipped episode rebuild (--skip-episodes)")

    # ── Step 5: Re-upload to R2 ──────────────────────────────────────
    if not args.local_only and not args.skip_r2_upload:
        logger.info("=" * 60)
        logger.info("STEP 5: Uploading refreshed data to R2")
        logger.info("=" * 60)

        from scripts.r2_sync import R2_PREFIXES, upload_data

        upload_data(
            data_dir=args.data_dir,
            prefix=R2_PREFIXES["processed"],
        )

        if not args.skip_episodes:
            upload_data(
                data_dir=args.episode_dir,
                prefix=R2_PREFIXES["episodes"],
                recursive=True,
            )
    else:
        logger.info("STEP 5: Skipped R2 upload")

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("REFRESH COMPLETE")
    logger.info("=" * 60)

    ok_k = sum(1 for r in results.values() if r["klines"])
    ok_f = sum(1 for r in results.values() if r["funding"])
    total = len(results)

    for asset, r in sorted(results.items()):
        k = "OK" if r["klines"] else "--"
        f = "OK" if r["funding"] else "--"
        logger.info(f"  {asset:10s}  klines={k}  funding={f}")

    logger.info(f"\nRefreshed: {ok_k}/{total} klines, {ok_f}/{total} funding")
    logger.info(f"Months fetched: {month_str}")

    if ok_k < total:
        logger.warning(
            f"{total - ok_k} assets had no new kline data — this is normal "
            f"if the current month's ZIP isn't published yet on Binance Vision"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
