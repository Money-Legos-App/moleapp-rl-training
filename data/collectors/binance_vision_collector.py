"""
Binance Vision OHLCV + Funding Rate Collector
===============================================
Downloads pre-packaged monthly CSV/ZIP dumps from Binance Vision
(data.binance.vision) for USDⓈ-M perpetuals.  Stitches monthly chunks
into continuous 2-year timelines, validates via the existing gauntlet,
and saves parquet files identical to HL collector output.

Usage:
    python -m data.collectors.binance_vision_collector
    python -m data.collectors.binance_vision_collector --assets BTC ETH --lookback-days 365
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data.collectors.asset_config import (
    ALLOWED_ASSETS,
    BINANCE_LISTING_DATES,
    BINANCE_SYMBOL_MAP,
)
from data.validators.validate_hl_data import validate_dataset, validate_funding_data

logger = logging.getLogger(__name__)

BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/monthly"

# Binance kline CSVs have 12 columns, no header
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


# ──────────────────────────────────────────────────────────────────────
# HTTP session with retry
# ──────────────────────────────────────────────────────────────────────

def _get_retry_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    timeout: int = 60,
) -> requests.Session:
    """Create a requests session with exponential backoff retry."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.timeout = timeout
    return session


# ──────────────────────────────────────────────────────────────────────
# Month range generation
# ──────────────────────────────────────────────────────────────────────

def _generate_month_range(lookback_days: int = 730) -> list[tuple[int, int]]:
    """
    Generate (year, month) tuples covering lookback_days from today.

    Returns most-recent month first (descending) so we can stop early
    for assets that listed recently.
    """
    now = datetime.now(timezone.utc)
    months = []

    # Start from current month, go backwards
    year, month = now.year, now.month
    start_ts = now.timestamp() - (lookback_days * 86400)
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    start_year, start_month = start_dt.year, start_dt.month

    while (year, month) >= (start_year, start_month):
        months.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1

    return months  # Most recent first


# ──────────────────────────────────────────────────────────────────────
# Download & extract
# ──────────────────────────────────────────────────────────────────────

def _build_kline_url(symbol: str, interval: str, year: int, month: int) -> str:
    return (
        f"{BINANCE_VISION_BASE}/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{year}-{month:02d}.zip"
    )


def _build_funding_url(symbol: str, year: int, month: int) -> str:
    return (
        f"{BINANCE_VISION_BASE}/fundingRate/{symbol}/"
        f"{symbol}-fundingRate-{year}-{month:02d}.zip"
    )


def _download_and_extract_zip(
    url: str,
    session: requests.Session,
    raw_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Download a ZIP from Binance Vision, extract the CSV inside.

    Returns None on 404 (asset not listed yet for that month).
    Saves raw ZIP to raw_dir if provided.
    """
    try:
        resp = session.get(url, timeout=60)

        if resp.status_code == 404:
            return None  # Not listed yet for this month

        resp.raise_for_status()

        # Save raw ZIP as backup
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            zip_name = url.split("/")[-1]
            (raw_dir / zip_name).write_bytes(resp.content)

        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                logger.warning(f"No CSV found in {url}")
                return None

            with zf.open(csv_names[0]) as csv_file:
                # Try reading without header first (standard Binance format)
                csv_bytes = csv_file.read()

        return csv_bytes

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error downloading {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# CSV Parsing
# ──────────────────────────────────────────────────────────────────────

def _parse_kline_csv(csv_bytes: bytes) -> pd.DataFrame:
    """
    Parse Binance kline CSV bytes into a clean OHLCV DataFrame.

    Binance kline CSVs have 12 columns with no header row.
    Output matches HL collector schema: [timestamp, open, high, low, close, volume]
    """
    # Detect if CSV has a header by peeking at the first line
    first_line = csv_bytes.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    has_header = not first_line.split(",")[0].strip().isdigit()

    if has_header:
        df = pd.read_csv(io.BytesIO(csv_bytes), header=0)
        # Normalize column names to our standard
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if "open_time" in col_lower or col_lower == "open time":
                col_map[col] = "open_time"
            elif col_lower == "open":
                col_map[col] = "open"
            elif col_lower == "high":
                col_map[col] = "high"
            elif col_lower == "low":
                col_map[col] = "low"
            elif col_lower == "close" and "close_time" not in col_lower:
                col_map[col] = "close"
            elif col_lower == "volume":
                col_map[col] = "volume"
        df = df.rename(columns=col_map)
    else:
        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            header=None,
            names=KLINE_COLUMNS,
        )

    # Convert timestamp from milliseconds
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    # Extract OHLCV columns and cast types
    result = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")

    return result


def _parse_funding_csv(csv_bytes: bytes) -> pd.DataFrame:
    """
    Parse Binance funding rate CSV bytes.

    Binance funding CSVs have variable columns across time periods.
    We extract calc_time → timestamp, last_funding_rate → funding_rate.
    Output matches HL collector schema: [timestamp, funding_rate, premium]
    """
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Normalize column names (Binance uses various formats)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "calc_time" in col_lower or "time" in col_lower:
            col_map[col] = "calc_time"
        elif "funding" in col_lower and "rate" in col_lower and "interval" not in col_lower:
            col_map[col] = "funding_rate"
        elif "mark" in col_lower and "price" in col_lower:
            col_map[col] = "mark_price"

    df = df.rename(columns=col_map)

    if "calc_time" not in df.columns:
        # Fall back to first column as timestamp
        df = df.rename(columns={df.columns[0]: "calc_time"})

    if "funding_rate" not in df.columns:
        raise ValueError(f"No funding rate column found. Columns: {list(df.columns)}")

    # Convert timestamp
    calc_time = df["calc_time"]
    if pd.api.types.is_numeric_dtype(calc_time):
        df["timestamp"] = pd.to_datetime(calc_time, unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(calc_time)

    # Build output
    result = pd.DataFrame({
        "timestamp": df["timestamp"],
        "funding_rate": pd.to_numeric(df["funding_rate"], errors="coerce").astype("float64"),
        "premium": 0.0,  # Binance Vision doesn't include premium
    })

    return result


# ──────────────────────────────────────────────────────────────────────
# Gap filling for maintenance windows
# ──────────────────────────────────────────────────────────────────────

def _fill_maintenance_gaps(
    df: pd.DataFrame,
    max_gap_hours: int = 3,
) -> pd.DataFrame:
    """
    Forward-fill small gaps (≤max_gap_hours) in OHLCV data.

    Filled candles carry forward OHLC from last known candle with volume=0
    and get a 'maintenance' column set to True.
    Gaps larger than max_gap_hours are left for the validator to flag.
    """
    if df.empty or len(df) < 2:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    expected_delta = pd.Timedelta(hours=1)
    max_gap = pd.Timedelta(hours=max_gap_hours)

    filled_rows = []
    for i in range(len(df) - 1):
        filled_rows.append(df.iloc[i])
        gap = df.iloc[i + 1]["timestamp"] - df.iloc[i]["timestamp"]

        if expected_delta < gap <= max_gap:
            # Fill the gap
            n_missing = int(gap / expected_delta) - 1
            last_close = df.iloc[i]["close"]
            for j in range(1, n_missing + 1):
                fill_ts = df.iloc[i]["timestamp"] + expected_delta * j
                filled_rows.append(pd.Series({
                    "timestamp": fill_ts,
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                    "volume": 0.0,
                    "maintenance": True,
                }))

    # Add the last row
    filled_rows.append(df.iloc[-1])

    result = pd.DataFrame(filled_rows).reset_index(drop=True)

    # Ensure maintenance column exists (False for non-filled rows)
    if "maintenance" not in result.columns:
        result["maintenance"] = False
    else:
        result["maintenance"] = result["maintenance"].fillna(False)

    return result


# ──────────────────────────────────────────────────────────────────────
# Wick clamping for extreme liquidation cascades
# ──────────────────────────────────────────────────────────────────────

def _clamp_extreme_wicks(
    df: pd.DataFrame,
    max_wick_pct: float = 0.40,
) -> pd.DataFrame:
    """
    Clamp candles where high-low range exceeds max_wick_pct of close.

    Real market events (liquidation cascades) can produce 70-90% wicks on
    smaller-cap assets.  These are valid data but blow up the validator's
    outlier_wick check and distort RL training.  We cap the wick to
    max_wick_pct while preserving the open→close directional move.

    Returns a copy of the DataFrame with clamped high/low values.
    """
    df = df.copy()
    close = df["close"].clip(lower=0.01)
    wick_ratio = (df["high"] - df["low"]) / close

    mask = wick_ratio > max_wick_pct
    n_clamped = mask.sum()

    if n_clamped > 0:
        # For each extreme candle, set high/low symmetrically around midpoint
        mid = (df.loc[mask, "open"] + df.loc[mask, "close"]) / 2
        half_range = close[mask] * max_wick_pct / 2
        df.loc[mask, "high"] = mid + half_range
        df.loc[mask, "low"] = mid - half_range

        # Ensure high >= max(open, close) and low <= min(open, close)
        oc_max = df.loc[mask, ["open", "close"]].max(axis=1)
        oc_min = df.loc[mask, ["open", "close"]].min(axis=1)
        df.loc[mask, "high"] = df.loc[mask, "high"].clip(lower=oc_max)
        df.loc[mask, "low"] = df.loc[mask, "low"].clip(upper=oc_min)

        logger.info(f"Clamped {n_clamped} extreme wicks (>{max_wick_pct*100:.0f}% of price)")

    return df


# ──────────────────────────────────────────────────────────────────────
# Main collection functions
# ──────────────────────────────────────────────────────────────────────

def fetch_klines(
    asset: str,
    lookback_days: int = 730,
    raw_dir: str = "data/raw/binance",
    output_dir: str = "data/datasets",
) -> Optional[Path]:
    """
    Fetch 1h kline history for an asset from Binance Vision.

    Downloads monthly ZIPs, stitches into continuous timeline,
    fills small gaps, validates, saves as parquet.

    Returns path to saved parquet file, or None on failure.
    """
    symbol = BINANCE_SYMBOL_MAP.get(asset)
    if not symbol:
        logger.error(f"No Binance symbol mapping for asset: {asset}")
        return None

    session = _get_retry_session()
    months = _generate_month_range(lookback_days)
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check listing date to avoid unnecessary 404s
    listing = BINANCE_LISTING_DATES.get(symbol, "2019-01")
    listing_year, listing_month = map(int, listing.split("-"))

    all_dfs = []
    skipped = 0

    for year, month in months:
        # Skip months before listing
        if (year, month) < (listing_year, listing_month):
            skipped += 1
            continue

        url = _build_kline_url(symbol, "1h", year, month)
        logger.info(f"Fetching {asset} klines: {year}-{month:02d}")

        csv_bytes = _download_and_extract_zip(url, session, raw_path)

        if csv_bytes is None:
            logger.warning(f"No data for {asset} {year}-{month:02d} (404 or empty)")
            continue

        try:
            df = _parse_kline_csv(csv_bytes)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to parse {asset} {year}-{month:02d}: {e}")

        time.sleep(0.3)  # Rate limit courtesy

    if skipped > 0:
        logger.info(f"Skipped {skipped} months before {asset} listing date ({listing})")

    if not all_dfs:
        logger.warning(f"No kline data collected for {asset}")
        return None

    # Stitch monthly chunks
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    full_df = full_df.reset_index(drop=True)

    logger.info(f"{asset}: {len(full_df)} raw candles stitched")

    # Fill small maintenance gaps
    full_df = _fill_maintenance_gaps(full_df, max_gap_hours=3)
    n_filled = full_df["maintenance"].sum() if "maintenance" in full_df.columns else 0
    if n_filled > 0:
        logger.info(f"{asset}: {n_filled} maintenance-gap candles filled")

    # Clamp extreme wicks (liquidation cascades)
    full_df = _clamp_extreme_wicks(full_df, max_wick_pct=0.40)

    # Drop maintenance column before validation (validator expects standard OHLCV)
    validate_df = full_df.drop(columns=["maintenance"], errors="ignore")

    # Validate
    validation = validate_dataset(validate_df, asset=asset, timeframe="1h")
    logger.info(
        f"Validation for {asset}: {len(validation.errors)} errors, "
        f"{len(validation.warnings)} warnings"
    )

    if not validation.passed:
        logger.error(f"DATA VALIDATION FAILED for {asset}:\n{validation.summary()}")
        raise ValueError(
            f"Data validation failed for {asset} 1h. "
            f"{len(validation.errors)} errors found. "
            f"Fix the data or investigate the timestamps above before training."
        )

    if validation.warnings:
        logger.warning(f"Validation warnings for {asset}:\n{validation.summary()}")

    # Save (without maintenance column — EpisodeBuilder expects standard OHLCV)
    file_path = output_path / f"{asset}_1h_{lookback_days}d.parquet"
    validate_df.to_parquet(file_path, index=False)
    logger.info(f"Saved {len(validate_df)} validated candles to {file_path}")

    return file_path


def fetch_funding(
    asset: str,
    lookback_days: int = 730,
    raw_dir: str = "data/raw/binance",
    output_dir: str = "data/datasets",
) -> Optional[Path]:
    """
    Fetch funding rate history for an asset from Binance Vision.

    Downloads monthly ZIPs, stitches, validates, saves as parquet.
    Funding rates are kept at native 8h granularity (EpisodeBuilder
    handles forward-fill to 1h via merge_asof).

    Returns path to saved parquet file, or None on failure.
    """
    symbol = BINANCE_SYMBOL_MAP.get(asset)
    if not symbol:
        logger.error(f"No Binance symbol mapping for asset: {asset}")
        return None

    session = _get_retry_session()
    months = _generate_month_range(lookback_days)
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    listing = BINANCE_LISTING_DATES.get(symbol, "2019-01")
    listing_year, listing_month = map(int, listing.split("-"))

    all_dfs = []

    for year, month in months:
        if (year, month) < (listing_year, listing_month):
            continue

        url = _build_funding_url(symbol, year, month)
        logger.info(f"Fetching {asset} funding: {year}-{month:02d}")

        csv_bytes = _download_and_extract_zip(url, session, raw_path)

        if csv_bytes is None:
            logger.warning(f"No funding data for {asset} {year}-{month:02d}")
            continue

        try:
            df = _parse_funding_csv(csv_bytes)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to parse funding {asset} {year}-{month:02d}: {e}")

        time.sleep(0.3)

    if not all_dfs:
        logger.warning(f"No funding data collected for {asset}")
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    full_df = full_df.reset_index(drop=True)

    logger.info(f"{asset}: {len(full_df)} funding records stitched")

    # Validate
    validation = validate_funding_data(full_df, asset=asset)
    if validation.errors:
        logger.error(f"Funding validation FAILED for {asset}:\n{validation.summary()}")
    if validation.warnings:
        logger.warning(f"Funding warnings for {asset}:\n{validation.summary()}")

    file_path = output_path / f"{asset}_funding_{lookback_days}d.parquet"
    full_df.to_parquet(file_path, index=False)
    logger.info(f"Saved {len(full_df)} funding records to {file_path}")

    return file_path


def collect_all_assets(
    assets: Optional[list[str]] = None,
    lookback_days: int = 730,
    raw_dir: str = "data/raw/binance",
    output_dir: str = "data/datasets",
) -> dict[str, dict[str, Optional[Path]]]:
    """
    Collect klines and funding for all (or specified) assets.

    Returns dict mapping asset → {"klines": Path, "funding": Path}.
    Continues on per-asset failures.
    """
    target_assets = assets or ALLOWED_ASSETS
    results: dict[str, dict[str, Optional[Path]]] = {}

    for asset in target_assets:
        logger.info(f"\n{'='*60}\nCollecting {asset}\n{'='*60}")
        result: dict[str, Optional[Path]] = {"klines": None, "funding": None}

        try:
            result["klines"] = fetch_klines(
                asset=asset,
                lookback_days=lookback_days,
                raw_dir=raw_dir,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error(f"FAILED klines for {asset}: {e}")

        try:
            result["funding"] = fetch_funding(
                asset=asset,
                lookback_days=lookback_days,
                raw_dir=raw_dir,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error(f"FAILED funding for {asset}: {e}")

        results[asset] = result
        time.sleep(0.5)

    # Summary
    success_klines = sum(1 for r in results.values() if r["klines"])
    success_funding = sum(1 for r in results.values() if r["funding"])
    logger.info(
        f"\nCollection complete: {success_klines}/{len(target_assets)} klines, "
        f"{success_funding}/{len(target_assets)} funding"
    )

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Binance Vision data collector")
    parser.add_argument(
        "--assets", nargs="*", default=None,
        help="Assets to collect (default: all 15)",
    )
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--raw-dir", default="data/raw/binance")
    parser.add_argument("--output-dir", default="data/datasets")
    args = parser.parse_args()

    results = collect_all_assets(
        assets=args.assets,
        lookback_days=args.lookback_days,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
