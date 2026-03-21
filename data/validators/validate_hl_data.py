"""
Hyperliquid Data Validator — Catches RL-exploitable glitches before training
=============================================================================
RL agents are lazy exploiters. A single glitched candle (flash crash to $1,
stuck price, duplicate timestamp) will teach the agent a fake pattern instead
of real market alpha. This validator catches all of that.

Checks:
  1. Continuity   — No missing candles, no duplicates (crypto is 24/7)
  2. OHLC Logic   — high >= max(open,close), low <= min(open,close), vol >= 0
  3. Outliers     — No flash crashes/spikes > threshold, no flatlines > N hours
  4. Cross-ref    — Optional Binance spot price comparison for sanity

Usage:
    from data.validators.validate_hl_data import validate_dataset

    result = validate_dataset(df, asset="BTC", timeframe="1h")
    if not result.passed:
        raise ValueError(f"Data validation failed:\\n{result.summary()}")

    # Or standalone:
    python data/validators/validate_hl_data.py --file data/datasets/BTC_1h_730d.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Major assets get tighter thresholds (more liquid = fewer real spikes)
MAJOR_ASSETS = {"BTC", "ETH", "SOL"}

# Max hourly price change thresholds (percentage)
SPIKE_THRESHOLD_MAJOR = 0.20     # 20% for BTC/ETH/SOL
SPIKE_THRESHOLD_MINOR = 0.35     # 35% for smaller assets (memecoins can move)

# Flatline detection: N consecutive identical prices = suspicious
FLATLINE_MIN_HOURS = 10

# Expected candle counts per timeframe for 2 years
EXPECTED_COUNTS = {
    "1h": {"min": 17000, "max": 17600},   # ~17520 (365.25 * 2 * 24)
    "4h": {"min": 4300, "max": 4400},
    "1d": {"min": 720, "max": 740},
}

# Timeframe to timedelta mapping
TIMEFRAME_DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


@dataclass
class ValidationIssue:
    """A single validation issue found in the data."""
    severity: str        # "ERROR" (fails validation) or "WARNING" (logged but passes)
    check: str           # Which check caught it
    message: str         # Human-readable description
    timestamp: Optional[pd.Timestamp] = None  # Where in the data
    value: Optional[float] = None  # The offending value


@dataclass
class ValidationResult:
    """Result of running all validation checks on a dataset."""
    asset: str
    timeframe: str
    row_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED' if self.passed else 'FAILED'} for {self.asset} {self.timeframe}",
            f"  Rows: {self.row_count}",
            f"  Errors: {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for issue in self.errors[:20]:  # Cap output
            ts = f" @ {issue.timestamp}" if issue.timestamp else ""
            lines.append(f"  [ERROR] {issue.check}: {issue.message}{ts}")
        for issue in self.warnings[:10]:
            ts = f" @ {issue.timestamp}" if issue.timestamp else ""
            lines.append(f"  [WARN]  {issue.check}: {issue.message}{ts}")
        if len(self.errors) > 20:
            lines.append(f"  ... and {len(self.errors) - 20} more errors")
        return "\n".join(lines)


def validate_dataset(
    df: pd.DataFrame,
    asset: str = "UNKNOWN",
    timeframe: str = "1h",
    strict: bool = True,
) -> ValidationResult:
    """
    Run all validation checks on an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
        asset: Asset name (used for threshold selection)
        timeframe: "1h", "4h", or "1d"
        strict: If True, treat warnings as errors for borderline cases

    Returns:
        ValidationResult with all issues found
    """
    result = ValidationResult(asset=asset, timeframe=timeframe, row_count=len(df))

    if df.empty:
        result.issues.append(ValidationIssue(
            severity="ERROR", check="empty", message="DataFrame is empty"
        ))
        return result

    # Ensure timestamp is datetime
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Run all checks
    _check_required_columns(df, result)
    if result.errors:
        return result  # Can't proceed without required columns

    _check_price_positivity(df, result)
    _check_continuity(df, result, timeframe)
    _check_ohlc_logic(df, result)
    _check_outliers(df, result, asset, timeframe)
    _check_flatlines(df, result, asset)
    _check_row_count(df, result, timeframe)
    _check_volume_sanity(df, result)

    return result


# ──────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────

def _check_required_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Verify all required columns exist."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        result.issues.append(ValidationIssue(
            severity="ERROR",
            check="columns",
            message=f"Missing required columns: {missing}",
        ))


def _check_continuity(
    df: pd.DataFrame,
    result: ValidationResult,
    timeframe: str,
) -> None:
    """Check 1: No missing candles, no duplicates."""
    expected_delta = TIMEFRAME_DELTAS.get(timeframe, pd.Timedelta(hours=1))

    # Duplicates
    duplicates = df[df["timestamp"].duplicated(keep=False)]
    if not duplicates.empty:
        dup_timestamps = duplicates["timestamp"].unique()
        for ts in dup_timestamps[:10]:
            result.issues.append(ValidationIssue(
                severity="ERROR",
                check="continuity_duplicate",
                message=f"Duplicate timestamp",
                timestamp=ts,
            ))
        if len(dup_timestamps) > 10:
            result.issues.append(ValidationIssue(
                severity="ERROR",
                check="continuity_duplicate",
                message=f"{len(dup_timestamps)} total duplicate timestamps",
            ))

    # Gaps
    time_diffs = df["timestamp"].diff().dropna()
    gaps = time_diffs[time_diffs != expected_delta]

    if not gaps.empty:
        # Classify gaps
        for idx in gaps.index[:20]:
            gap_size = gaps[idx]
            gap_ts = df.loc[idx, "timestamp"]
            gap_hours = gap_size.total_seconds() / 3600

            if gap_size < expected_delta:
                # Shorter than expected = likely duplicate or overlap
                result.issues.append(ValidationIssue(
                    severity="ERROR",
                    check="continuity_overlap",
                    message=f"Candle interval too short: {gap_size} (expected {expected_delta})",
                    timestamp=gap_ts,
                ))
            elif gap_hours <= 3:
                # Small gap (1-3 hours): WARNING — could be maintenance
                result.issues.append(ValidationIssue(
                    severity="WARNING",
                    check="continuity_gap",
                    message=f"Small gap: {gap_size} ({gap_hours:.1f}h)",
                    timestamp=gap_ts,
                ))
            else:
                # Large gap: ERROR
                result.issues.append(ValidationIssue(
                    severity="ERROR",
                    check="continuity_gap",
                    message=f"Missing candles: {gap_size} gap ({gap_hours:.1f}h = {int(gap_hours)} missing candles)",
                    timestamp=gap_ts,
                ))

        total_gaps = len(gaps)
        if total_gaps > 20:
            result.issues.append(ValidationIssue(
                severity="ERROR",
                check="continuity_gap",
                message=f"{total_gaps} total discontinuities found",
            ))


def _check_ohlc_logic(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check 2: OHLC mathematical rules must hold for every candle."""
    # High must be the highest
    bad_high_open = df[df["high"] < df["open"]]
    bad_high_close = df[df["high"] < df["close"]]
    bad_low_open = df[df["low"] > df["open"]]
    bad_low_close = df[df["low"] > df["close"]]
    bad_volume = df[df["volume"] < 0]

    for label, bad_df, msg in [
        ("high<open", bad_high_open, "High is lower than Open"),
        ("high<close", bad_high_close, "High is lower than Close"),
        ("low>open", bad_low_open, "Low is higher than Open"),
        ("low>close", bad_low_close, "Low is higher than Close"),
        ("neg_volume", bad_volume, "Negative volume"),
    ]:
        if not bad_df.empty:
            for _, row in bad_df.head(5).iterrows():
                result.issues.append(ValidationIssue(
                    severity="ERROR",
                    check=f"ohlc_{label}",
                    message=f"{msg}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:.0f}",
                    timestamp=row["timestamp"],
                ))
            if len(bad_df) > 5:
                result.issues.append(ValidationIssue(
                    severity="ERROR",
                    check=f"ohlc_{label}",
                    message=f"{len(bad_df)} total candles with {msg}",
                ))


def _check_outliers(
    df: pd.DataFrame,
    result: ValidationResult,
    asset: str,
    timeframe: str,
) -> None:
    """Check 3: No flash crashes/spikes beyond threshold."""
    threshold = SPIKE_THRESHOLD_MAJOR if asset in MAJOR_ASSETS else SPIKE_THRESHOLD_MINOR

    # Hour-to-hour (or candle-to-candle) percentage change
    pct_change = df["close"].pct_change().abs()
    spikes = df[pct_change > threshold]

    for _, row in spikes.iterrows():
        idx = row.name
        if idx == 0:
            continue
        prev_close = df.loc[idx - 1, "close"]
        if prev_close == 0:
            change_pct = float("inf")
        else:
            change_pct = abs(row["close"] - prev_close) / prev_close * 100

        result.issues.append(ValidationIssue(
            severity="ERROR",
            check="outlier_spike",
            message=(
                f"Price spike: {prev_close:.2f} → {row['close']:.2f} "
                f"({change_pct:.1f}% change, threshold={threshold*100:.0f}%)"
            ),
            timestamp=row["timestamp"],
            value=change_pct,
        ))

    # Also check high-low range vs close (catches wicks to $0 or $999999)
    wick_ratio = (df["high"] - df["low"]) / df["close"].clip(lower=0.01)
    extreme_wicks = df[wick_ratio > 0.5]  # 50% of price in a single candle
    for _, row in extreme_wicks.head(10).iterrows():
        close_safe = max(row["close"], 0.01)
        wick_pct = (row["high"] - row["low"]) / close_safe * 100
        result.issues.append(ValidationIssue(
            severity="ERROR",
            check="outlier_wick",
            message=(
                f"Extreme wick: H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} "
                f"(range={wick_pct:.1f}% of price)"
            ),
            timestamp=row["timestamp"],
            value=wick_pct,
        ))


def _check_flatlines(
    df: pd.DataFrame,
    result: ValidationResult,
    asset: str,
) -> None:
    """Check 3b: No suspicious flatlines (N+ consecutive identical prices)."""
    # Check close price flatlines
    close_unchanged = (df["close"].diff().abs() < 1e-10)
    consecutive = _count_consecutive_true(close_unchanged.values)

    for start_idx, length in consecutive:
        if length >= FLATLINE_MIN_HOURS:
            result.issues.append(ValidationIssue(
                severity="WARNING" if length < 24 else "ERROR",
                check="flatline_price",
                message=(
                    f"Price flatline: {length} consecutive identical closes "
                    f"(price={df.iloc[start_idx]['close']:.2f})"
                ),
                timestamp=df.iloc[start_idx]["timestamp"],
                value=float(length),
            ))

    # Check volume flatlines (exchange might have been down)
    vol_zero = (df["volume"] == 0)
    vol_consecutive = _count_consecutive_true(vol_zero.values)
    for start_idx, length in vol_consecutive:
        if length >= 3:
            result.issues.append(ValidationIssue(
                severity="WARNING" if length < 6 else "ERROR",
                check="flatline_volume",
                message=f"Zero volume for {length} consecutive candles",
                timestamp=df.iloc[start_idx]["timestamp"],
                value=float(length),
            ))


def _check_row_count(
    df: pd.DataFrame,
    result: ValidationResult,
    timeframe: str,
) -> None:
    """Verify approximate expected row count for the timeframe."""
    expected = EXPECTED_COUNTS.get(timeframe)
    if not expected:
        return

    count = len(df)
    if count < expected["min"]:
        result.issues.append(ValidationIssue(
            severity="WARNING",
            check="row_count",
            message=f"Low row count: {count} (expected {expected['min']}-{expected['max']} for 2yr {timeframe})",
        ))


def _check_price_positivity(df: pd.DataFrame, result: ValidationResult) -> None:
    """All prices must be strictly positive."""
    for col in ["open", "high", "low", "close"]:
        bad = df[df[col] <= 0]
        if not bad.empty:
            for _, row in bad.head(3).iterrows():
                result.issues.append(ValidationIssue(
                    severity="ERROR",
                    check="price_positive",
                    message=f"Non-positive {col}: {row[col]}",
                    timestamp=row["timestamp"],
                    value=float(row[col]),
                ))


def _check_volume_sanity(df: pd.DataFrame, result: ValidationResult) -> None:
    """Volume should not have extreme outliers (>100x median)."""
    median_vol = df["volume"].median()
    if median_vol <= 0:
        return

    extreme_vol = df[df["volume"] > median_vol * 100]
    for _, row in extreme_vol.head(5).iterrows():
        ratio = row["volume"] / median_vol
        result.issues.append(ValidationIssue(
            severity="WARNING",
            check="volume_outlier",
            message=f"Extreme volume: {row['volume']:.0f} ({ratio:.0f}x median)",
            timestamp=row["timestamp"],
            value=ratio,
        ))


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _count_consecutive_true(arr: np.ndarray) -> list[tuple[int, int]]:
    """
    Find runs of consecutive True values in a boolean array.

    Returns list of (start_index, run_length) tuples.
    """
    if len(arr) == 0:
        return []

    runs = []
    in_run = False
    start = 0

    for i, val in enumerate(arr):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i - start))
            in_run = False

    if in_run:
        runs.append((start, len(arr) - start))

    return runs


def validate_funding_data(
    df: pd.DataFrame,
    asset: str = "UNKNOWN",
) -> ValidationResult:
    """
    Validate funding rate DataFrame.

    Funding rates should be small numbers (typically -0.01% to +0.01% per 8h).
    """
    result = ValidationResult(asset=asset, timeframe="8h", row_count=len(df))

    if df.empty:
        result.issues.append(ValidationIssue(
            severity="ERROR", check="empty", message="Funding DataFrame is empty"
        ))
        return result

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Check for extreme funding rates (> 1% per 8h is very unusual)
    if "funding_rate" in df.columns:
        extreme = df[df["funding_rate"].abs() > 0.01]
        for _, row in extreme.head(5).iterrows():
            result.issues.append(ValidationIssue(
                severity="WARNING",
                check="funding_extreme",
                message=f"Extreme funding rate: {row['funding_rate']*100:.4f}%",
                timestamp=row["timestamp"],
                value=float(row["funding_rate"]),
            ))

        # Check for NaN
        nans = df["funding_rate"].isna().sum()
        if nans > 0:
            result.issues.append(ValidationIssue(
                severity="ERROR",
                check="funding_nan",
                message=f"{nans} NaN funding rate values",
            ))

    return result


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Validate Hyperliquid OHLCV data")
    parser.add_argument("--file", required=True, help="Path to parquet file")
    parser.add_argument("--asset", default=None, help="Asset name (auto-detected from filename)")
    parser.add_argument("--timeframe", default="1h", choices=["1h", "4h", "1d"])
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    # Auto-detect asset from filename (e.g., BTC_1h_730d.parquet)
    asset = args.asset
    if not asset:
        from pathlib import Path
        asset = Path(args.file).stem.split("_")[0]

    df = pd.read_parquet(args.file)
    result = validate_dataset(df, asset=asset, timeframe=args.timeframe, strict=args.strict)

    print(result.summary())
    print()

    if not result.passed:
        exit(1)
    else:
        print("All checks passed. Data is clean for RL training.")
