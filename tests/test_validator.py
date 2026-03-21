"""
Tests for the data validator — ensures it catches all RL-exploitable glitches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.validators.validate_hl_data import (
    ValidationResult,
    validate_dataset,
    validate_funding_data,
)


def _make_clean_df(n_hours: int = 100, base_price: float = 50000.0) -> pd.DataFrame:
    """Generate a clean, valid OHLCV DataFrame."""
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="1h")
    prices = base_price * np.cumprod(1.0 + rng.normal(0, 0.002, n_hours))

    rows = []
    for i, (ts, p) in enumerate(zip(timestamps, prices)):
        noise = abs(rng.normal(0, p * 0.001))
        rows.append({
            "timestamp": ts,
            "open": p - noise * 0.5,
            "high": p + noise,
            "low": p - noise,
            "close": p,
            "volume": float(rng.uniform(1e6, 5e7)),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Test: Clean data passes
# ──────────────────────────────────────────────────────────────────────

class TestCleanDataPasses:
    def test_valid_data_passes(self):
        df = _make_clean_df(100)
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        assert result.passed, f"Clean data should pass:\n{result.summary()}"

    def test_result_has_correct_metadata(self):
        df = _make_clean_df(50)
        result = validate_dataset(df, asset="ETH", timeframe="1h")
        assert result.asset == "ETH"
        assert result.timeframe == "1h"
        assert result.row_count == 50


# ──────────────────────────────────────────────────────────────────────
# Test 1: Continuity — gaps and duplicates
# ──────────────────────────────────────────────────────────────────────

class TestContinuity:
    def test_missing_candle_detected(self):
        """Remove a candle from the middle — should flag a gap."""
        df = _make_clean_df(100)
        # Drop row 50 to create a 2-hour gap
        df = df.drop(index=50).reset_index(drop=True)
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        gap_issues = [i for i in result.issues if "gap" in i.check]
        assert len(gap_issues) > 0, "Should detect missing candle"

    def test_large_gap_is_error(self):
        """A 12-hour gap should be an ERROR."""
        df = _make_clean_df(100)
        # Drop 12 consecutive rows
        df = df.drop(index=range(40, 52)).reset_index(drop=True)
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        errors = [i for i in result.errors if "gap" in i.check]
        assert len(errors) > 0, "12-hour gap should be an ERROR"

    def test_duplicate_timestamp_detected(self):
        """Duplicate timestamps should be flagged."""
        df = _make_clean_df(100)
        # Duplicate row 50
        dup = df.iloc[50:51].copy()
        df = pd.concat([df, dup], ignore_index=True).sort_values("timestamp")
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        dup_issues = [i for i in result.issues if "duplicate" in i.check]
        assert len(dup_issues) > 0, "Should detect duplicate timestamp"


# ──────────────────────────────────────────────────────────────────────
# Test 2: OHLC logic violations
# ──────────────────────────────────────────────────────────────────────

class TestOHLCLogic:
    def test_high_below_close(self):
        """High lower than Close = broken candle."""
        df = _make_clean_df(100)
        df.loc[50, "high"] = df.loc[50, "close"] - 100  # high below close
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        ohlc_errors = [i for i in result.errors if "ohlc" in i.check]
        assert len(ohlc_errors) > 0, "Should detect high < close"

    def test_low_above_open(self):
        """Low higher than Open = broken candle."""
        df = _make_clean_df(100)
        df.loc[30, "low"] = df.loc[30, "open"] + 100  # low above open
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        ohlc_errors = [i for i in result.errors if "ohlc" in i.check]
        assert len(ohlc_errors) > 0, "Should detect low > open"

    def test_negative_volume(self):
        """Negative volume = impossible."""
        df = _make_clean_df(100)
        df.loc[20, "volume"] = -1000
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        vol_errors = [i for i in result.errors if "neg_volume" in i.check]
        assert len(vol_errors) > 0, "Should detect negative volume"


# ──────────────────────────────────────────────────────────────────────
# Test 3: Outlier detection — spikes and extreme wicks
# ──────────────────────────────────────────────────────────────────────

class TestOutliers:
    def test_flash_crash_detected(self):
        """BTC dropping 50% in one candle = glitch, not real."""
        df = _make_clean_df(100)
        df.loc[60, "close"] = df.loc[59, "close"] * 0.5  # 50% drop
        df.loc[60, "low"] = df.loc[60, "close"] * 0.99
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        spike_errors = [i for i in result.errors if "spike" in i.check]
        assert len(spike_errors) > 0, "Should detect 50% flash crash"

    def test_flash_pump_detected(self):
        """BTC doubling in one candle = API glitch."""
        df = _make_clean_df(100)
        df.loc[60, "close"] = df.loc[59, "close"] * 2.0  # 100% pump
        df.loc[60, "high"] = df.loc[60, "close"] * 1.01
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        spike_errors = [i for i in result.errors if "spike" in i.check]
        assert len(spike_errors) > 0, "Should detect 100% flash pump"

    def test_extreme_wick_detected(self):
        """A wick that spans 60% of price = suspicious."""
        df = _make_clean_df(100)
        price = df.loc[50, "close"]
        df.loc[50, "high"] = price * 1.4  # 40% above close
        df.loc[50, "low"] = price * 0.7   # 30% below close → 70% range
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        wick_errors = [i for i in result.errors if "wick" in i.check]
        assert len(wick_errors) > 0, "Should detect extreme wick"

    def test_minor_asset_higher_threshold(self):
        """Minor assets (DOGE) tolerate higher moves than BTC."""
        df = _make_clean_df(100, base_price=0.30)
        # 25% move — would fail BTC (20% threshold) but pass DOGE (35%)
        df.loc[60, "close"] = df.loc[59, "close"] * 1.25
        df.loc[60, "high"] = df.loc[60, "close"] * 1.01

        btc_result = validate_dataset(df, asset="BTC", timeframe="1h")
        doge_result = validate_dataset(df, asset="DOGE", timeframe="1h")

        btc_spikes = [i for i in btc_result.errors if "spike" in i.check]
        doge_spikes = [i for i in doge_result.errors if "spike" in i.check]

        assert len(btc_spikes) > 0, "25% move should fail for BTC"
        assert len(doge_spikes) == 0, "25% move should pass for DOGE"


# ──────────────────────────────────────────────────────────────────────
# Test 3b: Flatline detection
# ──────────────────────────────────────────────────────────────────────

class TestFlatlines:
    def test_price_flatline_detected(self):
        """12 hours of identical close = suspicious (exchange down?)."""
        df = _make_clean_df(100)
        frozen_price = 50000.0
        for i in range(40, 52):  # 12 identical closes
            df.loc[i, "close"] = frozen_price
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        flat_issues = [i for i in result.issues if "flatline_price" in i.check]
        assert len(flat_issues) > 0, "Should detect 12-hour price flatline"

    def test_zero_volume_streak_detected(self):
        """Multiple zero-volume candles = exchange offline."""
        df = _make_clean_df(100)
        for i in range(30, 37):  # 7 zero-volume candles
            df.loc[i, "volume"] = 0.0
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        vol_issues = [i for i in result.issues if "flatline_volume" in i.check]
        assert len(vol_issues) > 0, "Should detect zero-volume streak"


# ──────────────────────────────────────────────────────────────────────
# Test: Price positivity
# ──────────────────────────────────────────────────────────────────────

class TestPricePositivity:
    def test_zero_price_detected(self):
        df = _make_clean_df(100)
        df.loc[50, "close"] = 0.0
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        pos_errors = [i for i in result.errors if "price_positive" in i.check]
        assert len(pos_errors) > 0, "Should detect zero price"

    def test_negative_price_detected(self):
        df = _make_clean_df(100)
        df.loc[50, "low"] = -1.0
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        pos_errors = [i for i in result.errors if "price_positive" in i.check]
        assert len(pos_errors) > 0, "Should detect negative price"


# ──────────────────────────────────────────────────────────────────────
# Test: Empty DataFrame
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_df_fails(self):
        df = pd.DataFrame()
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        assert not result.passed

    def test_missing_columns_fails(self):
        df = pd.DataFrame({"timestamp": [1, 2], "close": [100, 101]})
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        assert not result.passed
        col_errors = [i for i in result.errors if "columns" in i.check]
        assert len(col_errors) > 0


# ──────────────────────────────────────────────────────────────────────
# Test: Funding rate validator
# ──────────────────────────────────────────────────────────────────────

class TestFundingValidator:
    def test_extreme_funding_flagged(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="8h"),
            "funding_rate": [0.0001] * 9 + [0.05],  # Last one is extreme
        })
        result = validate_funding_data(df, asset="BTC")
        warnings = [i for i in result.warnings if "funding_extreme" in i.check]
        assert len(warnings) > 0

    def test_nan_funding_detected(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="8h"),
            "funding_rate": [0.0001, 0.0002, float("nan"), 0.0001, 0.0003],
        })
        result = validate_funding_data(df, asset="BTC")
        nan_errors = [i for i in result.errors if "nan" in i.check]
        assert len(nan_errors) > 0


# ──────────────────────────────────────────────────────────────────────
# Test: ValidationResult API
# ──────────────────────────────────────────────────────────────────────

class TestValidationResult:
    def test_summary_output(self):
        result = ValidationResult(asset="BTC", timeframe="1h", row_count=100)
        s = result.summary()
        assert "PASSED" in s
        assert "BTC" in s

    def test_failed_summary(self):
        df = _make_clean_df(100)
        df.loc[50, "high"] = 0.0  # Will trigger multiple errors
        result = validate_dataset(df, asset="BTC", timeframe="1h")
        s = result.summary()
        assert "FAILED" in s
        assert "[ERROR]" in s
