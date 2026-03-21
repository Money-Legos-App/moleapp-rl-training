"""
Tests for the Binance Vision data collector.
All tests are offline (no network calls) using synthetic CSV data.
"""

from __future__ import annotations

import io
import struct
import zipfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.collectors.asset_config import (
    ALLOWED_ASSETS,
    BINANCE_LISTING_DATES,
    BINANCE_SYMBOL_MAP,
    REVERSE_SYMBOL_MAP,
)
from data.collectors.binance_vision_collector import (
    _build_funding_url,
    _build_kline_url,
    _clamp_extreme_wicks,
    _fill_maintenance_gaps,
    _generate_month_range,
    _parse_funding_csv,
    _parse_kline_csv,
)
from data.validators.validate_hl_data import validate_dataset


# ──────────────────────────────────────────────────────────────────────
# Helpers: Generate synthetic Binance CSV data
# ──────────────────────────────────────────────────────────────────────

def _make_kline_csv_bytes(
    n_candles: int = 100,
    start_ms: int = 1704067200000,  # 2024-01-01 00:00 UTC
    interval_ms: int = 3600000,      # 1h
    base_price: float = 50000.0,
    include_header: bool = False,
) -> bytes:
    """Generate synthetic Binance kline CSV bytes (12 columns, no header)."""
    rng = np.random.RandomState(42)
    lines = []

    if include_header:
        lines.append(
            "open_time,open,high,low,close,volume,"
            "close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore"
        )

    price = base_price
    for i in range(n_candles):
        open_time = start_ms + i * interval_ms
        close_time = open_time + interval_ms - 1

        noise = abs(rng.normal(0, price * 0.001))
        o = price
        c = price + rng.normal(0, price * 0.0005)
        h = max(o, c) + noise
        l = min(o, c) - noise
        v = rng.uniform(100, 50000)

        price = c  # Next candle's base

        lines.append(
            f"{open_time},{o:.8f},{h:.8f},{l:.8f},{c:.8f},{v:.8f},"
            f"{close_time},{v * c:.2f},{int(rng.uniform(100, 5000))},"
            f"{v * 0.6:.8f},{v * 0.6 * c:.2f},0"
        )

    return "\n".join(lines).encode("utf-8")


def _make_funding_csv_bytes(
    n_records: int = 30,
    start_ms: int = 1704067200000,  # 2024-01-01 00:00 UTC
    interval_ms: int = 28800000,     # 8h
) -> bytes:
    """Generate synthetic Binance funding rate CSV bytes."""
    rng = np.random.RandomState(42)
    lines = ["calc_time,funding_interval_hours,last_funding_rate,mark_price"]

    for i in range(n_records):
        ts = start_ms + i * interval_ms
        rate = rng.normal(0.0001, 0.0002)
        mark = 50000 + rng.normal(0, 100)
        lines.append(f"{ts},8,{rate:.10f},{mark:.2f}")

    return "\n".join(lines).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────
# Test: Symbol Mapping
# ──────────────────────────────────────────────────────────────────────

class TestSymbolMapping:
    def test_all_assets_have_binance_symbols(self):
        """Every MoleApp asset maps to a Binance symbol."""
        for asset in ALLOWED_ASSETS:
            assert asset in BINANCE_SYMBOL_MAP, f"Missing Binance symbol for {asset}"

    def test_kpepe_maps_to_1000pepe(self):
        """kPEPE → 1000PEPEUSDT (Binance uses 1000x multiplier)."""
        assert BINANCE_SYMBOL_MAP["kPEPE"] == "1000PEPEUSDT"

    def test_reverse_map_works(self):
        """Reverse map resolves Binance symbols back to MoleApp names."""
        assert REVERSE_SYMBOL_MAP["BTCUSDT"] == "BTC"
        assert REVERSE_SYMBOL_MAP["1000PEPEUSDT"] == "kPEPE"

    def test_all_symbols_have_usdt_suffix(self):
        """All Binance symbols end with USDT."""
        for asset, symbol in BINANCE_SYMBOL_MAP.items():
            assert symbol.endswith("USDT"), f"{asset} → {symbol} missing USDT suffix"

    def test_all_symbols_have_listing_dates(self):
        """Every Binance symbol has a listing date."""
        for asset, symbol in BINANCE_SYMBOL_MAP.items():
            assert symbol in BINANCE_LISTING_DATES, f"Missing listing date for {symbol}"

    def test_map_count_matches_assets(self):
        """15 assets = 15 symbols."""
        assert len(BINANCE_SYMBOL_MAP) == len(ALLOWED_ASSETS)


# ──────────────────────────────────────────────────────────────────────
# Test: URL Generation
# ──────────────────────────────────────────────────────────────────────

class TestURLGeneration:
    def test_kline_url_format(self):
        url = _build_kline_url("BTCUSDT", "1h", 2024, 3)
        expected = (
            "https://data.binance.vision/data/futures/um/monthly/"
            "klines/BTCUSDT/1h/BTCUSDT-1h-2024-03.zip"
        )
        assert url == expected

    def test_funding_url_format(self):
        url = _build_funding_url("ETHUSDT", 2024, 12)
        expected = (
            "https://data.binance.vision/data/futures/um/monthly/"
            "fundingRate/ETHUSDT/ETHUSDT-fundingRate-2024-12.zip"
        )
        assert url == expected

    def test_month_zero_padded(self):
        url = _build_kline_url("SOLUSDT", "1h", 2024, 1)
        assert "2024-01.zip" in url


# ──────────────────────────────────────────────────────────────────────
# Test: Month Range Generation
# ──────────────────────────────────────────────────────────────────────

class TestMonthRange:
    def test_730_days_produces_about_24_months(self):
        months = _generate_month_range(730)
        assert 23 <= len(months) <= 25

    def test_months_are_descending(self):
        months = _generate_month_range(365)
        for i in range(len(months) - 1):
            assert months[i] >= months[i + 1]

    def test_short_lookback(self):
        months = _generate_month_range(45)
        assert 1 <= len(months) <= 3

    def test_year_boundary_handled(self):
        """Verify months span across year boundaries correctly."""
        months = _generate_month_range(730)
        years = set(y for y, m in months)
        assert len(years) >= 2  # Must span at least 2 years


# ──────────────────────────────────────────────────────────────────────
# Test: Kline CSV Parsing
# ──────────────────────────────────────────────────────────────────────

class TestKlineCSVParsing:
    def test_standard_format_no_header(self):
        """Standard Binance kline CSV: 12 columns, no header."""
        csv_bytes = _make_kline_csv_bytes(50)
        df = _parse_kline_csv(csv_bytes)

        assert len(df) == 50
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["open"].dtype == "float64"
        assert df["volume"].dtype == "float64"

    def test_format_with_header(self):
        """Some Binance CSVs include a header row."""
        csv_bytes = _make_kline_csv_bytes(50, include_header=True)
        df = _parse_kline_csv(csv_bytes)

        assert len(df) == 50
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_timestamps_are_sequential(self):
        csv_bytes = _make_kline_csv_bytes(100)
        df = _parse_kline_csv(csv_bytes)

        diffs = df["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_all_prices_positive(self):
        csv_bytes = _make_kline_csv_bytes(100, base_price=50000.0)
        df = _parse_kline_csv(csv_bytes)

        assert (df["open"] > 0).all()
        assert (df["high"] > 0).all()
        assert (df["low"] > 0).all()
        assert (df["close"] > 0).all()
        assert (df["volume"] >= 0).all()


# ──────────────────────────────────────────────────────────────────────
# Test: Funding CSV Parsing
# ──────────────────────────────────────────────────────────────────────

class TestFundingCSVParsing:
    def test_standard_format(self):
        csv_bytes = _make_funding_csv_bytes(30)
        df = _parse_funding_csv(csv_bytes)

        assert len(df) == 30
        assert list(df.columns) == ["timestamp", "funding_rate", "premium"]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["funding_rate"].dtype == "float64"

    def test_funding_rates_are_small(self):
        """Funding rates should be small (typically < 0.01)."""
        csv_bytes = _make_funding_csv_bytes(30)
        df = _parse_funding_csv(csv_bytes)

        assert (df["funding_rate"].abs() < 0.01).all()

    def test_premium_is_zero(self):
        """Binance Vision doesn't include premium; should default to 0."""
        csv_bytes = _make_funding_csv_bytes(10)
        df = _parse_funding_csv(csv_bytes)

        assert (df["premium"] == 0.0).all()


# ──────────────────────────────────────────────────────────────────────
# Test: Gap Filling
# ──────────────────────────────────────────────────────────────────────

class TestGapFilling:
    def _make_gapped_df(self, gap_hours: int, gap_at: int = 50) -> pd.DataFrame:
        """Create OHLCV df with a gap_hours gap at position gap_at.

        Before the gap: gap_at candles at 1h intervals.
        After the gap: (100 - gap_at) candles at 1h intervals starting
        gap_hours after the last pre-gap candle.
        """
        rng = np.random.RandomState(42)
        n_before = gap_at
        n_after = 100 - gap_at

        ts_before = pd.date_range("2024-01-01", periods=n_before, freq="1h")
        gap_start = ts_before[-1] + pd.Timedelta(hours=gap_hours)
        ts_after = pd.date_range(gap_start, periods=n_after, freq="1h")

        ts_list = list(ts_before) + list(ts_after)
        n = len(ts_list)

        prices = 50000.0 * np.cumprod(1.0 + rng.normal(0, 0.002, n))

        return pd.DataFrame({
            "timestamp": ts_list,
            "open": prices - 10,
            "high": prices + 10,
            "low": prices - 10,
            "close": prices,
            "volume": rng.uniform(1000, 50000, n),
        })

    def test_2h_gap_filled(self):
        """A 2-hour gap should be filled with volume=0."""
        df = self._make_gapped_df(gap_hours=2)
        original_len = len(df)
        filled = _fill_maintenance_gaps(df, max_gap_hours=3)

        # Should have 1 extra row (filling the 1 missing candle)
        assert len(filled) == original_len + 1
        # The filled candle should have volume=0
        maintenance_rows = filled[filled["maintenance"] == True]
        assert len(maintenance_rows) == 1
        assert maintenance_rows.iloc[0]["volume"] == 0.0

    def test_3h_gap_filled(self):
        """A 3-hour gap should be filled (exactly at threshold)."""
        df = self._make_gapped_df(gap_hours=3)
        original_len = len(df)
        filled = _fill_maintenance_gaps(df, max_gap_hours=3)

        # Should have 2 extra rows (filling 2 missing candles)
        assert len(filled) == original_len + 2
        maintenance_rows = filled[filled["maintenance"] == True]
        assert len(maintenance_rows) == 2

    def test_12h_gap_not_filled(self):
        """A 12-hour gap should NOT be filled (too large)."""
        df = self._make_gapped_df(gap_hours=12)
        original_len = len(df)
        filled = _fill_maintenance_gaps(df, max_gap_hours=3)

        # Same number of rows (gap not filled)
        assert len(filled) == original_len
        maintenance_rows = filled[filled["maintenance"] == True]
        assert len(maintenance_rows) == 0

    def test_no_gap_unchanged(self):
        """Clean data with no gaps should pass through unchanged."""
        n = 50
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": np.ones(n) * 50000,
            "high": np.ones(n) * 50100,
            "low": np.ones(n) * 49900,
            "close": np.ones(n) * 50000,
            "volume": np.ones(n) * 1000,
        })
        filled = _fill_maintenance_gaps(df)

        assert len(filled) == n
        assert (filled["maintenance"] == False).all()


# ──────────────────────────────────────────────────────────────────────
# Test: Monthly Stitching & Dedup
# ──────────────────────────────────────────────────────────────────────

class TestStitching:
    def test_overlapping_months_deduped(self):
        """When two monthly chunks overlap at boundaries, dedup produces
        a continuous timeline without duplicates."""
        # Month 1: 720 candles (30 days)
        csv1 = _make_kline_csv_bytes(720, start_ms=1704067200000)
        # Month 2: 720 candles starting 10 candles before month 1 ends (overlap)
        overlap_start = 1704067200000 + (710 * 3600000)
        csv2 = _make_kline_csv_bytes(720, start_ms=overlap_start)

        df1 = _parse_kline_csv(csv1)
        df2 = _parse_kline_csv(csv2)

        combined = pd.concat([df1, df2], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        combined = combined.reset_index(drop=True)

        # Should have no duplicate timestamps
        assert combined["timestamp"].is_unique
        # Should be continuous (all 1h gaps)
        diffs = combined["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()


# ──────────────────────────────────────────────────────────────────────
# Test: Validator Integration
# ──────────────────────────────────────────────────────────────────────

class TestValidatorIntegration:
    def test_clean_binance_data_passes_validator(self):
        """Synthetic clean Binance-format data should pass validate_dataset()."""
        csv_bytes = _make_kline_csv_bytes(200, base_price=50000.0)
        df = _parse_kline_csv(csv_bytes)

        result = validate_dataset(df, asset="BTC", timeframe="1h")

        # Should pass (no OHLC violations, no spikes, continuous)
        ohlc_errors = [i for i in result.errors if "ohlc" in i.check]
        spike_errors = [i for i in result.errors if "spike" in i.check]
        gap_errors = [i for i in result.errors if "gap" in i.check]

        assert len(ohlc_errors) == 0, f"Unexpected OHLC errors: {ohlc_errors}"
        assert len(spike_errors) == 0, f"Unexpected spike errors: {spike_errors}"
        assert len(gap_errors) == 0, f"Unexpected gap errors: {gap_errors}"

    def test_schema_matches_hl_collector(self):
        """Output columns must match HL collector schema exactly."""
        csv_bytes = _make_kline_csv_bytes(50)
        df = _parse_kline_csv(csv_bytes)

        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_cols


# ──────────────────────────────────────────────────────────────────────
# Test: Partial History (404 Handling)
# ──────────────────────────────────────────────────────────────────────

class TestPartialHistory:
    def test_popcat_listing_date_is_recent(self):
        """POPCAT listed Aug 2024 — should have <2 years of data."""
        listing = BINANCE_LISTING_DATES["POPCATUSDT"]
        year, month = map(int, listing.split("-"))
        listing_dt = datetime(year, month, 1, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        months_available = (now.year - listing_dt.year) * 12 + (now.month - listing_dt.month)
        assert months_available < 24, "POPCAT should have less than 2 years"

    def test_btc_has_full_history(self):
        """BTC listed Sep 2019 — should have full 2+ years."""
        listing = BINANCE_LISTING_DATES["BTCUSDT"]
        year, month = map(int, listing.split("-"))
        listing_dt = datetime(year, month, 1, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        months_available = (now.year - listing_dt.year) * 12 + (now.month - listing_dt.month)
        assert months_available >= 24, "BTC should have 2+ years"


# ──────────────────────────────────────────────────────────────────────
# Test: Wick Clamping
# ──────────────────────────────────────────────────────────────────────

class TestWickClamping:
    def test_extreme_wick_clamped(self):
        """A candle with 90% wick should be clamped to ≤40%."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [102.0, 102.0, 190.0, 102.0, 102.0],  # idx 2: extreme
            "low": [98.0, 98.0, 10.0, 98.0, 98.0],         # idx 2: extreme
            "close": [101.0, 101.0, 100.0, 101.0, 101.0],
            "volume": [1000.0] * 5,
        })
        result = _clamp_extreme_wicks(df, max_wick_pct=0.40)

        # Extreme candle should now have wick ≤40% of close
        row = result.iloc[2]
        wick_pct = (row["high"] - row["low"]) / row["close"]
        assert wick_pct <= 0.45, f"Wick still too large: {wick_pct:.2%}"

        # Non-extreme candles should be unchanged
        assert result.iloc[0]["high"] == 102.0
        assert result.iloc[0]["low"] == 98.0

    def test_normal_wick_unchanged(self):
        """Candles with normal wicks should not be modified."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "open": [100.0, 100.0, 100.0],
            "high": [105.0, 103.0, 102.0],
            "low": [96.0, 97.0, 98.0],
            "close": [101.0, 101.0, 101.0],
            "volume": [1000.0] * 3,
        })
        result = _clamp_extreme_wicks(df, max_wick_pct=0.40)

        pd.testing.assert_frame_equal(result, df)

    def test_open_close_preserved(self):
        """Clamping must not change open or close values."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="1h"),
            "open": [95.0],
            "high": [200.0],
            "low": [5.0],
            "close": [105.0],
            "volume": [5000.0],
        })
        result = _clamp_extreme_wicks(df, max_wick_pct=0.40)

        assert result.iloc[0]["open"] == 95.0
        assert result.iloc[0]["close"] == 105.0
        # High must be >= max(open, close) = 105
        assert result.iloc[0]["high"] >= 105.0
        # Low must be <= min(open, close) = 95
        assert result.iloc[0]["low"] <= 95.0

    def test_high_ge_low_after_clamp(self):
        """After clamping, high must still be >= low."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "open": [100.0, 50.0, 200.0],
            "high": [300.0, 150.0, 400.0],
            "low": [10.0, 5.0, 20.0],
            "close": [110.0, 55.0, 210.0],
            "volume": [1000.0] * 3,
        })
        result = _clamp_extreme_wicks(df, max_wick_pct=0.40)

        assert (result["high"] >= result["low"]).all()
