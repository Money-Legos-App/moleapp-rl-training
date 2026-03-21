"""
Shared Asset Configuration
===========================
Single source of truth for MoleApp's 15 allowed assets and their
Binance USDⓈ-M perpetual symbol mappings.
"""

from __future__ import annotations

# MoleApp's 15 allowed assets
ALLOWED_ASSETS = [
    "BTC", "ETH", "SOL", "SUI", "SEI", "AVAX", "TAO", "FET",
    "NEAR", "WIF", "POPCAT", "kPEPE", "DOGE", "PENDLE", "ARB",
]

# MoleApp asset name → Binance USDⓈ-M perpetual symbol
BINANCE_SYMBOL_MAP: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "SUI": "SUIUSDT",
    "SEI": "SEIUSDT",
    "AVAX": "AVAXUSDT",
    "TAO": "TAOUSDT",
    "FET": "FETUSDT",
    "NEAR": "NEARUSDT",
    "WIF": "WIFUSDT",
    "POPCAT": "POPCATUSDT",
    "kPEPE": "1000PEPEUSDT",
    "DOGE": "DOGEUSDT",
    "PENDLE": "PENDLEUSDT",
    "ARB": "ARBUSDT",
}

# Reverse map: Binance symbol → MoleApp asset name
REVERSE_SYMBOL_MAP: dict[str, str] = {v: k for k, v in BINANCE_SYMBOL_MAP.items()}

# Approximate Binance Futures listing dates (YYYY-MM).
# Assets listed before 2024-04 have ≥2 years of history as of 2026-03.
BINANCE_LISTING_DATES: dict[str, str] = {
    "BTCUSDT": "2019-09",
    "ETHUSDT": "2019-09",
    "SOLUSDT": "2021-07",
    "SUIUSDT": "2023-05",
    "SEIUSDT": "2023-08",
    "AVAXUSDT": "2021-09",
    "TAOUSDT": "2024-04",
    "FETUSDT": "2023-04",
    "NEARUSDT": "2021-11",
    "WIFUSDT": "2024-03",
    "POPCATUSDT": "2024-08",
    "1000PEPEUSDT": "2023-04",
    "DOGEUSDT": "2021-01",
    "PENDLEUSDT": "2023-07",
    "ARBUSDT": "2023-03",
}
