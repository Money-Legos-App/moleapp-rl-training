"""
Production Trade Exporter
==========================
Exports closed AgentPosition records from the production PostgreSQL database
for reward calibration during RL training.

Captures:
- Entry/exit prices, PnL, fees, funding
- Risk profile of the mission
- Duration, leverage, direction
- DeepSeek confidence and reasoning

This data is used to:
1. Calibrate reward functions (what does "good" look like in prod?)
2. Build a baseline for RL-vs-LLM comparison
3. Identify patterns the LLM consistently gets wrong (RL improvement targets)

Usage:
    # Set PROD_DATABASE_URL in .env first
    python scripts/export_prod_trades.py --output data/datasets/prod_trades.parquet

Requires: psycopg2-binary, pandas
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# SQL query to extract closed positions with mission context
EXPORT_QUERY = """
SELECT
    -- Position fields
    ap.id AS position_id,
    ap."missionId" AS mission_id,
    ap.asset,
    ap.direction,
    ap."entryPrice" AS entry_price,
    ap."exitPrice" AS exit_price,
    ap."sizeUsd" AS size_usd,
    ap.leverage,
    ap."unrealizedPnl" AS unrealized_pnl,
    ap."realizedPnl" AS realized_pnl,
    ap."feesUsd" AS fees_usd,
    ap."fundingFeesUsd" AS funding_fees_usd,
    ap.status AS position_status,
    ap."openedAt" AS opened_at,
    ap."closedAt" AS closed_at,
    ap."closedReason" AS close_reason,

    -- Mission context
    am."riskLevel" AS risk_level,
    am."capitalUsd" AS mission_capital,
    am.status AS mission_status,
    am."startedAt" AS mission_started_at,

    -- Signal context (from the signal that triggered this position)
    asig.confidence AS signal_confidence,
    asig."strategyTag" AS strategy_tag,
    asig."stopLossPercent" AS signal_sl_pct,
    asig."takeProfitPercent" AS signal_tp_pct,
    asig."recommendedLeverage" AS signal_leverage,
    asig."llmProvider" AS llm_provider,
    asig."responseTimeMs" AS llm_response_ms

FROM "AgentPosition" ap
JOIN "AgentMission" am ON ap."missionId" = am.id
LEFT JOIN "AgentSignal" asig ON asig.id = ap."signalId"
WHERE ap.status IN ('CLOSED', 'LIQUIDATED', 'FORCE_CLOSED')
ORDER BY ap."closedAt" DESC;
"""


def export_trades(
    database_url: str,
    output_path: str = "data/datasets/prod_trades.parquet",
) -> pd.DataFrame:
    """
    Export closed trades from production database.

    Args:
        database_url: PostgreSQL connection string
        output_path: Where to save the parquet file

    Returns:
        DataFrame of exported trades
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2-binary is required: pip install psycopg2-binary")
        raise

    logger.info("Connecting to production database...")

    try:
        conn = psycopg2.connect(database_url)
        df = pd.read_sql(EXPORT_QUERY, conn)
        conn.close()
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise

    if df.empty:
        logger.warning("No closed positions found in database")
        return df

    # Compute derived fields for training calibration
    df["duration_hours"] = (
        pd.to_datetime(df["closed_at"]) - pd.to_datetime(df["opened_at"])
    ).dt.total_seconds() / 3600.0

    df["pnl_pct"] = df["realized_pnl"] / df["size_usd"].clip(lower=1.0) * 100.0
    df["net_pnl"] = df["realized_pnl"] - df["fees_usd"].fillna(0) - df["funding_fees_usd"].fillna(0)
    df["net_pnl_pct"] = df["net_pnl"] / df["size_usd"].clip(lower=1.0) * 100.0
    df["is_winner"] = df["realized_pnl"] > 0

    # Risk-profile mapping
    df["risk_level"] = df["risk_level"].fillna("MODERATE")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    # Summary stats
    total = len(df)
    winners = df["is_winner"].sum()
    by_profile = df.groupby("risk_level").agg(
        count=("position_id", "count"),
        avg_pnl_pct=("net_pnl_pct", "mean"),
        win_rate=("is_winner", "mean"),
        avg_duration_h=("duration_hours", "mean"),
    )

    logger.info(f"Exported {total} trades ({winners} winners, {total - winners} losers)")
    logger.info(f"Output: {output}")
    logger.info(f"\nPer-profile stats:\n{by_profile.to_string()}")

    return df


def compute_reward_calibration(df: pd.DataFrame) -> dict:
    """
    Compute reward calibration targets from real trade data.

    These values inform reward shaping in Shield/Builder envs:
    - What PnL % distribution does each profile achieve?
    - What drawdown levels are acceptable?
    - How often does each profile trade?
    """
    calibration = {}

    for profile in ["CONSERVATIVE", "MODERATE"]:
        profile_df = df[df["risk_level"] == profile]
        if profile_df.empty:
            continue

        calibration[profile] = {
            "trade_count": len(profile_df),
            "win_rate": float(profile_df["is_winner"].mean()),
            "avg_pnl_pct": float(profile_df["net_pnl_pct"].mean()),
            "median_pnl_pct": float(profile_df["net_pnl_pct"].median()),
            "p25_pnl_pct": float(profile_df["net_pnl_pct"].quantile(0.25)),
            "p75_pnl_pct": float(profile_df["net_pnl_pct"].quantile(0.75)),
            "worst_trade_pct": float(profile_df["net_pnl_pct"].min()),
            "best_trade_pct": float(profile_df["net_pnl_pct"].max()),
            "avg_duration_hours": float(profile_df["duration_hours"].mean()),
            "avg_leverage": float(profile_df["leverage"].mean()),
            "avg_fee_pct": float(
                (profile_df["fees_usd"].fillna(0) / profile_df["size_usd"].clip(lower=1)).mean() * 100
            ),
            "avg_funding_pct": float(
                (profile_df["funding_fees_usd"].fillna(0) / profile_df["size_usd"].clip(lower=1)).mean() * 100
            ),
        }

    return calibration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export production trades for RL training")
    parser.add_argument(
        "--output",
        default="data/datasets/prod_trades.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("PROD_DATABASE_URL"),
        help="PostgreSQL connection string (or set PROD_DATABASE_URL env var)",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Also output reward calibration analysis",
    )
    args = parser.parse_args()

    if not args.database_url:
        logger.error("PROD_DATABASE_URL not set. Pass --database-url or set env var.")
        exit(1)

    df = export_trades(
        database_url=args.database_url,
        output_path=args.output,
    )

    if args.calibration and not df.empty:
        import json
        cal = compute_reward_calibration(df)
        cal_path = Path(args.output).with_suffix(".calibration.json")
        with open(cal_path, "w") as f:
            json.dump(cal, f, indent=2)
        logger.info(f"Reward calibration saved to {cal_path}")
