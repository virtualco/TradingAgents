#!/usr/bin/env python3
"""Backfill historical observations into the paper trading database.

Replays the last N trading days using real yfinance OHLCV data so the
observation period counter reaches the 90-day minimum required for
live-readiness assessment.

Usage:
    python3 scripts/backfill_observations.py --days 89
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from tradingagents.execution.observer import DailyObserver, ObservationConfig
from tradingagents.research.strategy_rules import TechnicalStrategyRules
from tradingagents.research.signal_registry import SignalRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA",
    "AMZN", "META", "JPM", "V", "UNH",
]


def get_trading_days(n: int) -> list[str]:
    """Return the last n weekdays (Mon-Fri) before today, oldest first."""
    days = []
    current = date.today() - timedelta(days=1)
    while len(days) < n:
        if current.weekday() < 5:  # Mon=0 … Fri=4
            days.append(current.isoformat())
        current -= timedelta(days=1)
    return list(reversed(days))


def fetch_bulk_ohlcv(tickers: list[str], lookback_days: int = 200) -> dict[str, pd.DataFrame]:
    """Download OHLCV history for all tickers in one yfinance call."""
    import yfinance as yf

    logger.info(f"Downloading {lookback_days}d OHLCV for {len(tickers)} tickers…")
    raw = yf.download(
        tickers,
        period=f"{lookback_days}d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[ticker].copy()
            else:
                df = raw.copy()
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index = pd.to_datetime(df.index)
            result[ticker] = df.dropna(subset=["close"])
        except Exception as e:
            logger.warning(f"Could not extract OHLCV for {ticker}: {e}")
    logger.info(f"OHLCV ready for {len(result)}/{len(tickers)} tickers")
    return result


def generate_signals_for_date(
    tickers: list[str],
    ohlcv_map: dict[str, pd.DataFrame],
    trade_date: str,
    threshold: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate signals and prices for a specific historical date."""
    import hashlib

    tech_rules = TechnicalStrategyRules()
    rows = []
    prices: dict[str, float] = {}

    cutoff = pd.Timestamp(trade_date)

    for ticker in tickers:
        if ticker not in ohlcv_map:
            continue
        hist = ohlcv_map[ticker]
        # Time-safe slice: only rows on or before trade_date
        hist_safe = hist[hist.index <= cutoff]
        if len(hist_safe) < 20:
            continue

        price = float(hist_safe["close"].iloc[-1])
        prices[ticker] = price

        tech_summary = tech_rules.compute(hist_safe, ticker=ticker, trade_date=trade_date)
        score = tech_summary.composite_score

        if score >= threshold:
            direction = "long"
            conviction = min(0.95, 0.5 + score * 0.5)
        elif score <= -threshold:
            direction = "short"
            conviction = min(0.95, 0.5 + abs(score) * 0.5)
        else:
            direction = "flat"
            conviction = 0.0

        if direction != "flat":
            signal_id = hashlib.sha256(
                f"{ticker}-{trade_date}-{direction}".encode()
            ).hexdigest()[:16]
            rows.append({
                "ticker": ticker,
                "direction": direction,
                "conviction": round(conviction, 4),
                "signal_id": signal_id,
                "ensemble_score": round(score, 4),
                "tech_score": round(score, 4),
            })

    return pd.DataFrame(rows), prices


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill historical observations")
    parser.add_argument("--days", type=int, default=89,
                        help="Number of historical trading days to backfill (default: 89)")
    parser.add_argument("--db", type=str,
                        default=os.environ.get(
                            "TRADINGAGENTS_DB",
                            str(REPO_ROOT / "data" / "paper_trading.db")
                        ))
    parser.add_argument("--capital", type=float,
                        default=float(os.environ.get("TRADINGAGENTS_CAPITAL", "100000")))
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="Signal score threshold (default: 0.20)")
    args = parser.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    trading_days = get_trading_days(args.days)
    logger.info(f"Backfilling {len(trading_days)} days: {trading_days[0]} → {trading_days[-1]}")

    # Fetch all OHLCV data once
    ohlcv_map = fetch_bulk_ohlcv(DEFAULT_TICKERS, lookback_days=200)

    config = ObservationConfig(
        db_path=args.db,
        initial_capital=args.capital,
    )
    observer = DailyObserver(config=config)

    # Check which dates already exist in DB
    import sqlite3
    existing_dates: set[str] = set()
    try:
        with sqlite3.connect(args.db) as conn:
            rows = conn.execute(
                "SELECT trade_date FROM daily_observations"
            ).fetchall()
            existing_dates = {r[0] for r in rows}
    except Exception:
        pass

    filled = 0
    skipped = 0
    for trade_date in trading_days:
        if trade_date in existing_dates:
            logger.info(f"  {trade_date}: already exists — skipping")
            skipped += 1
            continue

        signals, prices = generate_signals_for_date(
            DEFAULT_TICKERS, ohlcv_map, trade_date, threshold=args.threshold
        )

        if not prices:
            logger.warning(f"  {trade_date}: no prices available — skipping")
            skipped += 1
            continue

        obs = observer.run_daily_cycle(
            signals=signals,
            prices=prices,
            trade_date=trade_date,
        )
        logger.info(
            f"  {trade_date}: NAV=${obs.nav:,.2f} PnL={obs.daily_pnl:+,.2f} "
            f"signals={obs.signals_received} filled={obs.orders_filled}"
        )
        filled += 1

    logger.info(f"Backfill complete: {filled} days inserted, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
