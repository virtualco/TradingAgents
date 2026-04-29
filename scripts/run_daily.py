#!/usr/bin/env python3
"""Daily TradingAgents Observation Runner.

Runs one full DailyObserver cycle at market open (09:35 ET).
Designed to be called by cron, Airflow, or a Manus scheduled task.

Usage:
    python3 scripts/run_daily.py
    python3 scripts/run_daily.py --date 2026-04-29
    python3 scripts/run_daily.py --tickers AAPL MSFT NVDA GOOGL TSLA
    python3 scripts/run_daily.py --capital 200000 --dry-run

Environment variables (optional overrides):
    TRADINGAGENTS_DB        Path to SQLite DB (default: data/paper_trading.db)
    TRADINGAGENTS_TICKERS   Comma-separated tickers (default: AAPL,MSFT,NVDA,GOOGL,TSLA,AMZN)
    TRADINGAGENTS_CAPITAL   Initial capital in USD (default: 100000)
    TRADINGAGENTS_REPORTS   Directory for daily reports (default: data/observation_reports)
    OPENAI_API_KEY          Required for Deep Mode signal generation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Ensure repo root is on PYTHONPATH ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from tradingagents.execution.observer import DailyObserver, ObservationConfig
from tradingagents.research.signal_registry import SignalRegistry
from tradingagents.research.strategy_rules import (
    TechnicalStrategyRules,
    FundamentalStrategyRules,
    SentimentStrategyRules,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily")

# ── Default universe ───────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA",
    "AMZN", "META", "JPM", "V", "UNH",
]


# ── Price fetcher ──────────────────────────────────────────────────────────

def fetch_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch latest close prices via yfinance (OpenBB fallback)."""
    prices: Dict[str, float] = {}
    try:
        import yfinance as yf
        data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError("yfinance returned empty data")
        close = data["Close"] if "Close" in data.columns else data
        latest = close.iloc[-1]
        for ticker in tickers:
            if ticker in latest.index and not pd.isna(latest[ticker]):
                prices[ticker] = float(latest[ticker])
        logger.info(f"Fetched prices for {len(prices)}/{len(tickers)} tickers via yfinance")
    except Exception as e:
        logger.warning(f"yfinance price fetch failed ({e}), trying OpenBB...")
        try:
            from tradingagents.dataflows.openbb_connector import OpenBBConnector
            connector = OpenBBConnector()
            for ticker in tickers:
                try:
                    df = connector.get_ohlcv(ticker, interval="1d", lookback_days=3)
                    if df is not None and not df.empty:
                        prices[ticker] = float(df["close"].iloc[-1])
                except Exception:
                    pass
            logger.info(f"Fetched prices for {len(prices)}/{len(tickers)} tickers via OpenBB")
        except Exception as e2:
            logger.error(f"All price fetches failed: {e2}")
    return prices


# ── Signal generator ───────────────────────────────────────────────────────

def generate_signals(tickers: List[str], prices: Dict[str, float], trade_date: str) -> pd.DataFrame:
    """Generate rule-based signals for each ticker using strategy rules.

    Uses Quick Mode (no LLM) so it works without API keys.
    Returns a DataFrame with columns: ticker, direction, conviction, signal_id.
    """
    rows = []
    tech_rules = TechnicalStrategyRules()
    fund_rules = FundamentalStrategyRules()
    sent_rules = SentimentStrategyRules()

    for ticker in tickers:
        if ticker not in prices:
            continue
        try:
            # Fetch OHLCV history for technical indicators
            import yfinance as yf
            hist = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
            if hist.empty or len(hist) < 20:
                logger.warning(f"Insufficient history for {ticker}, skipping")
                continue

            # yfinance returns multi-level columns (Price, Ticker) — flatten first
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            ohlcv = hist.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume"
            })

            # Technical signal
            tech_summary = tech_rules.compute(ohlcv, ticker=ticker, trade_date=trade_date)

            # Ensemble: use technical only (fundamentals/sentiment need additional data)
            ensemble_score = tech_summary.composite_score

            # Map score to direction + conviction
            # Score range: -1.0 (strong sell) to +1.0 (strong buy)
            if ensemble_score >= 0.3:
                direction = "long"
                conviction = min(0.95, 0.5 + ensemble_score * 0.5)
            elif ensemble_score <= -0.3:
                direction = "short"
                conviction = min(0.95, 0.5 + abs(ensemble_score) * 0.5)
            else:
                direction = "flat"
                conviction = 0.0

            if direction != "flat":
                import hashlib
                signal_id = hashlib.sha256(
                    f"{ticker}-{trade_date}-{direction}".encode()
                ).hexdigest()[:16]
                rows.append({
                    "ticker": ticker,
                    "direction": direction,
                    "conviction": round(conviction, 4),
                    "signal_id": signal_id,
                    "ensemble_score": round(ensemble_score, 4),
                    "tech_score": round(tech_summary.composite_score, 4),
                })
                logger.info(
                    f"  {ticker}: {direction.upper()} conviction={conviction:.2f} "
                    f"(tech={tech_summary.composite_score:.2f})"
                )
            else:
                logger.info(f"  {ticker}: FLAT (score={ensemble_score:.2f})")

        except Exception as e:
            logger.warning(f"Signal generation failed for {ticker}: {e}")

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} actionable signals from {len(tickers)} tickers")
    return df


# ── Save signals to registry ───────────────────────────────────────────────

def persist_signals(signals: pd.DataFrame, db_path: str, trade_date: str) -> None:
    """Persist generated signals to the signal registry for tracking."""
    if signals.empty:
        return
    try:
        registry = SignalRegistry(db_path=db_path)
        for _, row in signals.iterrows():
            registry.save_signal(
                signal_id=row["signal_id"],
                ticker=row["ticker"],
                trade_date=trade_date,
                direction=row["direction"],
                conviction=row["conviction"],
                ensemble_score=row.get("ensemble_score", row["conviction"]),
                role_scores={"technical": row.get("tech_score", 0.0)},
                context=f"Daily runner — {trade_date}",
            )
        logger.info(f"Persisted {len(signals)} signals to registry")
    except Exception as e:
        logger.warning(f"Signal registry persist failed (non-fatal): {e}")


# ── Save daily report ──────────────────────────────────────────────────────

def save_report(obs, report_dir: str, trade_date: str) -> Path:
    """Save the daily observation as a JSON report."""
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    report_file = report_path / f"daily_{trade_date}.json"

    import dataclasses
    report_data = dataclasses.asdict(obs)
    report_file.write_text(json.dumps(report_data, indent=2, default=str))
    logger.info(f"Daily report saved to {report_file}")
    return report_file


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TradingAgents Daily Observation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Trade date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated list of tickers to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Initial capital in USD (default: 100000)"
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--reports", type=str, default=None,
        help="Directory for daily report files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate signals and prices but do not submit orders"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print the full observation period summary after running"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Resolve configuration from args → env → defaults ──────────────────
    trade_date = args.date or date.today().isoformat()
    tickers_env = os.environ.get("TRADINGAGENTS_TICKERS", "")
    tickers = (
        args.tickers
        or (tickers_env.split(",") if tickers_env else None)
        or DEFAULT_TICKERS
    )
    capital = (
        args.capital
        or float(os.environ.get("TRADINGAGENTS_CAPITAL", "100000"))
    )
    db_path = (
        args.db
        or os.environ.get("TRADINGAGENTS_DB", str(REPO_ROOT / "data" / "paper_trading.db"))
    )
    report_dir = (
        args.reports
        or os.environ.get("TRADINGAGENTS_REPORTS", str(REPO_ROOT / "data" / "observation_reports"))
    )

    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"TradingAgents Daily Runner — {trade_date}")
    logger.info(f"  Tickers  : {', '.join(tickers)}")
    logger.info(f"  Capital  : ${capital:,.0f}")
    logger.info(f"  DB       : {db_path}")
    logger.info(f"  Reports  : {report_dir}")
    logger.info(f"  Dry Run  : {args.dry_run}")
    logger.info("=" * 60)

    # ── Step 1: Fetch prices ───────────────────────────────────────────────
    logger.info("Step 1/4 — Fetching prices...")
    prices = fetch_prices(tickers)
    if not prices:
        logger.error("No prices fetched — aborting daily cycle")
        return 1
    logger.info(f"  Prices: { {t: f'${p:.2f}' for t, p in list(prices.items())[:5]} }...")

    # ── Step 2: Generate signals ───────────────────────────────────────────
    logger.info("Step 2/4 — Generating signals...")
    signals = generate_signals(list(prices.keys()), prices, trade_date)

    # Persist signals to registry
    persist_signals(signals, db_path, trade_date)

    if args.dry_run:
        logger.info("DRY RUN — skipping order submission")
        print("\n[DRY RUN] Signals generated:")
        print(signals.to_string(index=False) if not signals.empty else "  (no actionable signals)")
        return 0

    # ── Step 3: Run daily observation cycle ───────────────────────────────
    logger.info("Step 3/4 — Running daily observation cycle...")
    config = ObservationConfig(
        db_path=db_path,
        initial_capital=capital,
        report_dir=report_dir,
    )
    observer = DailyObserver(config=config)

    obs = observer.run_daily_cycle(
        signals=signals,
        prices=prices,
        trade_date=trade_date,
    )

    # ── Step 4: Save report ────────────────────────────────────────────────
    logger.info("Step 4/4 — Saving daily report...")
    report_file = save_report(obs, report_dir, trade_date)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"DAILY CYCLE COMPLETE — {trade_date}")
    print("=" * 60)
    print(f"  NAV              : ${obs.nav:>12,.2f}")
    print(f"  Daily P&L        : ${obs.daily_pnl:>+12,.2f}")
    print(f"  Total P&L        : ${obs.total_pnl:>+12,.2f}")
    print(f"  Drawdown         : {obs.drawdown_pct * 100:>8.2f}%")
    print(f"  Signals received : {obs.signals_received:>5}")
    print(f"  Orders submitted : {obs.orders_submitted:>5}")
    print(f"  Orders filled    : {obs.orders_filled:>5}")
    print(f"  Orders rejected  : {obs.orders_rejected:>5}")
    print(f"  Kill switch      : {'ACTIVE' if obs.kill_switch_active else 'OK'}")
    print(f"  Circuit breaker  : {'TRIGGERED' if obs.circuit_breaker_triggered else 'OK'}")
    print(f"  Reconciliation   : {'CLEAN' if obs.reconciliation_clean else f'{obs.reconciliation_breaks} BREAKS'}")
    print(f"  Report           : {report_file}")
    print("=" * 60)

    if obs.notes:
        print(f"\nNotes: {obs.notes}")

    # ── Optional: print full observation period summary ────────────────────
    if args.summary:
        from tradingagents.execution.observer import ObservationLogger
        obs_logger = ObservationLogger(db_path=db_path)
        summary = obs_logger.get_summary()
        print("\n" + summary.summary())

    # Return non-zero if kill switch or circuit breaker fired
    if obs.kill_switch_active or obs.circuit_breaker_triggered:
        logger.warning("Kill switch or circuit breaker is active — review before next run")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
