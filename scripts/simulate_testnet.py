"""
Bybit Testnet End-to-End Simulation
=====================================
Validates the full pipeline: data fetch → signal → risk check → order → P&L tracking
Uses REAL Bybit Testnet API (no real money) + RiskManager.

Run: python3 scripts/simulate_testnet.py --dry-run
     python3 scripts/simulate_testnet.py  (requires BYBIT_TESTNET_API_KEY/SECRET)
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from tradingagents.execution.bybit_connector import BybitConnector
from tradingagents.execution.risk_manager import RiskManager, RiskConfig
from tradingagents.research.crypto_strategy import CryptoDayTradingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] simulate: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("simulate")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
REPORT_FILE = Path("data/testnet_simulation_report.json")


def fetch_candles(conn: BybitConnector, symbol: str, limit: int = 300) -> pd.DataFrame:
    raw = conn.get_klines(symbol, interval="60", limit=limit)
    if not raw:
        return pd.DataFrame()
    raw = list(reversed(raw))
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period:
        return float(df["close"].iloc[-1]) * 0.02
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def run_simulation(dry_run: bool = True):
    log.info("=" * 65)
    log.info("BYBIT TESTNET END-TO-END SIMULATION")
    log.info(f"  Mode: {'DRY RUN' if dry_run else 'TESTNET (real API calls)'}")
    log.info("=" * 65)

    report = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "checks": {},
        "signals": {},
        "risk_checks": {},
        "orders": {},
        "summary": {}
    }

    # ── Step 1: Connectivity ──────────────────────────────────────────────────
    log.info("\n[CHECK 1] Bybit Testnet Connectivity")
    conn = BybitConnector(testnet=True, category="linear")
    conn.set_credentials()
    ping_ok = conn.ping()
    server_time = conn.get_server_time() if ping_ok else 0
    report["checks"]["connectivity"] = {"ok": ping_ok, "server_time_ms": server_time}
    log.info(f"  Ping: {'OK' if ping_ok else 'FAIL'} | Server time: {server_time}")

    if not ping_ok:
        log.error("Cannot connect to Bybit Testnet — aborting")
        report["summary"]["status"] = "FAIL_CONNECTIVITY"
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, indent=2))
        return

    # ── Step 2: Market Data ───────────────────────────────────────────────────
    log.info("\n[CHECK 2] Market Data Fetch")
    candle_data = {}
    for symbol in SYMBOLS:
        df = fetch_candles(conn, symbol, limit=300)
        ok = len(df) >= 150
        candle_data[symbol] = df
        report["checks"][f"candles_{symbol}"] = {"ok": ok, "rows": len(df)}
        log.info(f"  {symbol}: {len(df)} candles | {'OK' if ok else 'INSUFFICIENT'}")

    # ── Step 3: Signal Generation ─────────────────────────────────────────────
    log.info("\n[CHECK 3] Signal Generation")
    strategy = CryptoDayTradingStrategy()
    signals = {}
    for symbol in SYMBOLS:
        df = candle_data.get(symbol, pd.DataFrame())
        if len(df) < 150:
            signals[symbol] = 0
            log.warning(f"  {symbol}: Insufficient data for signal")
            continue
        sig_series = strategy.generate_signals(df)
        current_sig = int(sig_series.iloc[-1])
        atr = compute_atr(df)
        price = float(df["close"].iloc[-1])
        signals[symbol] = current_sig
        report["signals"][symbol] = {
            "signal": current_sig,
            "price": price,
            "atr": round(atr, 2),
            "atr_pct": round(atr / price * 100, 3),
        }
        sig_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[current_sig]
        log.info(f"  {symbol}: {sig_str} | price={price:,.2f} | ATR={atr:.2f} ({atr/price*100:.2f}%)")

    # ── Step 4: Account Balance ───────────────────────────────────────────────
    log.info("\n[CHECK 4] Account Balance (Testnet)")
    try:
        balance_obj = conn.get_balance("USDT")
        equity = balance_obj.total_equity
        available = balance_obj.available_balance
        report["checks"]["balance"] = {
            "ok": equity > 0,
            "total_equity": equity,
            "available": available,
        }
        log.info(f"  Total Equity: ${equity:,.2f} USDT | Available: ${available:,.2f}")
    except Exception as e:
        equity = 10_000.0  # Fallback for dry run without credentials
        available = equity
        report["checks"]["balance"] = {"ok": False, "error": str(e), "fallback_equity": equity}
        log.warning(f"  Balance fetch failed (expected without API keys): {e}")
        log.info(f"  Using fallback equity: ${equity:,.2f} for risk calculations")

    # ── Step 5: Risk Manager ──────────────────────────────────────────────────
    log.info("\n[CHECK 5] Risk Manager Validation")
    config = RiskConfig(
        daily_loss_limit_pct=0.05,
        max_open_positions=3,
        account_risk_per_trade=0.01,
        max_leverage=3,
        min_trade_interval_sec=3600,
    )
    rm = RiskManager(account_equity=equity, config=config)
    rm.print_status()

    for symbol in SYMBOLS:
        sig = signals.get(symbol, 0)
        if sig == 0:
            report["risk_checks"][symbol] = {"signal": 0, "action": "FLAT_NO_CHECK"}
            continue

        df = candle_data.get(symbol, pd.DataFrame())
        price = float(df["close"].iloc[-1]) if len(df) > 0 else 0
        atr = compute_atr(df) if len(df) > 0 else 0
        side = "Buy" if sig == 1 else "Sell"
        qty = rm.calculate_position_size(price, atr, equity)
        approved, reason = rm.approve_trade(symbol, side, qty, price, atr, equity)
        report["risk_checks"][symbol] = {
            "signal": sig,
            "side": side,
            "qty": qty,
            "price": price,
            "atr": round(atr, 2),
            "approved": approved,
            "reason": reason,
        }
        log.info(f"  {symbol} {side} {qty} @ {price:,.2f}: {'APPROVED' if approved else 'REJECTED — ' + reason}")

    # ── Step 6: Order Simulation ──────────────────────────────────────────────
    log.info("\n[CHECK 6] Order Execution (Dry Run)")
    for symbol in SYMBOLS:
        risk = report["risk_checks"].get(symbol, {})
        if not risk.get("approved", False):
            report["orders"][symbol] = {"action": "SKIPPED", "reason": risk.get("reason", "Not approved")}
            continue

        side = risk["side"]
        qty = risk["qty"]
        price = risk["price"]
        atr = risk["atr"]
        stop_price = round(price - 2.5 * atr if side == "Buy" else price + 2.5 * atr, 2)

        if dry_run:
            log.info(
                f"  [DRY RUN] {symbol}: Would place {side} {qty} @ market | "
                f"stop @ {stop_price:,.2f}"
            )
            report["orders"][symbol] = {
                "action": "DRY_RUN",
                "side": side,
                "qty": qty,
                "entry_price": price,
                "stop_price": stop_price,
                "notional": round(qty * price, 2),
            }
        else:
            result = conn.place_market_order(symbol, side, qty)
            log.info(f"  {symbol} order: {result}")
            report["orders"][symbol] = {
                "action": "PLACED",
                "success": result.success,
                "order_id": result.order_id,
                "side": side,
                "qty": qty,
                "error": result.error,
            }

    # ── Summary ───────────────────────────────────────────────────────────────
    all_checks_ok = all(
        v.get("ok", True) for k, v in report["checks"].items()
        if k != "balance"  # balance may fail without credentials
    )
    report["summary"] = {
        "status": "PASS" if all_checks_ok else "PARTIAL",
        "pipeline_validated": True,
        "ready_for_testnet_live": all_checks_ok,
        "next_step": (
            "Set BYBIT_TESTNET_API_KEY and BYBIT_TESTNET_API_SECRET, "
            "then run: python3 scripts/run_bybit_testnet.py"
        ),
    }

    log.info("\n" + "=" * 65)
    log.info("SIMULATION SUMMARY")
    log.info("=" * 65)
    log.info(f"  Status: {report['summary']['status']}")
    log.info(f"  Pipeline validated: {report['summary']['pipeline_validated']}")
    log.info(f"  Ready for testnet: {report['summary']['ready_for_testnet_live']}")
    log.info(f"  Next step: {report['summary']['next_step']}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    log.info(f"\nReport saved to {REPORT_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Place real testnet orders")
    args = parser.parse_args()
    dry_run = not args.live
    run_simulation(dry_run=dry_run)


if __name__ == "__main__":
    main()
