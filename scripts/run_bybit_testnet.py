"""
Bybit Testnet Paper Trading Simulation
=======================================
Runs the CryptoDayTradingStrategy in real-time against Bybit Testnet.
Fetches live 1-hour candles from Bybit, generates signals, and places
paper orders on the testnet (no real money).

Features:
  - Hourly signal generation loop
  - ATR-based position sizing (1% account risk per trade)
  - Exchange-level stop-loss orders
  - State persistence (JSON) for crash recovery
  - Detailed trade log with P&L tracking
  - Kill switch via environment variable KILL_SWITCH=1

Setup:
  1. Create a Bybit Testnet account at https://testnet.bybit.com
  2. Generate API keys (Testnet)
  3. Set environment variables:
       BYBIT_TESTNET_API_KEY=your_key
       BYBIT_TESTNET_API_SECRET=your_secret
  4. Run: python3 scripts/run_bybit_testnet.py

Usage:
    python3 scripts/run_bybit_testnet.py [--dry-run]
    python3 scripts/run_bybit_testnet.py --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from tradingagents.execution.bybit_connector import BybitConnector
from tradingagents.research.crypto_strategy import CryptoDayTradingStrategy

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS         = ["BTCUSDT", "ETHUSDT"]
CANDLE_INTERVAL = "60"          # 1-hour candles
CANDLE_LIMIT    = 300           # Lookback for indicator warmup
ACCOUNT_RISK    = 0.01          # 1% account risk per trade
MAX_LEVERAGE    = 3             # Maximum leverage
LOOP_SLEEP_SEC  = 60            # Check every 60 seconds
STATE_FILE      = Path("data/testnet_state.json")
TRADE_LOG       = Path("data/testnet_trades.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] testnet: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("testnet")


# ── State Management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "last_signal": {}, "last_candle_time": {}, "trade_count": 0}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def log_trade(trade: dict):
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(trade) + "\n")


# ── Position Sizing ───────────────────────────────────────────────────────────

def calculate_qty(
    balance: float,
    price: float,
    atr: float,
    atr_stop_mult: float = 2.5,
    max_leverage: int = MAX_LEVERAGE,
    min_qty: float = 0.001,
) -> float:
    """
    ATR-based position sizing: risk 1% of account per trade.
    Stop distance = atr_stop_mult * ATR
    Position size = (account_risk * balance) / stop_distance
    """
    stop_distance = atr_stop_mult * atr
    if stop_distance <= 0 or price <= 0:
        return min_qty

    risk_amount = ACCOUNT_RISK * balance
    qty_by_risk = risk_amount / stop_distance

    # Cap by max leverage
    max_qty = (balance * max_leverage) / price
    qty = min(qty_by_risk, max_qty)

    # Round to 3 decimal places (BTC min lot)
    qty = max(round(qty, 3), min_qty)
    return qty


# ── Candle Fetching ───────────────────────────────────────────────────────────

def fetch_candles(conn: BybitConnector, symbol: str) -> pd.DataFrame:
    """Fetch recent OHLCV candles and return as DataFrame."""
    raw = conn.get_klines(symbol, interval=CANDLE_INTERVAL, limit=CANDLE_LIMIT)
    if not raw:
        return pd.DataFrame()

    # Bybit returns newest first — reverse to chronological
    raw = list(reversed(raw))
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


# ── ATR Calculation ───────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR for position sizing."""
    if len(df) < period:
        return float(df["close"].iloc[-1]) * 0.02  # fallback: 2% of price

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_testnet(dry_run: bool = False, symbols: list = None):
    if symbols is None:
        symbols = SYMBOLS

    log.info("=" * 65)
    log.info("BYBIT TESTNET PAPER TRADING — CryptoDayTradingStrategy")
    log.info(f"  Symbols: {symbols}")
    log.info(f"  Dry Run: {dry_run}")
    log.info(f"  Account Risk: {ACCOUNT_RISK*100:.1f}% per trade")
    log.info(f"  Max Leverage: {MAX_LEVERAGE}x")
    log.info("=" * 65)

    # Initialise connector
    conn = BybitConnector(testnet=True, category="linear")
    conn.set_credentials()

    # Test connectivity
    if not conn.ping():
        log.error("Cannot connect to Bybit Testnet API. Check credentials and network.")
        sys.exit(1)
    log.info(f"Connected to Bybit Testnet | Server time: {conn.get_server_time()}")

    # Initialise strategy
    strategy = CryptoDayTradingStrategy()
    state = load_state()

    # Set leverage for all symbols
    if not dry_run:
        for symbol in symbols:
            conn.set_leverage(symbol, MAX_LEVERAGE)

    log.info("Starting main loop... (Ctrl+C to stop)")

    while True:
        # Kill switch check
        if os.getenv("KILL_SWITCH", "0") == "1":
            log.warning("KILL SWITCH activated — closing all positions and exiting")
            if not dry_run:
                for symbol in symbols:
                    conn.close_position(symbol)
            break

        try:
            # Get account balance
            balance_obj = conn.get_balance("USDT")
            balance = balance_obj.total_equity
            log.info(f"Balance: ${balance:,.2f} USDT | Available: ${balance_obj.available_balance:,.2f}")

            for symbol in symbols:
                try:
                    _process_symbol(conn, strategy, symbol, balance, state, dry_run)
                except Exception as e:
                    log.error(f"Error processing {symbol}: {e}")

            save_state(state)

        except KeyboardInterrupt:
            log.info("Interrupted by user — saving state and exiting")
            save_state(state)
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")

        log.info(f"Sleeping {LOOP_SLEEP_SEC}s until next check...")
        time.sleep(LOOP_SLEEP_SEC)


def _process_symbol(
    conn: BybitConnector,
    strategy: CryptoDayTradingStrategy,
    symbol: str,
    balance: float,
    state: dict,
    dry_run: bool,
):
    """Process one symbol: fetch candles, generate signal, execute if needed."""
    df = fetch_candles(conn, symbol)
    if df is None or len(df) < 150:
        log.warning(f"{symbol}: Insufficient candle data ({len(df) if df is not None else 0} rows)")
        return

    # Check if we have a new candle since last check
    latest_candle_time = str(df.index[-1])
    if state["last_candle_time"].get(symbol) == latest_candle_time:
        log.debug(f"{symbol}: No new candle — skipping")
        return
    state["last_candle_time"][symbol] = latest_candle_time

    # Generate signal
    signals = strategy.generate_signals(df)
    current_signal = int(signals.iloc[-1])
    last_signal = state["last_signal"].get(symbol, 0)

    current_price = float(df["close"].iloc[-1])
    atr = compute_atr(df)

    log.info(
        f"{symbol} | price={current_price:,.2f} | ATR={atr:.2f} | "
        f"signal={current_signal:+d} | last_signal={last_signal:+d}"
    )

    # No change in signal
    if current_signal == last_signal:
        return

    # Signal changed — execute
    state["last_signal"][symbol] = current_signal

    # Close existing position if signal reversed or went flat
    if last_signal != 0 and (current_signal == 0 or current_signal != last_signal):
        log.info(f"{symbol}: Signal change {last_signal:+d} → {current_signal:+d} — closing position")
        if not dry_run:
            result = conn.close_position(symbol)
            log.info(f"  Close result: {result}")
            log_trade({
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": "CLOSE",
                "side": "Sell" if last_signal == 1 else "Buy",
                "price": current_price,
                "dry_run": dry_run,
            })
        else:
            log.info(f"  [DRY RUN] Would close {symbol} position")

    # Open new position if signal is non-zero
    if current_signal != 0:
        side = "Buy" if current_signal == 1 else "Sell"
        qty = calculate_qty(balance, current_price, atr)
        log.info(f"{symbol}: Opening {side} | qty={qty} | price={current_price:,.2f} | ATR={atr:.2f}")

        if not dry_run:
            result = conn.place_market_order(symbol, side, qty)
            log.info(f"  Order result: {result}")
            state["trade_count"] = state.get("trade_count", 0) + 1

            # Place stop-loss order
            stop_price = current_price - 2.5 * atr if side == "Buy" else current_price + 2.5 * atr
            stop_price = round(stop_price, 2)
            stop_side = "Sell" if side == "Buy" else "Buy"
            stop_result = conn.place_limit_order(symbol, stop_side, qty, stop_price, reduce_only=True)
            log.info(f"  Stop-loss at {stop_price:,.2f}: {stop_result}")

            log_trade({
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "action": "OPEN",
                "side": side,
                "qty": qty,
                "price": current_price,
                "stop_price": stop_price,
                "atr": atr,
                "order_id": result.order_id,
                "dry_run": dry_run,
            })
        else:
            log.info(
                f"  [DRY RUN] Would place {side} {qty} {symbol} @ market | "
                f"stop @ {current_price - 2.5*atr if side=='Buy' else current_price + 2.5*atr:,.2f}"
            )


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bybit Testnet Paper Trading")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without placing orders")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS, help="Symbols to trade")
    args = parser.parse_args()

    run_testnet(dry_run=args.dry_run, symbols=args.symbols)


if __name__ == "__main__":
    main()
