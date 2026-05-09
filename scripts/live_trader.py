"""
TradingAgents — Production Live Trading Orchestrator
=====================================================
Integrates:
  - DualRegimeStrategy (Hurst + ADX regime detection → Momentum / Mean-Reversion)
  - BybitConnector (Bybit V5 Unified API — Testnet or Mainnet)
  - RiskManager (circuit breakers, ATR position sizing, kill switch)
  - SQLite state persistence (positions, trades, regime history)
  - Prometheus-compatible metrics export (for dashboard)

Usage:
  # Testnet (paper trading)
  export BYBIT_API_KEY="your_testnet_key"
  export BYBIT_API_SECRET="your_testnet_secret"
  export BYBIT_TESTNET=true
  python3 scripts/live_trader.py

  # Mainnet (LIVE — real money)
  export BYBIT_API_KEY="your_mainnet_key"
  export BYBIT_API_SECRET="your_mainnet_secret"
  export BYBIT_TESTNET=false
  python3 scripts/live_trader.py

Configuration via environment variables (all optional):
  TRADING_SYMBOLS       Comma-separated symbols (default: BTCUSDT,ETHUSDT)
  TRADING_INTERVAL      Candle interval in minutes (default: 60)
  TRADING_CAPITAL_USDT  Capital allocation per symbol in USDT (default: 1000)
  TRADING_MAX_POSITIONS Max concurrent open positions (default: 2)
  TRADING_MAX_DD_PCT    Max daily drawdown % before circuit break (default: 5)
  TRADING_LOG_LEVEL     Logging level (default: INFO)
  TRADING_STATE_DB      Path to SQLite state DB (default: data/live_trading.db)
  TRADING_DRY_RUN       If true, generate signals but skip order placement (default: false)
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tradingagents.research.dual_regime_strategy import DualRegimeStrategy
from tradingagents.execution.bybit_connector import BybitConnector
from tradingagents.execution.risk_manager import RiskManager

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("TRADING_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "data" / "live_trader.log", mode="a"),
    ],
)
log = logging.getLogger("live_trader")


# ── Configuration ─────────────────────────────────────────────────────────────

class TradingConfig:
    """Central configuration loaded from environment variables."""

    def __init__(self):
        self.api_key       = os.getenv("BYBIT_API_KEY", "")
        self.api_secret    = os.getenv("BYBIT_API_SECRET", "")
        self.testnet       = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
        self.symbols       = os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
        self.interval_min  = int(os.getenv("TRADING_INTERVAL", "60"))
        self.capital_usdt  = float(os.getenv("TRADING_CAPITAL_USDT", "1000"))
        self.max_positions = int(os.getenv("TRADING_MAX_POSITIONS", "2"))
        self.max_dd_pct    = float(os.getenv("TRADING_MAX_DD_PCT", "5.0"))
        self.state_db      = Path(os.getenv("TRADING_STATE_DB", str(ROOT / "data" / "live_trading.db")))
        self.dry_run       = os.getenv("TRADING_DRY_RUN", "false").lower() == "true"
        self.metrics_port  = int(os.getenv("TRADING_METRICS_PORT", "8765"))

    def validate(self):
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "BYBIT_API_KEY and BYBIT_API_SECRET must be set.\n"
                "For testnet: get keys at https://testnet.bybit.com/app/user/api-management"
            )
        if self.interval_min not in [1, 3, 5, 15, 30, 60, 120, 240, 360, 720]:
            raise ValueError(f"Invalid interval: {self.interval_min}. Must be one of [1,3,5,15,30,60,120,240,360,720]")
        log.info(
            f"Config: symbols={self.symbols} interval={self.interval_min}m "
            f"capital={self.capital_usdt}USDT testnet={self.testnet} dry_run={self.dry_run}"
        )


# ── State Database ────────────────────────────────────────────────────────────

class StateDB:
    """SQLite-backed state persistence for positions, trades, and regime history."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol      TEXT PRIMARY KEY,
                side        TEXT,
                qty         REAL,
                entry_price REAL,
                entry_time  TEXT,
                stop_loss   REAL,
                take_profit REAL,
                regime      TEXT,
                strategy    TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT,
                side        TEXT,
                qty         REAL,
                price       REAL,
                pnl         REAL,
                regime      TEXT,
                strategy    TEXT,
                timestamp   TEXT
            );

            CREATE TABLE IF NOT EXISTS regime_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT,
                regime      TEXT,
                hurst       REAL,
                adx         REAL,
                timestamp   TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                date        TEXT PRIMARY KEY,
                realized_pnl REAL,
                unrealized_pnl REAL,
                n_trades    INTEGER,
                peak_nav    REAL
            );
        """)
        self.conn.commit()

    def upsert_position(self, symbol: str, side: str, qty: float, entry_price: float,
                        stop_loss: float, take_profit: float, regime: str, strategy: str):
        self.conn.execute("""
            INSERT OR REPLACE INTO positions
            (symbol, side, qty, entry_price, entry_time, stop_loss, take_profit, regime, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, side, qty, entry_price, datetime.now(timezone.utc).isoformat(),
              stop_loss, take_profit, regime, strategy))
        self.conn.commit()

    def remove_position(self, symbol: str):
        self.conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        self.conn.commit()

    def get_position(self, symbol: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row:
            cols = ["symbol", "side", "qty", "entry_price", "entry_time",
                    "stop_loss", "take_profit", "regime", "strategy"]
            return dict(zip(cols, row))
        return None

    def log_trade(self, symbol: str, side: str, qty: float, price: float,
                  pnl: float, regime: str, strategy: str):
        self.conn.execute("""
            INSERT INTO trades (symbol, side, qty, price, pnl, regime, strategy, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, side, qty, price, pnl, regime, strategy,
              datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def log_regime(self, symbol: str, regime: str, hurst: float, adx: float):
        self.conn.execute("""
            INSERT INTO regime_history (symbol, regime, hurst, adx, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, regime, hurst, adx, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def get_recent_trades(self, limit: int = 50) -> list:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = ["id", "symbol", "side", "qty", "price", "pnl", "regime", "strategy", "timestamp"]
        return [dict(zip(cols, r)) for r in rows]

    def get_daily_stats(self) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        row = self.conn.execute(
            "SELECT * FROM daily_pnl WHERE date = ?", (today,)
        ).fetchone()
        if row:
            cols = ["date", "realized_pnl", "unrealized_pnl", "n_trades", "peak_nav"]
            return dict(zip(cols, row))
        return {"date": today, "realized_pnl": 0.0, "unrealized_pnl": 0.0, "n_trades": 0, "peak_nav": 0.0}


# ── Signal Processor ──────────────────────────────────────────────────────────

class SignalProcessor:
    """
    Fetches candle data, runs the DualRegimeStrategy, and returns
    structured signal objects with regime diagnostics.
    """

    BYBIT_INTERVAL_MAP = {
        1: "1", 3: "3", 5: "5", 15: "15", 30: "30",
        60: "60", 120: "120", 240: "240", 360: "360", 720: "720",
    }

    def __init__(self, strategy: DualRegimeStrategy, connector: BybitConnector,
                 interval_min: int = 60, lookback_bars: int = 300):
        self.strategy      = strategy
        self.connector     = connector
        self.interval_min  = interval_min
        self.lookback_bars = lookback_bars
        self.interval_str  = self.BYBIT_INTERVAL_MAP[interval_min]

    def fetch_candles(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV candles from Bybit."""
        try:
            result = self.connector.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=self.interval_str,
                limit=self.lookback_bars,
            )
            if result["retCode"] != 0:
                raise RuntimeError(f"Bybit kline error: {result['retMsg']}")

            rows = result["result"]["list"]
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df = df.astype({"timestamp": "int64", "open": "float64", "high": "float64",
                            "low": "float64", "close": "float64", "volume": "float64"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            return df
        except Exception as e:
            log.error(f"Failed to fetch candles for {symbol}: {e}")
            raise

    def generate_signal(self, symbol: str) -> dict:
        """
        Fetch candles, run dual-regime strategy, return signal dict.
        Returns:
            {
                symbol, signal (+1/-1/0), regime, hurst, adx,
                sub_strategy, close_price, atr, timestamp
            }
        """
        df = self.fetch_candles(symbol)
        if len(df) < 150:
            log.warning(f"{symbol}: Insufficient candles ({len(df)}), returning FLAT")
            return self._flat_signal(symbol, df["close"].iloc[-1] if len(df) > 0 else 0)

        diag = self.strategy.get_diagnostics(df)
        latest = diag.iloc[-1]

        # Determine which sub-strategy is active
        sub_strategy = "NONE"
        if latest["regime"] == "TRENDING":
            sub_strategy = "MOMENTUM"
        elif latest["regime"] == "RANGING":
            sub_strategy = "MEAN_REVERSION"

        # ATR for position sizing
        from tradingagents.research.dual_regime_strategy import _atr
        atr = _atr(df).iloc[-1]

        return {
            "symbol":       symbol,
            "signal":       int(latest["combined_sig"]),
            "regime":       str(latest["regime"]),
            "hurst":        round(float(latest["hurst"]), 4),
            "adx":          round(float(latest["adx"]), 2),
            "sub_strategy": sub_strategy,
            "close_price":  round(float(latest["close"]), 4),
            "atr":          round(float(atr), 4),
            "bb_upper":     round(float(latest["bb_upper"]), 4),
            "bb_mid":       round(float(latest["bb_mid"]), 4),
            "bb_lower":     round(float(latest["bb_lower"]), 4),
            "rsi":          round(float(latest["rsi"]), 2),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        }

    def _flat_signal(self, symbol: str, price: float) -> dict:
        return {
            "symbol": symbol, "signal": 0, "regime": "TRANSITION",
            "hurst": 0.5, "adx": 0.0, "sub_strategy": "NONE",
            "close_price": price, "atr": 0.0,
            "bb_upper": 0.0, "bb_mid": 0.0, "bb_lower": 0.0, "rsi": 50.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Order Executor ────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Translates signals into Bybit orders, applying risk manager checks.
    Handles entry, exit, stop-loss, and take-profit logic.
    """

    def __init__(self, connector: BybitConnector, risk_manager: RiskManager,
                 state_db: StateDB, capital_usdt: float, dry_run: bool = False):
        self.connector    = connector
        self.risk_manager = risk_manager
        self.state_db     = state_db
        self.capital_usdt = capital_usdt
        self.dry_run      = dry_run

    def process_signal(self, sig: dict) -> dict:
        """
        Process a signal and execute orders as appropriate.
        Returns an execution report dict.
        """
        symbol      = sig["symbol"]
        new_signal  = sig["signal"]
        price       = sig["close_price"]
        atr         = sig["atr"]
        regime      = sig["regime"]
        sub_strategy = sig["sub_strategy"]

        # Log regime to DB
        self.state_db.log_regime(symbol, regime, sig["hurst"], sig["adx"])

        # Get current position
        current_pos = self.state_db.get_position(symbol)
        current_side = current_pos["side"] if current_pos else None

        report = {
            "symbol": symbol, "action": "NONE", "regime": regime,
            "sub_strategy": sub_strategy, "signal": new_signal,
            "price": price, "timestamp": sig["timestamp"],
        }

        # ── Check stop-loss / take-profit on existing position ──
        if current_pos:
            hit_stop = self._check_stops(current_pos, price)
            if hit_stop:
                report.update(self._close_position(symbol, current_pos, price, "STOP_HIT"))
                current_pos = None
                current_side = None

        # ── Regime change: close position if regime switches ──
        if current_pos and regime == "TRANSITION":
            log.info(f"{symbol}: Regime → TRANSITION, closing position")
            report.update(self._close_position(symbol, current_pos, price, "REGIME_CHANGE"))
            current_pos = None
            current_side = None

        # ── No signal → hold or stay flat ──
        if new_signal == 0:
            report["action"] = "HOLD" if current_pos else "FLAT"
            return report

        # ── Signal reversal: close existing position first ──
        if current_pos and current_side != ("LONG" if new_signal == 1 else "SHORT"):
            log.info(f"{symbol}: Signal reversal, closing {current_side}")
            self._close_position(symbol, current_pos, price, "SIGNAL_REVERSAL")
            current_pos = None
            current_side = None

        # ── Open new position ──
        if not current_pos:
            report.update(self._open_position(sig))

        return report

    def _check_stops(self, pos: dict, current_price: float) -> bool:
        """Returns True if stop-loss or take-profit is hit."""
        if pos["side"] == "LONG":
            if current_price <= pos["stop_loss"]:
                log.warning(f"{pos['symbol']}: Stop-loss hit at {current_price:.4f} (stop={pos['stop_loss']:.4f})")
                return True
            if pos["take_profit"] > 0 and current_price >= pos["take_profit"]:
                log.info(f"{pos['symbol']}: Take-profit hit at {current_price:.4f}")
                return True
        elif pos["side"] == "SHORT":
            if current_price >= pos["stop_loss"]:
                log.warning(f"{pos['symbol']}: Stop-loss hit at {current_price:.4f} (stop={pos['stop_loss']:.4f})")
                return True
            if pos["take_profit"] > 0 and current_price <= pos["take_profit"]:
                log.info(f"{pos['symbol']}: Take-profit hit at {current_price:.4f}")
                return True
        return False

    def _compute_position_size(self, price: float, atr: float) -> float:
        """ATR-based position sizing: risk 1% of capital per trade."""
        if atr <= 0 or price <= 0:
            return 0.0
        risk_amount = self.capital_usdt * 0.01   # 1% risk
        stop_distance = atr * 2.0                # 2× ATR stop
        qty = risk_amount / stop_distance
        # Minimum notional check (Bybit requires ≥5 USDT)
        if qty * price < 5.0:
            qty = 5.0 / price
        return round(qty, 4)

    def _open_position(self, sig: dict) -> dict:
        """Open a new position."""
        symbol   = sig["symbol"]
        signal   = sig["signal"]
        price    = sig["close_price"]
        atr      = sig["atr"]
        regime   = sig["regime"]
        strategy = sig["sub_strategy"]

        side = "LONG" if signal == 1 else "SHORT"
        qty  = self._compute_position_size(price, atr)

        if qty <= 0:
            log.warning(f"{symbol}: Zero quantity computed, skipping")
            return {"action": "SKIP", "reason": "zero_qty"}

        # Risk manager pre-trade check
        risk_ok, reason = self.risk_manager.pre_trade_check(
            symbol=symbol, side=side, qty=qty, price=price
        )
        if not risk_ok:
            log.warning(f"{symbol}: Risk manager blocked trade: {reason}")
            return {"action": "BLOCKED", "reason": reason}

        # Stop-loss and take-profit levels
        if side == "LONG":
            stop_loss   = price - 2.0 * atr
            take_profit = price + 3.0 * atr   # 1.5:1 R:R
        else:
            stop_loss   = price + 2.0 * atr
            take_profit = price - 3.0 * atr

        if not self.dry_run:
            try:
                order = self.connector.place_order(
                    symbol=symbol,
                    side="Buy" if side == "LONG" else "Sell",
                    qty=qty,
                    order_type="Market",
                    stop_loss=round(stop_loss, 4),
                    take_profit=round(take_profit, 4),
                )
                order_id = order.get("orderId", "DRY_RUN")
            except Exception as e:
                log.error(f"{symbol}: Order placement failed: {e}")
                return {"action": "ORDER_FAILED", "reason": str(e)}
        else:
            order_id = f"DRY_{symbol}_{int(time.time())}"

        # Persist position
        self.state_db.upsert_position(
            symbol=symbol, side=side, qty=qty, entry_price=price,
            stop_loss=stop_loss, take_profit=take_profit,
            regime=regime, strategy=strategy,
        )

        log.info(
            f"OPENED {side} {symbol}: qty={qty} @ {price:.4f} | "
            f"SL={stop_loss:.4f} TP={take_profit:.4f} | "
            f"regime={regime} strategy={strategy} | order_id={order_id}"
        )

        return {
            "action": "OPEN", "side": side, "qty": qty, "price": price,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "order_id": order_id, "strategy": strategy,
        }

    def _close_position(self, symbol: str, pos: dict, price: float, reason: str) -> dict:
        """Close an existing position."""
        side = pos["side"]
        qty  = pos["qty"]
        pnl  = (price - pos["entry_price"]) * qty if side == "LONG" else (pos["entry_price"] - price) * qty

        if not self.dry_run:
            try:
                close_side = "Sell" if side == "LONG" else "Buy"
                self.connector.place_order(
                    symbol=symbol, side=close_side, qty=qty, order_type="Market"
                )
            except Exception as e:
                log.error(f"{symbol}: Close order failed: {e}")

        self.state_db.remove_position(symbol)
        self.state_db.log_trade(
            symbol=symbol, side=side, qty=qty, price=price,
            pnl=pnl, regime=pos["regime"], strategy=pos["strategy"],
        )
        self.risk_manager.record_trade_pnl(pnl)

        log.info(
            f"CLOSED {side} {symbol}: qty={qty} @ {price:.4f} | "
            f"P&L={pnl:+.4f} USDT | reason={reason}"
        )

        return {"action": "CLOSE", "side": side, "qty": qty, "price": price,
                "pnl": pnl, "reason": reason}


# ── Metrics Server ────────────────────────────────────────────────────────────

class MetricsServer:
    """
    Lightweight HTTP server exposing JSON metrics for the dashboard.
    Runs on a separate thread.
    """

    def __init__(self, port: int, state_db: StateDB):
        self.port     = port
        self.state_db = state_db
        self._latest  = {}

    def update(self, data: dict):
        self._latest = data

    def start(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        state_db = self.state_db
        latest   = self._latest

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    payload = json.dumps({
                        **latest,
                        "recent_trades": state_db.get_recent_trades(20),
                        "daily_stats":   state_db.get_daily_stats(),
                    }, indent=2).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                pass  # suppress access logs

        server = HTTPServer(("0.0.0.0", self.port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        log.info(f"Metrics server started on port {self.port} → GET /metrics")


# ── Main Orchestrator ─────────────────────────────────────────────────────────

class LiveTrader:
    """
    Main trading loop orchestrator.
    Runs one cycle per candle interval, processing all configured symbols.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self._running = True

        # Initialise components
        self.connector = BybitConnector(
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet,
        )
        self.risk_manager = RiskManager(
            max_daily_drawdown_pct=config.max_dd_pct,
            max_positions=config.max_positions,
            state_file=str(ROOT / "data" / "risk_state.json"),
        )
        self.state_db = StateDB(config.state_db)
        self.strategy = DualRegimeStrategy()
        self.signal_processor = SignalProcessor(
            strategy=self.strategy,
            connector=self.connector,
            interval_min=config.interval_min,
        )
        self.order_executor = OrderExecutor(
            connector=self.connector,
            risk_manager=self.risk_manager,
            state_db=self.state_db,
            capital_usdt=config.capital_usdt,
            dry_run=config.dry_run,
        )
        self.metrics = MetricsServer(config.metrics_port, self.state_db)

        # Graceful shutdown
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *args):
        log.info("Shutdown signal received — stopping after current cycle")
        self._running = False

    def _seconds_to_next_candle(self) -> float:
        """Calculate seconds until the next candle close."""
        now_ts = time.time()
        interval_sec = self.config.interval_min * 60
        return interval_sec - (now_ts % interval_sec)

    def run_cycle(self) -> list:
        """Execute one full trading cycle across all symbols."""
        reports = []
        for symbol in self.config.symbols:
            try:
                log.info(f"Processing {symbol}...")
                sig    = self.signal_processor.generate_signal(symbol)
                report = self.order_executor.process_signal(sig)
                reports.append({**sig, **report})
                log.info(
                    f"{symbol}: regime={sig['regime']} H={sig['hurst']:.3f} "
                    f"ADX={sig['adx']:.1f} signal={sig['signal']} "
                    f"action={report.get('action','?')}"
                )
            except Exception as e:
                log.error(f"Error processing {symbol}: {e}", exc_info=True)
                reports.append({"symbol": symbol, "error": str(e)})
        return reports

    def run(self):
        """Main event loop."""
        mode = "TESTNET" if self.config.testnet else "*** MAINNET — LIVE MONEY ***"
        dry  = " [DRY RUN]" if self.config.dry_run else ""
        log.info(f"{'='*60}")
        log.info(f"TradingAgents Live Trader — {mode}{dry}")
        log.info(f"Symbols: {self.config.symbols}")
        log.info(f"Interval: {self.config.interval_min}m | Capital: {self.config.capital_usdt} USDT/symbol")
        log.info(f"{'='*60}")

        # Start metrics server
        self.metrics.start()

        # Run immediately on startup, then on each candle close
        first_run = True
        while self._running:
            if not first_run:
                wait_sec = self._seconds_to_next_candle() + 5  # +5s buffer for candle to close
                log.info(f"Next cycle in {wait_sec:.0f}s ({datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC)")
                time.sleep(wait_sec)

            if not self._running:
                break

            # Check kill switch
            if self.risk_manager.is_kill_switch_active():
                log.critical("KILL SWITCH ACTIVE — halting all trading")
                break

            # Check circuit breaker
            if self.risk_manager.is_circuit_breaker_tripped():
                log.critical("CIRCUIT BREAKER TRIPPED — halting all trading")
                break

            cycle_start = time.time()
            log.info(f"--- Cycle start {datetime.now(timezone.utc).isoformat()} ---")

            reports = self.run_cycle()

            # Update metrics
            self.metrics.update({
                "cycle_time": datetime.now(timezone.utc).isoformat(),
                "symbols": self.config.symbols,
                "reports": reports,
                "risk_state": self.risk_manager.get_state(),
            })

            cycle_elapsed = time.time() - cycle_start
            log.info(f"--- Cycle complete in {cycle_elapsed:.1f}s ---")
            first_run = False

        log.info("LiveTrader stopped cleanly.")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    config = TradingConfig()
    try:
        config.validate()
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    trader = LiveTrader(config)
    trader.run()


if __name__ == "__main__":
    main()
