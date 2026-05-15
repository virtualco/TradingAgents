"""
TradingAgents — Multi-Asset Live Trading Orchestrator v2
=========================================================
Extends the validated per-asset strategy routing to all 8 crypto symbols
with dynamic risk-parity weighting and regime-aware allocation.

Architecture:
  - PerAssetRouter: Bayesian-optimised signal generation per symbol
  - BybitConnector: Bybit V5 Unified API (Testnet or Mainnet)
  - Dynamic portfolio: Risk-parity weights with regime adjustment
  - SQLite state persistence (positions, trades, signals, NAV, weights)
  - JSON metrics endpoint for dashboard integration

Supported Symbols:
  BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, DOGEUSDT, BNBUSDT, XRPUSDT, LINKUSDT

Usage:
  python3 scripts/live_trader_v2.py --dry-run        # signals only, no orders
  python3 scripts/live_trader_v2.py                   # live testnet orders
  python3 scripts/live_trader_v2.py --mainnet         # LIVE MONEY (requires mainnet keys)
  python3 scripts/live_trader_v2.py --single          # one cycle then exit
  python3 scripts/live_trader_v2.py --symbols BTCUSDT ETHUSDT SOLUSDT

Configuration via environment variables:
  BYBIT_TESTNET_API_KEY / BYBIT_TESTNET_API_SECRET   Testnet credentials
  BYBIT_API_KEY / BYBIT_API_SECRET                    Mainnet credentials
  TRADING_CAPITAL_USDT    Total capital allocation (default: 10000)
  TRADING_MAX_POS_PCT     Max % of NAV per trade (default: 0.02)
  TRADING_DAILY_LOSS_LIM  Daily loss circuit breaker (default: 0.03)
  TRADING_INTERVAL        Candle interval minutes (default: 60)
  TRADING_LOOKBACK        Bars to fetch (default: 200)
  TRADING_METRICS_PORT    Metrics HTTP port (default: 8765)
  TRADING_REBALANCE_HOURS Hours between weight rebalance (default: 24)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
import threading
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tradingagents.research.per_asset_router import PerAssetRouter, ASSET_CONFIG
from tradingagents.execution.bybit_connector import BybitConnector

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv('TRADING_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / 'data' / 'live_trader_v2.log', mode='a'),
    ]
)
log = logging.getLogger('live_trader_v2')

# ── Constants ─────────────────────────────────────────────────────────────────
CRYPTO_SYMBOLS = [s for s, c in ASSET_CONFIG.items() if c.get('timeframe') != '1d']

DEFAULT_INTERVAL     = '60'
DEFAULT_LOOKBACK     = 200
DEFAULT_CAPITAL      = 10000.0
DEFAULT_MAX_POS_PCT  = 0.02
DEFAULT_DAILY_LOSS   = 0.03
DEFAULT_METRICS_PORT = 8765
DEFAULT_REBALANCE_H  = 24

DB_PATH    = ROOT / 'data' / 'live_trader_v2.db'
REPORT_DIR = ROOT / 'data' / 'live_reports'


# ── Regime-Aware Weight Engine ────────────────────────────────────────────────

class WeightEngine:
    """
    Computes dynamic portfolio weights using risk-parity + regime adjustment.
    Rebalances at configurable intervals.
    """

    MAX_WEIGHT = 0.25
    MIN_WEIGHT = 0.02
    REGIME_BOOST  = 1.5
    REGIME_DAMPEN = 0.5
    TRENDING_THRESHOLD = 0.50
    RANGING_THRESHOLD  = 0.35

    def __init__(self, symbols: list, rebalance_hours: int = 24):
        self.symbols = symbols
        self.rebalance_hours = rebalance_hours
        self.weights = {s: 1.0 / len(symbols) for s in symbols}
        self.regimes = {s: 0.5 for s in symbols}
        self.last_rebalance = None
        self.return_history = {s: [] for s in symbols}

    def record_return(self, symbol: str, ret: float):
        if symbol in self.return_history:
            self.return_history[symbol].append(ret)
            if len(self.return_history[symbol]) > 720:
                self.return_history[symbol] = self.return_history[symbol][-720:]

    def detect_regime(self, returns: list, lookback: int = 90) -> float:
        if len(returns) < lookback:
            return 0.5
        window = np.array(returns[-lookback:])
        if np.std(window) < 1e-10:
            return 0.3
        net_move = abs(np.sum(window))
        gross_move = np.sum(np.abs(window))
        efficiency = net_move / gross_move if gross_move > 0 else 0
        var1 = np.var(window)
        ret2 = window[:-1] + window[1:]
        var2 = np.var(ret2)
        vr = var2 / (2 * var1) if var1 > 1e-12 else 1.0
        vr_score = np.clip((vr - 0.5) / 1.0, 0, 1)
        sharpe_sign = 1.0 if np.mean(window) > 0 else 0.3
        score = 0.4 * efficiency + 0.4 * vr_score + 0.2 * sharpe_sign
        return float(np.clip(score, 0, 1))

    def should_rebalance(self) -> bool:
        if self.last_rebalance is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_rebalance).total_seconds()
        return elapsed >= self.rebalance_hours * 3600

    def rebalance(self) -> dict:
        vols = {}
        for sym in self.symbols:
            rets = self.return_history.get(sym, [])
            if len(rets) >= 24:
                vols[sym] = max(float(np.std(rets[-120:])), 1e-8)
            else:
                vols[sym] = 0.01
        inv_vols = {s: 1.0 / v for s, v in vols.items()}
        total_iv = sum(inv_vols.values())
        base = {s: iv / total_iv for s, iv in inv_vols.items()}
        for sym in self.symbols:
            rets = self.return_history.get(sym, [])
            self.regimes[sym] = self.detect_regime(rets)
        adjusted = {}
        for sym, w in base.items():
            r = self.regimes[sym]
            if r >= self.TRENDING_THRESHOLD:
                adjusted[sym] = w * self.REGIME_BOOST
            elif r <= self.RANGING_THRESHOLD:
                adjusted[sym] = w * self.REGIME_DAMPEN
            else:
                adjusted[sym] = w
        for sym in adjusted:
            adjusted[sym] = max(adjusted[sym], self.MIN_WEIGHT)
        total = sum(adjusted.values())
        adjusted = {s: w / total for s, w in adjusted.items()}
        for _ in range(5):
            excess = 0
            n_uncapped = 0
            for sym, w in adjusted.items():
                if w > self.MAX_WEIGHT:
                    excess += w - self.MAX_WEIGHT
                    adjusted[sym] = self.MAX_WEIGHT
                else:
                    n_uncapped += 1
            if excess > 0 and n_uncapped > 0:
                per = excess / n_uncapped
                for sym in adjusted:
                    if adjusted[sym] < self.MAX_WEIGHT:
                        adjusted[sym] += per
        total = sum(adjusted.values())
        self.weights = {s: w / total for s, w in adjusted.items()}
        self.last_rebalance = datetime.now(timezone.utc)
        log.info(f'Rebalanced weights: {json.dumps({s: f"{w:.1%}" for s, w in sorted(self.weights.items(), key=lambda x: -x[1])}, indent=2)}')
        return self.weights

    def get_capital_allocation(self, symbol: str, total_capital: float) -> float:
        return total_capital * self.weights.get(symbol, 0)


# ── State Database ────────────────────────────────────────────────────────────

class TraderDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._init()

    def _init(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT, qty REAL, entry_price REAL,
                stop_loss REAL, take_profit REAL,
                strategy TEXT, order_type TEXT,
                entry_bar INTEGER, opened_at TEXT
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, side TEXT, qty REAL,
                entry_price REAL, exit_price REAL,
                pnl_pct REAL, pnl_usdt REAL,
                strategy TEXT, regime TEXT,
                opened_at TEXT, closed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, symbol TEXT, signal TEXT,
                regime TEXT, strategy TEXT,
                adx REAL, hurst REAL, rsi REAL,
                price REAL, conviction REAL
            );
            CREATE TABLE IF NOT EXISTS nav_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, nav REAL, realized_pnl REAL
            );
            CREATE TABLE IF NOT EXISTS weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, weights TEXT, regimes TEXT
            );
        """)
        self.conn.commit()

    def get_position(self, symbol: str) -> Optional[dict]:
        row = self.conn.execute(
            'SELECT * FROM positions WHERE symbol=?', (symbol,)
        ).fetchone()
        if not row:
            return None
        cols = ['symbol', 'side', 'qty', 'entry_price', 'stop_loss', 'take_profit',
                'strategy', 'order_type', 'entry_bar', 'opened_at']
        return dict(zip(cols, row))

    def get_all_positions(self) -> dict:
        rows = self.conn.execute('SELECT * FROM positions').fetchall()
        cols = ['symbol', 'side', 'qty', 'entry_price', 'stop_loss', 'take_profit',
                'strategy', 'order_type', 'entry_bar', 'opened_at']
        return {r[0]: dict(zip(cols, r)) for r in rows}

    def open_position(self, symbol, side, qty, entry_price, stop_loss, take_profit,
                      strategy, order_type, bar_idx):
        self.conn.execute("""
            INSERT OR REPLACE INTO positions
            (symbol,side,qty,entry_price,stop_loss,take_profit,strategy,order_type,entry_bar,opened_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (symbol, side, qty, entry_price, stop_loss, take_profit,
              strategy, order_type, bar_idx, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def close_position(self, symbol, exit_price, regime) -> float:
        pos = self.get_position(symbol)
        if not pos:
            return 0.0
        pnl_pct = (exit_price / pos['entry_price'] - 1) * (1 if pos['side'] == 'LONG' else -1)
        pnl_usdt = pnl_pct * pos['qty'] * pos['entry_price']
        self.conn.execute("""
            INSERT INTO trades
            (symbol,side,qty,entry_price,exit_price,pnl_pct,pnl_usdt,strategy,regime,opened_at,closed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (symbol, pos['side'], pos['qty'], pos['entry_price'], exit_price,
              pnl_pct, pnl_usdt, pos['strategy'], regime,
              pos['opened_at'], datetime.now(timezone.utc).isoformat()))
        self.conn.execute('DELETE FROM positions WHERE symbol=?', (symbol,))
        self.conn.commit()
        return pnl_usdt

    def log_signal(self, sig: dict):
        self.conn.execute("""
            INSERT INTO signals (ts,symbol,signal,regime,strategy,adx,hurst,rsi,price,conviction)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (datetime.now(timezone.utc).isoformat(), sig['symbol'], sig['signal'],
              sig['regime'], sig['strategy'], sig['adx'], sig['hurst'],
              sig['rsi'], sig['price'], sig.get('conviction', 0)))
        self.conn.commit()

    def log_nav(self, nav: float, realized_pnl: float):
        self.conn.execute(
            'INSERT INTO nav_history (ts,nav,realized_pnl) VALUES (?,?,?)',
            (datetime.now(timezone.utc).isoformat(), nav, realized_pnl)
        )
        self.conn.commit()

    def log_weights(self, weights: dict, regimes: dict):
        self.conn.execute(
            'INSERT INTO weight_history (ts,weights,regimes) VALUES (?,?,?)',
            (datetime.now(timezone.utc).isoformat(), json.dumps(weights), json.dumps(regimes))
        )
        self.conn.commit()

    def get_today_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        row = self.conn.execute(
            "SELECT SUM(pnl_usdt) FROM trades WHERE closed_at LIKE ?", (f'{today}%',)
        ).fetchone()
        return float(row[0] or 0)

    def get_all_stats(self) -> dict:
        rows = self.conn.execute('SELECT pnl_pct, pnl_usdt, strategy, symbol FROM trades').fetchall()
        if not rows:
            return {'n_trades': 0, 'win_rate': 0, 'total_pnl_usdt': 0,
                    'avg_trade_pct': 0, 'profit_factor': 0,
                    'by_strategy': {}, 'by_symbol': {}}
        pnls = [r[0] for r in rows]
        usdts = [r[1] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        gp = sum(u for u in usdts if u > 0)
        gl = abs(sum(u for u in usdts if u < 0))
        by_strat = {}
        by_sym = {}
        for r in rows:
            for d, idx in [(by_strat, 2), (by_sym, 3)]:
                k = r[idx]
                if k not in d:
                    d[k] = {'n': 0, 'wins': 0, 'pnl': 0}
                d[k]['n'] += 1
                d[k]['wins'] += 1 if r[0] > 0 else 0
                d[k]['pnl'] += r[1]
        def fmt(d):
            return {k: {'n': v['n'], 'win_rate': round(v['wins'] / v['n'] * 100, 1) if v['n'] > 0 else 0,
                         'pnl_usdt': round(v['pnl'], 2)} for k, v in d.items()}
        return {
            'n_trades': len(pnls),
            'win_rate': round(wins / len(pnls) * 100, 1),
            'total_pnl_usdt': round(sum(usdts), 2),
            'avg_trade_pct': round(sum(pnls) / len(pnls) * 100, 3),
            'profit_factor': round(gp / gl, 2) if gl > 0 else 99.0,
            'by_strategy': fmt(by_strat),
            'by_symbol': fmt(by_sym),
        }

    def get_recent_trades(self, limit=50) -> list:
        rows = self.conn.execute(
            'SELECT * FROM trades ORDER BY closed_at DESC LIMIT ?', (limit,)
        ).fetchall()
        cols = ['id', 'symbol', 'side', 'qty', 'entry_price', 'exit_price',
                'pnl_pct', 'pnl_usdt', 'strategy', 'regime', 'opened_at', 'closed_at']
        return [dict(zip(cols, r)) for r in rows]


# ── Risk Manager ──────────────────────────────────────────────────────────────

class PortfolioRisk:
    def __init__(self, daily_loss_limit: float = 0.03, max_pos_pct: float = 0.02):
        self.daily_loss_limit = daily_loss_limit
        self.max_pos_pct = max_pos_pct
        self.kill_switch = False
        self.circuit_breaker = False

    def check_daily_loss(self, today_pnl: float, nav: float) -> bool:
        if nav <= 0 or today_pnl >= 0:
            return True
        loss_pct = abs(today_pnl) / nav
        if loss_pct >= self.daily_loss_limit:
            self.circuit_breaker = True
            log.warning(f'CIRCUIT BREAKER: daily loss {loss_pct:.1%} >= {self.daily_loss_limit:.1%}')
            return False
        return True

    def size_position(self, symbol_capital: float, price: float, atr: float) -> float:
        if price <= 0 or atr <= 0:
            return 0.0
        risk_usdt = symbol_capital * self.max_pos_pct
        stop_dist = 2.0 * atr
        qty_by_risk = risk_usdt / stop_dist
        qty_by_cap = (symbol_capital * self.max_pos_pct) / price
        qty = min(qty_by_risk, qty_by_cap)
        if qty * price < 5.0:
            qty = 5.0 / price
        return round(qty, 6)

    def can_trade(self) -> bool:
        return not self.kill_switch and not self.circuit_breaker


# ── Metrics Server ────────────────────────────────────────────────────────────

_metrics_payload: dict = {}

def start_metrics_server(port: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/metrics', '/health', '/api/metrics'):
                body = json.dumps(_metrics_payload, default=str).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/api/weights':
                body = json.dumps({
                    'weights': _metrics_payload.get('weights', {}),
                    'regimes': _metrics_payload.get('regimes', {}),
                }).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/api/positions':
                body = json.dumps(_metrics_payload.get('positions', {}), default=str).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, *args):
            pass

    server = HTTPServer(('0.0.0.0', port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f'Metrics server: http://localhost:{port}/metrics')


# ── Daily Report ──────────────────────────────────────────────────────────────

def write_daily_report(db: TraderDB, nav: float, weights: dict, regimes: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    stats = db.get_all_stats()
    trades = db.get_recent_trades(100)
    positions = db.get_all_positions()
    report = {
        'date': today,
        'nav_usdt': round(nav, 2),
        'weights': weights,
        'regimes': regimes,
        'positions': positions,
        'stats': stats,
        'trades': trades,
        'generated': datetime.now(timezone.utc).isoformat(),
    }
    path = REPORT_DIR / f'{today}.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f'Daily report written -> {path}')
    return report


# ── Main Trading Orchestrator ─────────────────────────────────────────────────

class MultiAssetTrader:
    def __init__(self, args):
        self.dry_run = args.dry_run
        self.mainnet = args.mainnet
        self.running = True
        self.bar_idx = 0
        self.nav = 0.0
        self.signals = {}
        self.start_dt = datetime.now(timezone.utc)
        self.last_report_date = None

        self.capital = float(os.getenv('TRADING_CAPITAL_USDT', str(DEFAULT_CAPITAL)))
        self.interval = os.getenv('TRADING_INTERVAL', DEFAULT_INTERVAL)
        self.lookback = int(os.getenv('TRADING_LOOKBACK', str(DEFAULT_LOOKBACK)))
        self.metrics_port = int(os.getenv('TRADING_METRICS_PORT', str(DEFAULT_METRICS_PORT)))
        self.rebalance_hours = int(os.getenv('TRADING_REBALANCE_HOURS', str(DEFAULT_REBALANCE_H)))
        max_pos_pct = float(os.getenv('TRADING_MAX_POS_PCT', str(DEFAULT_MAX_POS_PCT)))
        daily_loss = float(os.getenv('TRADING_DAILY_LOSS_LIM', str(DEFAULT_DAILY_LOSS)))

        if args.symbols:
            self.symbols = [s.upper() for s in args.symbols]
        else:
            self.symbols = CRYPTO_SYMBOLS
        log.info(f'Trading symbols: {self.symbols}')

        testnet = not self.mainnet
        if testnet:
            api_key = os.environ.get('BYBIT_TESTNET_API_KEY', '')
            api_secret = os.environ.get('BYBIT_TESTNET_API_SECRET', '')
        else:
            api_key = os.environ.get('BYBIT_API_KEY', '')
            api_secret = os.environ.get('BYBIT_API_SECRET', '')

        if not api_key or not api_secret:
            key_type = 'BYBIT_TESTNET_API_KEY/SECRET' if testnet else 'BYBIT_API_KEY/SECRET'
            raise RuntimeError(f'{key_type} must be set')

        self.db = TraderDB(DB_PATH)
        self.router = PerAssetRouter()
        self.risk = PortfolioRisk(daily_loss_limit=daily_loss, max_pos_pct=max_pos_pct)
        self.weight_engine = WeightEngine(self.symbols, self.rebalance_hours)
        self.bybit = BybitConnector(testnet=testnet)
        self.bybit.set_credentials(api_key=api_key, api_secret=api_secret)

        mode = 'DRY-RUN' if self.dry_run else ('MAINNET LIVE' if self.mainnet else 'TESTNET LIVE')
        log.info(f'MultiAssetTrader initialised — {mode}')
        log.info(f'Symbols: {self.symbols} | Interval: {self.interval}m | Capital: ${self.capital:,.0f}')
        log.info(f'Rebalance: every {self.rebalance_hours}h | MaxPos: {max_pos_pct:.0%} | DailyLoss: {daily_loss:.0%}')

        signal.signal(signal.SIGINT, lambda s, f: self._shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self._shutdown())

    def _shutdown(self):
        log.info('Shutdown signal received — stopping gracefully...')
        self.running = False

    def _get_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        bars = self.bybit.get_klines(symbol=symbol, interval=self.interval, limit=self.lookback)
        if not bars:
            log.warning(f'{symbol}: No kline data — symbol may not be listed on this exchange')
            return None
        bars = list(reversed(bars))
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms', utc=True)
        df.set_index('ts', inplace=True)
        return df

    def _process_symbol(self, symbol: str):
        df = self._get_ohlcv(symbol)
        if df is None:
            return
        sig = self.router.generate_signals(df, symbol)
        self.signals[symbol] = sig
        self.db.log_signal(sig)

        if len(df) >= 2:
            ret = (df['close'].iloc[-1] / df['close'].iloc[-2]) - 1
            self.weight_engine.record_return(symbol, ret)

        log.info(
            f'{symbol}: signal={sig["signal"]:5s} regime={sig["regime"]:11s} '
            f'strategy={sig["strategy"]} ADX={sig["adx"]:.1f} '
            f'Hurst={sig.get("hurst", 0.5):.3f} RSI={sig["rsi"]:.1f} '
            f'price={sig["price"]:.2f} conv={sig.get("conviction", 0):.3f} '
            f'weight={self.weight_engine.weights.get(symbol, 0):.1%}'
        )

        pos = self.db.get_position(symbol)
        price = sig['price']

        # ── Exit logic ────────────────────────────────────────────────────
        if pos:
            bars_held = self.bar_idx - pos['entry_bar']
            stop_hit = (pos['side'] == 'LONG' and price <= pos['stop_loss']) or \
                       (pos['side'] == 'SHORT' and price >= pos['stop_loss'])
            tp_hit = (pos['side'] == 'LONG' and price >= pos['take_profit']) or \
                     (pos['side'] == 'SHORT' and price <= pos['take_profit'])
            rev_sig = (pos['side'] == 'LONG' and sig['signal'] == 'SHORT') or \
                      (pos['side'] == 'SHORT' and sig['signal'] == 'LONG')
            max_hold = bars_held >= sig['max_hold_bars']

            if stop_hit or tp_hit or rev_sig or max_hold:
                reason = 'STOP' if stop_hit else ('TP' if tp_hit else ('REVERSE' if rev_sig else 'MAX_HOLD'))
                log.info(f'{symbol}: CLOSING {pos["side"]} — reason={reason} bars_held={bars_held}')
                if not self.dry_run:
                    close_side = 'Sell' if pos['side'] == 'LONG' else 'Buy'
                    self.bybit.place_market_order(
                        symbol=symbol, side=close_side,
                        qty=round(pos['qty'], 6), reduce_only=True
                    )
                pnl = self.db.close_position(symbol, price, sig['regime'])
                log.info(f'{symbol}: Closed — P&L ${pnl:+.2f} USDT')
                pos = None

        # ── Entry logic ───────────────────────────────────────────────────
        if not pos and sig['signal'] in ('LONG', 'SHORT') and self.risk.can_trade():
            symbol_capital = self.weight_engine.get_capital_allocation(symbol, self.nav)
            qty = self.risk.size_position(symbol_capital, price, sig['atr'])
            if qty <= 0:
                log.warning(f'{symbol}: Position size too small (capital={symbol_capital:.0f}, atr={sig["atr"]:.2f})')
                return

            log.info(
                f'{symbol}: OPENING {sig["signal"]} qty={qty} price={price:.2f} '
                f'stop={sig["stop_loss"]:.2f} tp={sig["take_profit"]:.2f} '
                f'order_type={sig["order_type"]} capital_alloc=${symbol_capital:.0f}'
            )

            if not self.dry_run:
                order_side = 'Buy' if sig['signal'] == 'LONG' else 'Sell'
                if sig['order_type'] == 'Limit':
                    limit_px = price * 0.9999 if sig['signal'] == 'LONG' else price * 1.0001
                    result = self.bybit.place_limit_order(
                        symbol=symbol, side=order_side,
                        qty=round(qty, 6), price=round(limit_px, 2),
                        time_in_force='GTC'
                    )
                else:
                    result = self.bybit.place_market_order(
                        symbol=symbol, side=order_side, qty=round(qty, 6)
                    )
                if not result.success:
                    log.error(f'{symbol}: Order failed — {result.error}')
                    return
                log.info(f'{symbol}: Order placed — orderId={result.order_id}')

            self.db.open_position(
                symbol, sig['signal'], qty, price,
                sig['stop_loss'], sig['take_profit'],
                sig['strategy'], sig['order_type'], self.bar_idx
            )

    def _run_cycle(self):
        self.bar_idx += 1
        now = datetime.now(timezone.utc)
        log.info(f'=== Cycle {self.bar_idx} | {now.strftime("%Y-%m-%d %H:%M")} UTC ===')

        try:
            bal = self.bybit.get_balance('USDT')
            if bal.wallet_balance > 0:
                self.nav = bal.wallet_balance
            else:
                self.nav = max(self.nav, self.capital)
            log.info(f'NAV: ${self.nav:,.2f} USDT')
        except Exception as e:
            log.error(f'Balance fetch error: {e}')
            if self.nav <= 0:
                self.nav = self.capital

        today_pnl = self.db.get_today_pnl()
        if not self.risk.check_daily_loss(today_pnl, self.nav):
            log.warning('Circuit breaker active — skipping this cycle')
            return

        if self.weight_engine.should_rebalance():
            weights = self.weight_engine.rebalance()
            self.db.log_weights(weights, self.weight_engine.regimes)

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                log.error(f'{symbol} error: {e}', exc_info=True)

        self.db.log_nav(self.nav, today_pnl)

        stats = self.db.get_all_stats()
        global _metrics_payload
        _metrics_payload = {
            'timestamp': now.isoformat(),
            'bar_idx': self.bar_idx,
            'nav': round(self.nav, 2),
            'testnet': not self.mainnet,
            'dry_run': self.dry_run,
            'kill_switch': self.risk.kill_switch,
            'circuit_breaker': self.risk.circuit_breaker,
            'today_pnl': round(today_pnl, 2),
            'symbols': self.symbols,
            'weights': {s: round(w, 4) for s, w in self.weight_engine.weights.items()},
            'regimes': {s: round(r, 3) for s, r in self.weight_engine.regimes.items()},
            'stats': stats,
            'signals': {s: {k: v for k, v in sig.items()} for s, sig in self.signals.items()},
            'positions': {s: self.db.get_position(s) for s in self.symbols},
            'recent_trades': self.db.get_recent_trades(20),
        }

        today_str = now.strftime('%Y-%m-%d')
        if today_str != self.last_report_date:
            write_daily_report(self.db, self.nav, self.weight_engine.weights, self.weight_engine.regimes)
            self.last_report_date = today_str

    def run(self):
        start_metrics_server(self.metrics_port)
        mode = 'DRY-RUN' if self.dry_run else ('MAINNET' if self.mainnet else 'TESTNET')
        log.info('=' * 70)
        log.info(f'TradingAgents Multi-Asset Live Trader v2 — {mode}')
        log.info(f'Symbols: {self.symbols}')
        log.info(f'Capital: ${self.capital:,.0f} | Interval: {self.interval}m')
        log.info(f'Rebalance: {self.rebalance_hours}h | Metrics: :{self.metrics_port}')
        log.info('=' * 70)

        while self.running:
            try:
                self._run_cycle()
            except Exception as e:
                log.error(f'Cycle error: {e}', exc_info=True)
            now_ts = time.time()
            interval_sec = int(self.interval) * 60
            next_tick = (int(now_ts / interval_sec) + 1) * interval_sec + 5
            sleep_sec = max(10, next_tick - time.time())
            log.info(f'Next cycle in {sleep_sec:.0f}s')
            for _ in range(int(sleep_sec)):
                if not self.running:
                    break
                time.sleep(1)
        log.info('MultiAssetTrader stopped cleanly.')


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TradingAgents Multi-Asset Live Trader v2')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate signals but do not place orders')
    parser.add_argument('--mainnet', action='store_true',
                        help='Use mainnet credentials (LIVE MONEY)')
    parser.add_argument('--single', action='store_true',
                        help='Run one cycle and exit (for testing)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Override symbol list (e.g., --symbols BTCUSDT ETHUSDT SOLUSDT)')
    args = parser.parse_args()

    trader = MultiAssetTrader(args)

    if args.single:
        start_metrics_server(trader.metrics_port)
        trader._run_cycle()
        print(json.dumps(_metrics_payload, indent=2, default=str))
    else:
        trader.run()


if __name__ == '__main__':
    main()
