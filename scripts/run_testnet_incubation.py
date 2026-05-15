"""
Bybit Testnet 14-Day Incubation Runner
=======================================
Runs the validated per-asset strategy routing on Bybit Testnet.
Generates a daily JSON report and logs all activity.

  BTC → ATR Expansion Breakout (Limit orders only)
  ETH → Donchian Momentum (Market orders)

Usage:
  python3 scripts/run_testnet_incubation.py --dry-run   # safe simulation
  python3 scripts/run_testnet_incubation.py             # live testnet orders

The script runs one full cycle per hour (aligned to candle close + 5s buffer),
persists state to data/testnet_incubation.db, and writes daily reports to
data/testnet_reports/YYYY-MM-DD.json
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

from tradingagents.research.per_asset_router import PerAssetRouter
from tradingagents.execution.bybit_connector import BybitConnector

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / 'data' / 'testnet_incubation.log', mode='a'),
    ]
)
log = logging.getLogger('testnet_incubation')

SYMBOLS        = ['BTCUSDT', 'ETHUSDT']
INTERVAL       = '60'          # 1-hour candles
LOOKBACK       = 200           # bars to fetch
MAX_POS_PCT    = 0.02          # 2% NAV per trade
DAILY_LOSS_LIM = 0.03          # 3% daily circuit breaker
METRICS_PORT   = 8765
REPORT_DIR     = ROOT / 'data' / 'testnet_reports'
DB_PATH        = ROOT / 'data' / 'testnet_incubation.db'

# ── State DB ──────────────────────────────────────────────────────────────────
class IncubationDB:
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
        """)
        self.conn.commit()

    def get_position(self, symbol: str) -> Optional[dict]:
        row = self.conn.execute(
            'SELECT * FROM positions WHERE symbol=?', (symbol,)
        ).fetchone()
        if not row: return None
        cols = ['symbol','side','qty','entry_price','stop_loss','take_profit',
                'strategy','order_type','entry_bar','opened_at']
        return dict(zip(cols, row))

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
        if not pos: return 0.0
        pnl_pct  = (exit_price / pos['entry_price'] - 1) * (1 if pos['side'] == 'LONG' else -1)
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

    def get_today_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        row = self.conn.execute(
            "SELECT SUM(pnl_usdt) FROM trades WHERE closed_at LIKE ?", (f'{today}%',)
        ).fetchone()
        return float(row[0] or 0)

    def get_all_stats(self) -> dict:
        rows = self.conn.execute('SELECT pnl_pct, pnl_usdt, strategy FROM trades').fetchall()
        if not rows:
            return {'n_trades': 0, 'win_rate': 0, 'total_pnl_usdt': 0,
                    'avg_trade_pct': 0, 'profit_factor': 0, 'by_strategy': {}}
        pnls  = [r[0] for r in rows]
        usdts = [r[1] for r in rows]
        wins  = sum(1 for p in pnls if p > 0)
        gp    = sum(u for u in usdts if u > 0)
        gl    = abs(sum(u for u in usdts if u < 0))

        by_strat = {}
        for r in rows:
            s = r[2]
            if s not in by_strat:
                by_strat[s] = {'n': 0, 'wins': 0, 'pnl': 0}
            by_strat[s]['n']    += 1
            by_strat[s]['wins'] += 1 if r[0] > 0 else 0
            by_strat[s]['pnl']  += r[1]

        return {
            'n_trades':       len(pnls),
            'win_rate':       round(wins / len(pnls) * 100, 1),
            'total_pnl_usdt': round(sum(usdts), 2),
            'avg_trade_pct':  round(sum(pnls) / len(pnls) * 100, 3),
            'profit_factor':  round(gp / gl, 2) if gl > 0 else 99.0,
            'by_strategy':    {k: {
                'n': v['n'],
                'win_rate': round(v['wins'] / v['n'] * 100, 1),
                'pnl_usdt': round(v['pnl'], 2)
            } for k, v in by_strat.items()},
        }

    def get_recent_trades(self, limit=50) -> list:
        rows = self.conn.execute(
            'SELECT * FROM trades ORDER BY closed_at DESC LIMIT ?', (limit,)
        ).fetchall()
        cols = ['id','symbol','side','qty','entry_price','exit_price',
                'pnl_pct','pnl_usdt','strategy','regime','opened_at','closed_at']
        return [dict(zip(cols, r)) for r in rows]

# ── Risk Manager ──────────────────────────────────────────────────────────────
class IncubationRisk:
    def __init__(self):
        self.kill_switch     = False
        self.circuit_breaker = False

    def check_daily_loss(self, today_pnl: float, nav: float) -> bool:
        if nav <= 0 or today_pnl >= 0: return True
        loss_pct = abs(today_pnl) / nav
        if loss_pct >= DAILY_LOSS_LIM:
            self.circuit_breaker = True
            log.warning(f'CIRCUIT BREAKER: daily loss {loss_pct:.1%} ≥ {DAILY_LOSS_LIM:.1%}')
            return False
        return True

    def size_position(self, nav: float, price: float, atr: float) -> float:
        if price <= 0 or atr <= 0: return 0.0
        risk_usdt = nav * MAX_POS_PCT
        stop_dist = 2.0 * atr
        qty_by_risk = risk_usdt / stop_dist
        qty_by_nav  = (nav * MAX_POS_PCT) / price
        return round(min(qty_by_risk, qty_by_nav), 6)

    def can_trade(self) -> bool:
        return not self.kill_switch and not self.circuit_breaker

# ── Metrics Server ────────────────────────────────────────────────────────────
_metrics_payload: dict = {}

def start_metrics_server(port: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/metrics', '/health'):
                body = json.dumps(_metrics_payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *args): pass

    server = HTTPServer(('0.0.0.0', port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    log.info(f'Metrics server: http://localhost:{port}/metrics')

# ── Daily Report ──────────────────────────────────────────────────────────────
def write_daily_report(db: IncubationDB, nav: float, day: int):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    stats = db.get_all_stats()
    trades = db.get_recent_trades(100)
    report = {
        'date':       today,
        'day':        day,
        'nav_usdt':   round(nav, 2),
        'stats':      stats,
        'trades':     trades,
        'generated':  datetime.now(timezone.utc).isoformat(),
    }
    path = REPORT_DIR / f'{today}.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    log.info(f'Daily report written → {path}')
    return report

# ── Main Incubation Loop ──────────────────────────────────────────────────────
class TestnetIncubation:
    def __init__(self, dry_run: bool):
        self.dry_run  = dry_run
        self.running  = True
        self.bar_idx  = 0
        self.nav      = 0.0
        self.signals  = {}
        self.start_dt = datetime.now(timezone.utc)
        self.last_report_date = None

        api_key    = os.environ.get('BYBIT_TESTNET_API_KEY', '')
        api_secret = os.environ.get('BYBIT_TESTNET_API_SECRET', '')
        if not api_key or not api_secret:
            raise RuntimeError('BYBIT_TESTNET_API_KEY and BYBIT_TESTNET_API_SECRET must be set')

        self.db      = IncubationDB(DB_PATH)
        self.router  = PerAssetRouter()
        self.risk    = IncubationRisk()
        self.bybit   = BybitConnector(testnet=True)
        self.bybit.set_credentials(api_key=api_key, api_secret=api_secret)

        mode = 'DRY-RUN' if dry_run else 'LIVE TESTNET ORDERS'
        log.info(f'TestnetIncubation initialised — {mode}')
        log.info(f'Symbols: {SYMBOLS} | Interval: {INTERVAL}m | MaxPos: {MAX_POS_PCT:.0%} NAV')

        signal.signal(signal.SIGINT,  lambda s, f: self._shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self._shutdown())

    def _shutdown(self):
        log.info('Shutdown signal received — stopping gracefully...')
        self.running = False

    def _get_ohlcv(self, symbol: str) -> pd.DataFrame:
        bars = self.bybit.get_klines(symbol=symbol, interval=INTERVAL, limit=LOOKBACK)
        if not bars:
            raise RuntimeError(f'No kline data returned for {symbol}')
        bars = list(reversed(bars))
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms', utc=True)
        df.set_index('ts', inplace=True)
        return df

    def _process_symbol(self, symbol: str):
        df  = self._get_ohlcv(symbol)
        sig = self.router.generate_signals(df, symbol)
        self.signals[symbol] = sig
        self.db.log_signal(sig)

        log.info(
            f'{symbol}: signal={sig["signal"]:5s} regime={sig["regime"]:11s} '
            f'strategy={sig["strategy"]} ADX={sig["adx"]:.1f} '
            f'Hurst={sig["hurst"]:.3f} RSI={sig["rsi"]:.1f} '
            f'price={sig["price"]:.2f} conviction={sig.get("conviction",0):.3f}'
        )

        pos = self.db.get_position(symbol)
        price = sig['price']

        # ── Exit logic ────────────────────────────────────────────────────────
        if pos:
            bars_held = self.bar_idx - pos['entry_bar']
            stop_hit  = (pos['side'] == 'LONG'  and price <= pos['stop_loss']) or \
                        (pos['side'] == 'SHORT' and price >= pos['stop_loss'])
            tp_hit    = (pos['side'] == 'LONG'  and price >= pos['take_profit']) or \
                        (pos['side'] == 'SHORT' and price <= pos['take_profit'])
            rev_sig   = (pos['side'] == 'LONG'  and sig['signal'] == 'SHORT') or \
                        (pos['side'] == 'SHORT' and sig['signal'] == 'LONG')
            max_hold  = bars_held >= sig['max_hold_bars']

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

        # ── Entry logic ───────────────────────────────────────────────────────
        if not pos and sig['signal'] in ('LONG', 'SHORT') and self.risk.can_trade():
            qty = self.risk.size_position(self.nav, price, sig['atr'])
            if qty <= 0:
                log.warning(f'{symbol}: Position size too small (nav={self.nav:.0f}, atr={sig["atr"]:.2f})')
                return

            log.info(
                f'{symbol}: OPENING {sig["signal"]} qty={qty} price={price:.2f} '
                f'stop={sig["stop_loss"]:.2f} tp={sig["take_profit"]:.2f} '
                f'order_type={sig["order_type"]}'
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
        day = (now - self.start_dt).days + 1
        log.info(f'=== Cycle {self.bar_idx} | Day {day}/14 | {now.strftime("%Y-%m-%d %H:%M")} UTC ===')

        # Refresh NAV
        try:
            bal = self.bybit.get_balance('USDT')
            self.nav = bal.wallet_balance if bal.wallet_balance > 0 else self.nav
            log.info(f'NAV: ${self.nav:,.2f} USDT')
        except Exception as e:
            log.error(f'Balance fetch error: {e}')

        # Circuit breaker check
        today_pnl = self.db.get_today_pnl()
        if not self.risk.check_daily_loss(today_pnl, self.nav):
            log.warning('Circuit breaker active — skipping this cycle')
            return

        # Process each symbol
        for symbol in SYMBOLS:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                log.error(f'{symbol} error: {e}', exc_info=True)

        # Persist NAV
        self.db.log_nav(self.nav, today_pnl)

        # Update metrics payload
        stats = self.db.get_all_stats()
        _metrics_payload.update({
            'timestamp':       now.isoformat(),
            'day':             day,
            'nav':             round(self.nav, 2),
            'testnet':         True,
            'dry_run':         self.dry_run,
            'kill_switch':     self.risk.kill_switch,
            'circuit_breaker': self.risk.circuit_breaker,
            'today_pnl':       round(today_pnl, 2),
            'stats':           stats,
            'signals':         self.signals,
            'positions':       {s: self.db.get_position(s) for s in SYMBOLS},
            'recent_trades':   self.db.get_recent_trades(20),
        })

        # Daily report at midnight UTC
        today_str = now.strftime('%Y-%m-%d')
        if today_str != self.last_report_date:
            write_daily_report(self.db, self.nav, day)
            self.last_report_date = today_str

    def run(self):
        start_metrics_server(METRICS_PORT)
        log.info('=' * 60)
        log.info('TradingAgents — 14-Day Bybit Testnet Incubation')
        log.info(f'Start: {self.start_dt.strftime("%Y-%m-%d %H:%M")} UTC')
        log.info(f'End:   {(self.start_dt + timedelta(days=14)).strftime("%Y-%m-%d %H:%M")} UTC')
        log.info(f'Mode:  {"DRY-RUN (no orders)" if self.dry_run else "LIVE TESTNET ORDERS"}')
        log.info('=' * 60)

        end_dt = self.start_dt + timedelta(days=14)

        while self.running and datetime.now(timezone.utc) < end_dt:
            try:
                self._run_cycle()
            except Exception as e:
                log.error(f'Cycle error: {e}', exc_info=True)

            # Sleep until next candle close (1h aligned + 5s buffer)
            now_ts    = time.time()
            interval  = 3600
            next_tick = (int(now_ts / interval) + 1) * interval + 5
            sleep_sec = max(10, next_tick - time.time())
            log.info(f'Next cycle in {sleep_sec:.0f}s')
            for _ in range(int(sleep_sec)):
                if not self.running: break
                time.sleep(1)

        if datetime.now(timezone.utc) >= end_dt:
            log.info('14-day incubation period complete.')
            final = write_daily_report(self.db, self.nav, 14)
            log.info(f'Final stats: {json.dumps(final["stats"], indent=2)}')
        else:
            log.info('Incubation stopped early.')

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TradingAgents Testnet Incubation')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate signals but do not place orders')
    parser.add_argument('--single',  action='store_true',
                        help='Run one cycle and exit (for testing)')
    args = parser.parse_args()

    incubation = TestnetIncubation(dry_run=args.dry_run)

    if args.single:
        start_metrics_server(METRICS_PORT)
        incubation._run_cycle()
        print(json.dumps(_metrics_payload, indent=2, default=str))
    else:
        incubation.run()
