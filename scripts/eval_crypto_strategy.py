#!/usr/bin/env python3
"""
Crypto Day-Trading Strategy Evaluation Harness
===============================================
Fetches real intraday OHLCV data for BTC-USD and ETH-USD,
runs a full vectorised backtest with transaction costs and stop-losses,
and outputs a composite score for the autoresearch ratchet loop.

Primary metric: Composite Score = (weekly_return_pct * 0.4) + (sharpe * 10 * 0.4) + (win_rate * 100 * 0.2)
Output line: AUTORESEARCH_METRIC: <value>

Exit code 0 = success, 1 = data fetch failure, 2 = strategy crash.
"""
from __future__ import annotations
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_crypto")

TICKERS = ["BTC-USD", "ETH-USD"]
INTERVAL = "1h"
PERIOD = "60d"
TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100_000.0
MAX_POSITION_PCT = 0.20
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.05


def fetch_ohlcv(ticker: str, interval: str = "1h", period: str = "60d") -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(1, 4):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"Empty data for {ticker}")
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            logger.info(f"  {ticker}: {len(df)} candles fetched (attempt {attempt})")
            return df
        except Exception as e:
            logger.warning(f"  {ticker} fetch attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def load_strategy():
    # Force reload to pick up latest changes
    import importlib
    import tradingagents.research.crypto_strategy as mod
    importlib.reload(mod)
    return mod.CryptoDayTradingStrategy()


def run_backtest(
    df: pd.DataFrame,
    strategy,
    initial_capital: float = INITIAL_CAPITAL,
    max_pos_pct: float = MAX_POSITION_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    transaction_cost: float = TRANSACTION_COST,
) -> Dict:
    if df.empty or len(df) < 50:
        return {"error": "insufficient data"}

    try:
        signals = strategy.generate_signals(df)
    except Exception as e:
        return {"error": f"signal generation failed: {e}"}

    if not isinstance(signals, pd.Series):
        return {"error": "generate_signals must return pd.Series"}

    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trades: List[Dict] = []
    equity_curve: List[float] = [capital]
    daily_returns: List[float] = []

    prev_day = None
    day_open_equity = capital

    for i in range(1, len(df)):
        price = float(df["close"].iloc[i])
        if price <= 0 or np.isnan(price):
            equity_curve.append(capital + (position * price if position > 0 else 0))
            continue

        sig = int(signals.iloc[i - 1]) if i - 1 < len(signals) else 0
        current_day = str(df.index[i])[:10]

        if prev_day is not None and current_day != prev_day:
            mtm = capital + (position * price if position > 0 else 0)
            day_return = (mtm - day_open_equity) / day_open_equity if day_open_equity > 0 else 0.0
            daily_returns.append(day_return)
            day_open_equity = mtm
        prev_day = current_day

        # Stop-loss / take-profit
        if position != 0.0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price * (1 if position > 0 else -1)
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                if position > 0:
                    proceeds = position * price * (1 - transaction_cost)
                    capital += proceeds
                    trades.append({"type": "close_sl_tp", "price": price,
                                   "pnl_pct": pnl_pct, "win": pnl_pct > 0})
                else:
                    cost = abs(position) * price * (1 + transaction_cost)
                    pnl = abs(position) * (entry_price - price)
                    capital += pnl - cost * transaction_cost
                    trades.append({"type": "close_sl_tp", "price": price,
                                   "pnl_pct": pnl_pct, "win": pnl_pct > 0})
                position = 0.0
                entry_price = 0.0

        # Signal execution
        if sig == 1 and position <= 0:
            if position < 0:
                cost = abs(position) * price * (1 + transaction_cost)
                capital -= abs(position) * (price - entry_price)
                capital -= abs(position) * price * transaction_cost
                trades.append({"type": "close_short", "price": price, "win": False})
                position = 0.0

            position_size = (capital * max_pos_pct) / price
            cost = position_size * price * (1 + transaction_cost)
            if cost <= capital and position_size > 0:
                capital -= position_size * price * transaction_cost
                position = position_size
                entry_price = price
                trades.append({"type": "open_long", "price": price, "win": False})

        elif sig == -1 and position >= 0:
            if position > 0:
                capital += position * price * (1 - transaction_cost)
                trades.append({"type": "close_long", "price": price, "win": False})
                position = 0.0

            position_size = (capital * max_pos_pct) / price
            if position_size > 0:
                position = -position_size
                entry_price = price
                trades.append({"type": "open_short", "price": price, "win": False})

        elif sig == 0 and position != 0:
            # Close on flat signal
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                capital += position * price * (1 - transaction_cost)
                trades.append({"type": "close_long_flat", "price": price,
                               "pnl_pct": pnl_pct, "win": pnl_pct > 0})
            else:
                pnl_pct = (entry_price - price) / entry_price
                capital += abs(position) * (entry_price - price) - abs(position) * price * transaction_cost
                trades.append({"type": "close_short_flat", "price": price,
                               "pnl_pct": pnl_pct, "win": pnl_pct > 0})
            position = 0.0
            entry_price = 0.0

        mtm = capital + (position * price if position > 0 else
                         abs(position) * (2 * entry_price - price) if position < 0 else 0)
        equity_curve.append(max(mtm, 0))

    # Close open position at end
    if position != 0.0 and entry_price > 0:
        last_price = float(df["close"].iloc[-1])
        if position > 0:
            capital += position * last_price * (1 - transaction_cost)
        else:
            capital += abs(position) * (entry_price - last_price) - abs(position) * last_price * transaction_cost
        position = 0.0

    total_return = (capital - initial_capital) / initial_capital
    n_days = max(len(daily_returns), 1)
    n_weeks = n_days / 5.0
    weekly_return = (1 + total_return) ** (1 / max(n_weeks, 1)) - 1

    dr = np.array(daily_returns) if daily_returns else np.array([0.0])
    sharpe = (dr.mean() / (dr.std() + 1e-9)) * np.sqrt(252) if len(dr) > 1 else 0.0

    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / (peak + 1e-9)
    max_dd = abs(drawdown.min())

    closed_trades = [t for t in trades if t.get("type", "").startswith("close")]
    win_rate = sum(1 for t in closed_trades if t.get("win", False)) / max(len(closed_trades), 1)
    n_trades = len([t for t in trades if t.get("type", "").startswith("open")])

    return {
        "total_return_pct": total_return * 100,
        "weekly_return_pct": weekly_return * 100,
        "annualized_return_pct": ((1 + total_return) ** (252 / max(n_days, 1)) - 1) * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "n_days": n_days,
        "final_capital": capital,
    }


def composite_score(metrics: Dict) -> float:
    weekly = min(max(metrics.get("weekly_return_pct", 0) / 30.0, 0), 1.0)
    sharpe = min(max(metrics.get("sharpe", 0) / 3.0, 0), 1.0)
    winrate = max(metrics.get("win_rate", 0), 0)
    penalty = min(metrics.get("max_drawdown_pct", 0) / 50.0, 1.0) * 0.3
    score = (weekly * 0.40 + sharpe * 0.40 + winrate * 0.20 - penalty) * 100
    return max(score, 0.0)


def main() -> int:
    logger.info("=" * 60)
    logger.info("Crypto Day-Trading Strategy Evaluation")
    logger.info(f"  Tickers  : {', '.join(TICKERS)}")
    logger.info(f"  Interval : {INTERVAL} | Period: {PERIOD}")
    logger.info("=" * 60)

    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        df = fetch_ohlcv(ticker, INTERVAL, PERIOD)
        if df.empty:
            logger.error(f"Failed to fetch data for {ticker}")
            print("AUTORESEARCH_CRASH: price fetch failure")
            return 1
        all_data[ticker] = df

    try:
        strategy = load_strategy()
        logger.info(f"Strategy loaded: {strategy.__class__.__name__}")
    except Exception as e:
        logger.error(f"Strategy load failed: {e}")
        import traceback; traceback.print_exc()
        print(f"AUTORESEARCH_CRASH: strategy load error — {e}")
        return 2

    all_metrics: List[Dict] = []
    for ticker, df in all_data.items():
        logger.info(f"\nRunning backtest: {ticker}")
        try:
            metrics = run_backtest(df, strategy)
            if "error" in metrics:
                logger.warning(f"  {ticker}: {metrics['error']}")
                continue
            all_metrics.append(metrics)
            logger.info(f"  Weekly Return  : {metrics['weekly_return_pct']:+.2f}%")
            logger.info(f"  Annual Return  : {metrics['annualized_return_pct']:+.2f}%")
            logger.info(f"  Sharpe Ratio   : {metrics['sharpe']:.3f}")
            logger.info(f"  Max Drawdown   : {metrics['max_drawdown_pct']:.2f}%")
            logger.info(f"  Win Rate       : {metrics['win_rate']:.1%}")
            logger.info(f"  Trades         : {metrics['n_trades']}")
        except Exception as e:
            logger.error(f"  {ticker} backtest crashed: {e}")
            import traceback; traceback.print_exc()

    if not all_metrics:
        print("AUTORESEARCH_CRASH: all backtests failed")
        return 2

    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
                   for k in all_metrics[0].keys()}
    score = composite_score(avg_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Avg Weekly Return  : {avg_metrics['weekly_return_pct']:+.2f}%  (target: 30%)")
    logger.info(f"  Avg Annual Return  : {avg_metrics['annualized_return_pct']:+.2f}%")
    logger.info(f"  Avg Sharpe         : {avg_metrics['sharpe']:.3f}")
    logger.info(f"  Avg Max Drawdown   : {avg_metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"  Avg Win Rate       : {avg_metrics['win_rate']:.1%}")
    logger.info(f"  Composite Score    : {score:.2f}/100")
    logger.info("=" * 60)

    print(f"AUTORESEARCH_METRIC: {score:.4f}")
    print(f"SCORE: {score:.4f}")

    results_path = REPO_ROOT / "data" / "crypto_eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_path, "w") as f:
        json.dump({"score": score, "per_ticker": {TICKERS[i]: all_metrics[i]
                   for i in range(len(all_metrics))}, "aggregate": avg_metrics}, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
