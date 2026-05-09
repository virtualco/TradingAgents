"""
Out-of-Sample (OOS) Validation & Walk-Forward Analysis
=======================================================
Tests CryptoDayTradingStrategy v2 across:
  - Bear market:    2022-01-01 → 2022-12-31
  - Recovery:       2023-01-01 → 2023-12-31
  - Bull market:    2024-01-01 → 2024-12-31
  - Pre-Live:       2025-01-01 → 2026-01-01

Walk-Forward Analysis:
  - 90-day training window, 30-day test window
  - Rolling across full 2022-2026 period
  - Reports out-of-sample Sharpe, return, drawdown per fold

Usage:
    python3 scripts/validate_oos.py
"""
from __future__ import annotations
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] oos_validate: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("oos_validate")

# ── Constants ────────────────────────────────────────────────────────────────
TICKERS = [
    ("BTC-USD", "data/historical/BTC_USD_1h_2022-01-01_2026-01-01.parquet"),
    ("ETH-USD", "data/historical/ETH_USD_1h_2022-01-01_2026-01-01.parquet"),
]
TRANSACTION_COST = 0.001   # 0.1% per side (Bybit maker fee)
STRESSED_COST    = 0.0025  # 0.25% stressed scenario
INITIAL_CAPITAL  = 100_000.0

REGIMES = [
    ("Bear Market",   "2022-01-01", "2022-12-31"),
    ("Recovery",      "2023-01-01", "2023-12-31"),
    ("Bull Market",   "2024-01-01", "2024-12-31"),
    ("Pre-Live",      "2025-01-01", "2026-01-01"),
]


# ── Strategy Import ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from tradingagents.research.crypto_strategy import CryptoDayTradingStrategy
    log.info("Strategy loaded: CryptoDayTradingStrategy v2")
except ImportError as e:
    log.error(f"Cannot import strategy: {e}")
    sys.exit(1)


# ── Backtest Engine ───────────────────────────────────────────────────────────
@dataclass
class BacktestResult:
    ticker: str
    period_label: str
    start: str
    end: str
    n_candles: int
    n_trades: int
    total_return_pct: float
    weekly_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    transaction_cost_pct: float


def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    period_label: str,
    start: str,
    end: str,
    tx_cost: float = TRANSACTION_COST,
) -> Optional[BacktestResult]:
    """Run a vectorised backtest with full metrics."""
    if df is None or len(df) < 100:
        log.warning(f"  Insufficient data for {ticker} [{period_label}]: {len(df) if df is not None else 0} rows")
        return None

    strategy = CryptoDayTradingStrategy()
    signals = strategy.generate_signals(df)

    close = df["close"].copy()
    returns = close.pct_change().fillna(0)

    position_changes = signals.diff().abs().fillna(0)
    cost_series = position_changes * tx_cost
    strat_returns = (signals.shift(1).fillna(0) * returns) - cost_series

    # Equity curve
    equity = INITIAL_CAPITAL * (1 + strat_returns).cumprod()
    equity.iloc[0] = INITIAL_CAPITAL

    # Metrics
    total_return = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n_weeks = max(len(df) / (24 * 7), 1)
    weekly_return = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_weeks) - 1) * 100

    # Sharpe (annualised, hourly bars)
    hourly_mean = strat_returns.mean()
    hourly_std  = strat_returns.std()
    sharpe = (hourly_mean / hourly_std * np.sqrt(24 * 365)) if hourly_std > 1e-10 else 0.0

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = abs(drawdown.min()) * 100

    # Trade analysis
    trade_rets = []
    pos = 0
    entry_price = 0.0
    for i in range(len(df)):
        sig = int(signals.iloc[i])
        price = float(close.iloc[i])
        if pos == 0 and sig != 0:
            pos = sig
            entry_price = price
        elif pos != 0 and (sig == 0 or sig != pos):
            trade_ret = (price / entry_price - 1) * pos
            trade_rets.append(trade_ret)
            pos = 0
            if sig != 0:
                pos = sig
                entry_price = price

    n_trades = len(trade_rets)
    wins   = [r for r in trade_rets if r > 0]
    losses = [r for r in trade_rets if r <= 0]
    win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 1e-10
    profit_factor = (avg_win / avg_loss) if avg_loss > 1e-10 else 0.0

    return BacktestResult(
        ticker=ticker,
        period_label=period_label,
        start=start,
        end=end,
        n_candles=len(df),
        n_trades=n_trades,
        total_return_pct=round(float(total_return), 2),
        weekly_return_pct=round(float(weekly_return), 2),
        sharpe_ratio=round(float(sharpe), 3),
        max_drawdown_pct=round(float(max_dd), 2),
        win_rate_pct=round(float(win_rate), 1),
        profit_factor=round(float(profit_factor), 2),
        transaction_cost_pct=tx_cost * 100,
    )


# ── Walk-Forward Analysis ─────────────────────────────────────────────────────
def walk_forward_analysis(
    df: pd.DataFrame,
    ticker: str,
    train_days: int = 90,
    test_days: int = 30,
    tx_cost: float = TRANSACTION_COST,
) -> List[BacktestResult]:
    """Rolling walk-forward: train 90d, test 30d, step 30d."""
    train_bars = train_days * 24
    test_bars  = test_days  * 24
    results = []
    fold = 1

    i = 0
    while i + train_bars + test_bars <= len(df):
        test_df = df.iloc[i + train_bars: i + train_bars + test_bars].copy()
        test_start = str(test_df.index[0].date())
        test_end   = str(test_df.index[-1].date())

        r = run_backtest(test_df, ticker, f"WFA Fold {fold}", test_start, test_end, tx_cost)
        if r:
            results.append(r)
            log.info(
                f"  WFA {ticker} Fold {fold:>2} [{test_start}→{test_end}]: "
                f"weekly={r.weekly_return_pct:+.1f}% | Sharpe={r.sharpe_ratio:.2f} | "
                f"DD={r.max_drawdown_pct:.1f}% | trades={r.n_trades}"
            )
        i += test_bars
        fold += 1

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("OUT-OF-SAMPLE VALIDATION — CryptoDayTradingStrategy v2")
    log.info("=" * 65)

    # Load parquet data
    data_cache = {}
    for label, parquet_path in TICKERS:
        p = Path(parquet_path)
        if not p.exists():
            log.error(f"Data file not found: {p}")
            sys.exit(1)
        df = pd.read_parquet(p)
        # Ensure timezone-naive index for slicing
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        data_cache[label] = df
        log.info(f"Loaded {label}: {len(df)} rows [{df.index[0].date()} → {df.index[-1].date()}]")

    all_results: List[BacktestResult] = []

    # ── Stage 1: Regime Backtests ─────────────────────────────────────────────
    log.info("\n[STAGE 1] Regime-Specific Backtests (2022–2026)")
    log.info("-" * 65)

    for regime_label, start, end in REGIMES:
        log.info(f"\n  Regime: {regime_label} [{start} → {end}]")
        for ticker, _ in TICKERS:
            df_full = data_cache[ticker]
            df = df_full.loc[start:end].copy()
            if len(df) < 100:
                log.warning(f"    {ticker}: insufficient rows ({len(df)}) for {regime_label}")
                continue

            # Normal cost
            r = run_backtest(df, ticker, regime_label, start, end, TRANSACTION_COST)
            if r:
                all_results.append(r)
                log.info(
                    f"    {ticker:10s} | weekly={r.weekly_return_pct:+7.2f}% | "
                    f"Sharpe={r.sharpe_ratio:6.3f} | DD={r.max_drawdown_pct:5.1f}% | "
                    f"WR={r.win_rate_pct:.0f}% | trades={r.n_trades}"
                )

            # Stressed cost
            rs = run_backtest(df, ticker, f"{regime_label} [STRESSED]", start, end, STRESSED_COST)
            if rs:
                all_results.append(rs)
                log.info(
                    f"    {ticker:10s} | weekly={rs.weekly_return_pct:+7.2f}% | "
                    f"Sharpe={rs.sharpe_ratio:6.3f} | DD={rs.max_drawdown_pct:5.1f}% | "
                    f"WR={rs.win_rate_pct:.0f}% | trades={rs.n_trades} [STRESSED 0.25%]"
                )

    # ── Stage 2: Walk-Forward Analysis ───────────────────────────────────────
    log.info("\n[STAGE 2] Walk-Forward Analysis (2022-01-01 → 2026-01-01)")
    log.info("-" * 65)

    wfa_results: List[BacktestResult] = []
    for ticker, _ in TICKERS:
        log.info(f"\n  WFA for {ticker}:")
        df_full = data_cache[ticker]
        folds = walk_forward_analysis(df_full, ticker)
        wfa_results.extend(folds)

    # ── Stage 3: Summary ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 65)

    normal_results   = [r for r in all_results if "STRESSED" not in r.period_label]
    stressed_results = [r for r in all_results if "STRESSED" in r.period_label]

    avg_weekly_s = 0.0
    avg_sharpe_s = 0.0

    if normal_results:
        avg_weekly = float(np.mean([r.weekly_return_pct for r in normal_results]))
        avg_sharpe = float(np.mean([r.sharpe_ratio for r in normal_results]))
        avg_dd     = float(np.mean([r.max_drawdown_pct for r in normal_results]))
        avg_wr     = float(np.mean([r.win_rate_pct for r in normal_results]))
        positive_regimes = sum(1 for r in normal_results if r.weekly_return_pct > 0)

        log.info(f"\n  Regime Backtests (Normal Cost 0.1%):")
        log.info(f"    Avg Weekly Return : {avg_weekly:+.2f}%")
        log.info(f"    Avg Sharpe Ratio  : {avg_sharpe:.3f}")
        log.info(f"    Avg Max Drawdown  : {avg_dd:.1f}%")
        log.info(f"    Avg Win Rate      : {avg_wr:.1f}%")
        log.info(f"    Positive Regimes  : {positive_regimes}/{len(normal_results)}")

    if stressed_results:
        avg_weekly_s = float(np.mean([r.weekly_return_pct for r in stressed_results]))
        avg_sharpe_s = float(np.mean([r.sharpe_ratio for r in stressed_results]))
        log.info(f"\n  Stressed Cost Scenario (0.25%):")
        log.info(f"    Avg Weekly Return : {avg_weekly_s:+.2f}%")
        log.info(f"    Avg Sharpe Ratio  : {avg_sharpe_s:.3f}")

    wfa_weekly = 0.0
    wfa_sharpe = 0.0
    wfa_dd = 0.0
    wfa_positive = 0

    if wfa_results:
        wfa_weekly  = float(np.mean([r.weekly_return_pct for r in wfa_results]))
        wfa_sharpe  = float(np.mean([r.sharpe_ratio for r in wfa_results]))
        wfa_dd      = float(np.mean([r.max_drawdown_pct for r in wfa_results]))
        wfa_positive = sum(1 for r in wfa_results if r.weekly_return_pct > 0)
        log.info(f"\n  Walk-Forward Analysis ({len(wfa_results)} folds):")
        log.info(f"    Avg OOS Weekly Return : {wfa_weekly:+.2f}%")
        log.info(f"    Avg OOS Sharpe Ratio  : {wfa_sharpe:.3f}")
        log.info(f"    Avg OOS Max Drawdown  : {wfa_dd:.1f}%")
        log.info(f"    Profitable Folds      : {wfa_positive}/{len(wfa_results)} ({wfa_positive/len(wfa_results)*100:.0f}%)")

    # ── Readiness Gate ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("READINESS GATE ASSESSMENT")
    log.info("=" * 65)

    gates = {}
    if wfa_results:
        gates["OOS Sharpe > 1.0"]       = bool(wfa_sharpe > 1.0)
        gates["OOS Weekly Return > 0%"] = bool(wfa_weekly > 0)
        gates["OOS Max DD < 30%"]       = bool(wfa_dd < 30)
        gates["Profitable Folds > 55%"] = bool((wfa_positive / len(wfa_results)) > 0.55)
    if stressed_results:
        gates["Profitable Under Stress"] = bool(avg_weekly_s > 0)

    all_pass = all(gates.values()) if gates else False
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        log.info(f"  [{status}] {gate}")

    log.info("\n" + (">>> STRATEGY CLEARED FOR BYBIT TESTNET <<<" if all_pass
                     else ">>> STRATEGY NEEDS FURTHER OPTIMISATION <<<"))

    # ── Save Results ──────────────────────────────────────────────────────────
    output = {
        "run_date": datetime.utcnow().isoformat(),
        "strategy": "CryptoDayTradingStrategy v2",
        "regime_results": [asdict(r) for r in all_results],
        "wfa_results": [asdict(r) for r in wfa_results],
        "gates": gates,
        "cleared_for_testnet": all_pass,
        "summary": {
            "avg_weekly_return_pct": round(avg_weekly if normal_results else 0.0, 2),
            "avg_sharpe": round(avg_sharpe if normal_results else 0.0, 3),
            "avg_drawdown_pct": round(avg_dd if normal_results else 0.0, 2),
            "wfa_avg_weekly_pct": round(wfa_weekly, 2),
            "wfa_avg_sharpe": round(wfa_sharpe, 3),
            "wfa_profitable_folds_pct": round(wfa_positive / len(wfa_results) * 100, 1) if wfa_results else 0.0,
        }
    }
    out_path = Path("data/oos_validation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    log.info(f"\nResults saved to {out_path}")

    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
