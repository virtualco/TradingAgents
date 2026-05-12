"""Transaction Cost Sensitivity Sweep — Cost Realism Acceptance Gate.

Runs the backtest engine at multiple slippage/commission levels and reports
Sharpe ratio, max drawdown, and net return for each. The strategy must remain
acceptable (Sharpe > 1.0) under conservative cost assumptions (20 bps) to pass.

Usage:
    python scripts/eval_cost_sweep.py

Output:
    Table of results at 5, 10, 20, and 50 bps slippage.
    COST_SWEEP_PASS: True/False
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date, timedelta

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tradingagents.backtest.engine import BacktestEngine, BacktestConfig
from tradingagents.backtest.analytics import PerformanceAnalytics


def make_price_series(n=252, trend=0.0010, vol=0.004, seed=99) -> pd.DataFrame:
    """Generate the same deterministic uptrend used in eval_backtest_quality."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, vol, n)
    prices = 100.0 * np.cumprod(1 + returns)
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "close": prices,
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


def make_signals(price_df: pd.DataFrame, every_n: int = 3) -> pd.DataFrame:
    """Generate periodic long signals."""
    rows = []
    for i in range(0, len(price_df), every_n):
        row = price_df.iloc[i]
        rows.append({
            "signal_id": f"sig-{i}",
            "ticker": "TEST",
            "trade_date": str(row["date"]),
            "direction": "long",
            "conviction": 0.75,
            "stop_loss": None,
            "take_profit": None,
        })
    return pd.DataFrame(rows)


# Cost levels to sweep (in basis points)
COST_LEVELS_BPS = [5, 10, 20, 50]

# Acceptance threshold: strategy must have Sharpe >= 1.0 at 20 bps
ACCEPTANCE_BPS = 20
ACCEPTANCE_SHARPE = 1.0


def run_cost_sweep() -> bool:
    """Run the cost sweep and return True if acceptance gate passes."""
    price_df = make_price_series()
    signals = make_signals(price_df)

    ohlcv = price_df.rename(columns={"date": "event_time"})
    ohlcv["event_time"] = pd.to_datetime(ohlcv["event_time"])
    price_dict = {"TEST": ohlcv}

    analytics = PerformanceAnalytics(risk_free_rate=0.0)

    print("=" * 72)
    print("TRANSACTION COST SENSITIVITY SWEEP")
    print("=" * 72)
    print(f"{'Slippage (bps)':<16} {'Commission (bps)':<18} {'Sharpe':<10} "
          f"{'Max DD %':<10} {'Net Return %':<14} {'Status'}")
    print("-" * 72)

    results = []
    gate_passed = True

    for bps in COST_LEVELS_BPS:
        slippage_pct = bps / 10_000.0
        commission_pct = bps / 10_000.0  # Symmetric: slippage = commission

        config = BacktestConfig(
            initial_capital=100_000,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            max_position_pct=0.50,
            max_open_positions=3,
            max_hold_days=20,
            min_conviction=0.20,
        )

        engine = BacktestEngine(config=config)
        result = engine.run(signals=signals, price_data=price_dict)

        eq_series = pd.Series(
            [p.portfolio_value for p in result.equity_curve],
            index=pd.to_datetime([p.date for p in result.equity_curve]),
        )
        daily_returns = eq_series.pct_change().dropna()

        sharpe = analytics.sharpe_ratio(daily_returns)
        max_dd = analytics.max_drawdown(eq_series)
        net_return = (eq_series.iloc[-1] / eq_series.iloc[0] - 1) * 100

        # Check acceptance gate
        status = "PASS" if sharpe >= ACCEPTANCE_SHARPE else "FAIL"
        if bps == ACCEPTANCE_BPS and sharpe < ACCEPTANCE_SHARPE:
            gate_passed = False

        results.append({
            "bps": bps,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "net_return": net_return,
            "status": status,
        })

        print(f"{bps:<16} {bps:<18} {sharpe:<10.3f} "
              f"{max_dd*100:<10.2f} {net_return:<14.2f} {status}")

    print("-" * 72)
    print(f"\nAcceptance gate: Sharpe >= {ACCEPTANCE_SHARPE:.1f} at {ACCEPTANCE_BPS} bps")
    print(f"COST_SWEEP_PASS: {gate_passed}")
    print()

    # Gross vs Net comparison
    gross_result = results[0]  # 5 bps (closest to gross)
    conservative_result = next(r for r in results if r["bps"] == ACCEPTANCE_BPS)
    print(f"Gross-equivalent (5 bps):  Sharpe {gross_result['sharpe']:.3f}, "
          f"Return {gross_result['net_return']:.2f}%")
    print(f"Conservative ({ACCEPTANCE_BPS} bps):    Sharpe {conservative_result['sharpe']:.3f}, "
          f"Return {conservative_result['net_return']:.2f}%")
    print(f"Degradation:               Sharpe -{gross_result['sharpe'] - conservative_result['sharpe']:.3f}, "
          f"Return -{gross_result['net_return'] - conservative_result['net_return']:.2f}%")

    return gate_passed


if __name__ == "__main__":
    passed = run_cost_sweep()
    sys.exit(0 if passed else 1)
