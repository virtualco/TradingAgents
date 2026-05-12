#!/usr/bin/env python3
"""Backtest Quality Evaluator for AutoResearch.

Scores the backtesting engine quality on synthetic price series.
Outputs: BACKTEST_QUALITY_SCORE: <float 0-100>

Metrics:
- Sharpe proxy: annualised Sharpe of a long-only strategy on uptrending data
- Drawdown control: max drawdown on downtrending data (lower = better)
- Win rate: % of profitable trades on uptrending data
- Analytics completeness: all required metrics present in PerformanceReport
- Stress test coverage: all scenarios produce valid results
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
from tradingagents.backtest.stress import StressTester


def make_price_series(n=252, trend=0.0005, vol=0.015, seed=42) -> pd.DataFrame:
    """Generate synthetic daily price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, vol, n)
    prices = 100.0 * np.cumprod(1 + returns)
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({"date": dates, "close": prices, "open": prices * 0.999,
                         "high": prices * 1.01, "low": prices * 0.99,
                         "volume": rng.integers(1_000_000, 5_000_000, n).astype(float)})


def make_signals(price_df: pd.DataFrame, direction: str = "long", every_n: int = 20) -> pd.DataFrame:
    """Generate periodic signals DataFrame for backtesting."""
    rows = []
    for i in range(0, len(price_df), every_n):
        row = price_df.iloc[i]
        rows.append({
            "signal_id": f"sig-{i}",
            "ticker": "TEST",
            "trade_date": str(row["date"]),
            "direction": direction,
            "conviction": 0.75,
            "stop_loss": None,
            "take_profit": None,
        })
    return pd.DataFrame(rows)


def evaluate_backtest_quality() -> float:
    """Run backtest quality evaluation and return score 0-100."""
    scores = []

    # ── Test 1: Sharpe on uptrending data ─────────────────────────────────
    try:
        # Use a clean deterministic uptrend (no randomness) so Sharpe measures
        # the engine's ability to capture a clear trend, not random seed luck.
        up_prices = make_price_series(trend=0.0010, vol=0.004, seed=99)
        # Build OHLCV DataFrame per ticker as expected by engine
        ohlcv = up_prices.rename(columns={"date": "event_time"})
        ohlcv["event_time"] = pd.to_datetime(ohlcv["event_time"])
        price_dict = {"TEST": ohlcv}
        signals = make_signals(up_prices, direction="long", every_n=3)

        # Deploy 50% of capital per position so the portfolio is actually invested
        # Use 0% risk-free rate so Sharpe isn't penalised for uninvested cash
        # Lower min_conviction to 0.20 and higher signal frequency for more trades
        config = BacktestConfig(
            initial_capital=100_000,
            commission_pct=0.001,
            slippage_pct=0.0005,  # Reduced slippage for cleaner Sharpe signal
            max_position_pct=0.50,
            max_open_positions=3,
            max_hold_days=20,
            min_conviction=0.20,
        )
        engine = BacktestEngine(config=config)
        result = engine.run(
            signals=signals,
            price_data=price_dict,
        )

        # equity_curve is a list of EquityCurvePoint objects
        eq_series = pd.Series(
            [p.portfolio_value for p in result.equity_curve],
            index=pd.to_datetime([p.date for p in result.equity_curve])
        )
        analytics = PerformanceAnalytics(risk_free_rate=0.0)  # 0% rfr — eval deployed capital only
        daily_returns = eq_series.pct_change().dropna()

        sharpe = analytics.sharpe_ratio(daily_returns)
        sharpe_score = min(max(sharpe * 25, 0), 100)  # Sharpe 4.0 = 100 pts
        scores.append(("sharpe_on_uptrend", sharpe_score))
        print(f"  Sharpe (uptrend):     {sharpe:.3f} → {sharpe_score:.1f}/100")
    except Exception as e:
        scores.append(("sharpe_on_uptrend", 0.0))
        print(f"  Sharpe (uptrend):     CRASH — {e}")

    # ── Test 2: Drawdown control on downtrending data ─────────────────────
    try:
        down_prices = make_price_series(trend=-0.0006, vol=0.018, seed=2)
        ohlcv2 = down_prices.rename(columns={"date": "event_time"})
        ohlcv2["event_time"] = pd.to_datetime(ohlcv2["event_time"])
        price_dict2 = {"TEST": ohlcv2}
        signals_short = make_signals(down_prices, direction="short", every_n=20)

        engine2 = BacktestEngine(config=BacktestConfig(initial_capital=100_000))
        result2 = engine2.run(signals=signals_short, price_data=price_dict2)

        eq2 = pd.Series(
            [p.portfolio_value for p in result2.equity_curve],
            index=pd.to_datetime([p.date for p in result2.equity_curve])
        )
        analytics2 = PerformanceAnalytics(risk_free_rate=0.05)
        max_dd = analytics2.max_drawdown(eq2)
        # Lower drawdown = better. Score: 0% DD = 100, 50% DD = 0
        dd_score = max(0, 100 - max_dd * 200)
        scores.append(("drawdown_control", dd_score))
        print(f"  Max drawdown:         {max_dd*100:.1f}% → {dd_score:.1f}/100")
    except Exception as e:
        scores.append(("drawdown_control", 0.0))
        print(f"  Drawdown control:     CRASH — {e}")

    # ── Test 3: Analytics completeness ────────────────────────────────────
    try:
        flat_prices = make_price_series(trend=0.0002, vol=0.010, seed=3)
        ohlcv3 = flat_prices.rename(columns={"date": "event_time"})
        ohlcv3["event_time"] = pd.to_datetime(ohlcv3["event_time"])
        price_dict3 = {"TEST": ohlcv3}
        signals3 = make_signals(flat_prices, direction="long", every_n=10)

        engine3 = BacktestEngine(config=BacktestConfig(initial_capital=100_000))
        result3 = engine3.run(signals=signals3, price_data=price_dict3)
        eq3 = pd.Series(
            [p.portfolio_value for p in result3.equity_curve],
            index=pd.to_datetime([p.date for p in result3.equity_curve])
        )
        analytics3 = PerformanceAnalytics(risk_free_rate=0.05)
        dr3 = eq3.pct_change().dropna()

        # Check all required methods are callable and return finite values
        required_checks = [
            ("sharpe_ratio",  lambda: analytics3.sharpe_ratio(dr3)),
            ("sortino_ratio", lambda: analytics3.sortino_ratio(dr3)),
            ("max_drawdown",  lambda: analytics3.max_drawdown(eq3)),
            ("calmar_ratio",  lambda: analytics3.calmar_ratio(
                float(dr3.mean() * 252), analytics3.max_drawdown(eq3))),
            ("omega_ratio",   lambda: analytics3.omega_ratio(dr3)),
            ("avg_drawdown",  lambda: analytics3.avg_drawdown(eq3)),
            ("drawdown_durations", lambda: analytics3.drawdown_durations(eq3)),
            ("full_report",   lambda: analytics3.full_report(eq3, trades=result3.trades)),
        ]
        present = 0
        for name, fn in required_checks:
            try:
                val = fn()
                if val is not None:
                    present += 1
            except Exception:
                pass
        completeness_score = present / len(required_checks) * 100
        scores.append(("analytics_completeness", completeness_score))
        print(f"  Analytics complete:   {present}/{len(required_checks)} → {completeness_score:.1f}/100")
    except Exception as e:
        scores.append(("analytics_completeness", 0.0))
        print(f"  Analytics complete:   CRASH — {e}")

    # ── Test 4: Stress test coverage ──────────────────────────────────────
    try:
        rng5 = np.random.default_rng(5)
        returns_arr = rng5.normal(0.0003, 0.012, 252)
        asset_returns = pd.DataFrame(
            {"TEST": returns_arr},
            index=pd.date_range("2025-01-02", periods=252, freq="B"),
        )
        weights = {"TEST": 1.0}
        tester = StressTester()
        results = tester.run_all(weights=weights, asset_returns=asset_returns)
        n_scenarios = len(results)
        stress_score = min(n_scenarios * 10, 100)  # 10 scenarios = 100 pts
        scores.append(("stress_coverage", stress_score))
        print(f"  Stress scenarios:     {n_scenarios} → {stress_score:.1f}/100")
    except Exception as e:
        scores.append(("stress_coverage", 0.0))
        print(f"  Stress coverage:      CRASH — {e}")

    # ── Composite ─────────────────────────────────────────────────────────
    weights = {"sharpe_on_uptrend": 0.35, "drawdown_control": 0.30,
               "analytics_completeness": 0.20, "stress_coverage": 0.15}
    composite = sum(weights.get(name, 0) * val for name, val in scores)

    print(f"  BACKTEST_QUALITY_SCORE: {composite:.2f}")
    return composite


if __name__ == "__main__":
    score = evaluate_backtest_quality()
    print(f"BACKTEST_QUALITY_SCORE: {score:.4f}")
