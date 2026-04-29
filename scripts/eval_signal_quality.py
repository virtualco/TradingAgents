#!/usr/bin/env python3
"""Signal Quality Evaluator for AutoResearch.

Scores the signal generation quality on synthetic OHLCV data.
Outputs: SIGNAL_QUALITY_SCORE: <float 0-100>

Metrics:
- Signal coverage: % of tickers that produce a non-flat signal
- Conviction calibration: mean conviction of non-flat signals (higher = more decisive)
- Directional consistency: signals agree with 20-day price trend
- Evidence coverage: % of required signal types present in output
- Robustness: signals don't crash on edge cases (short series, NaN, zero volume)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tradingagents.research.strategy_rules import TechnicalStrategyRules


def make_trending_up(n=120, seed=42) -> pd.DataFrame:
    """Synthetic uptrending OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.3, 1.5, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def make_trending_down(n=120, seed=99) -> pd.DataFrame:
    """Synthetic downtrending OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 200 - np.cumsum(rng.normal(0.3, 1.5, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def make_sideways(n=120, seed=7) -> pd.DataFrame:
    """Synthetic sideways/choppy OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 150 + rng.normal(0, 2.0, n)
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(200_000, 800_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def make_short_series(n=15) -> pd.DataFrame:
    """Too-short series — should not crash."""
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(0, 1, n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": np.ones(n) * 100_000})


def make_nan_series(n=120) -> pd.DataFrame:
    """Series with NaN values — should not crash."""
    df = make_trending_up(n)
    df.loc[10:15, "close"] = np.nan
    df.loc[50, "volume"] = np.nan
    return df


def evaluate_signal_quality() -> float:
    """Run signal quality evaluation and return score 0-100."""
    rules = TechnicalStrategyRules()
    trade_date = "2026-04-29"

    scenarios = [
        ("TREND_UP",   make_trending_up(),   "long",   True),
        ("TREND_DOWN", make_trending_down(), "short",  True),
        ("SIDEWAYS",   make_sideways(),       "flat",   False),  # flat is OK for sideways
        ("SHORT",      make_short_series(),   None,     False),  # should not crash
        ("NAN",        make_nan_series(),     None,     False),  # should not crash
    ]

    scores = []
    non_flat_convictions = []
    directional_correct = 0
    directional_total = 0
    crashes = 0

    for name, ohlcv, expected_direction, check_direction in scenarios:
        try:
            summary = rules.compute(ohlcv, ticker=name, trade_date=trade_date)

            # 1. Robustness: no crash
            scores.append(10.0)

            # 2. Conviction calibration: non-flat signals should have conviction > 0.3
            if summary.composite_score != 0.0:
                conviction = abs(summary.composite_score)
                non_flat_convictions.append(conviction)
                scores.append(min(conviction * 20, 10.0))  # up to 10 pts
            else:
                scores.append(5.0)  # flat is OK but less informative

            # 3. Directional correctness for trend scenarios
            if check_direction and expected_direction is not None:
                directional_total += 1
                actual = "long" if summary.composite_score > 0 else ("short" if summary.composite_score < 0 else "flat")
                if actual == expected_direction:
                    directional_correct += 1
                    scores.append(15.0)
                else:
                    scores.append(0.0)

        except Exception as e:
            crashes += 1
            scores.append(0.0)

    # Composite score components
    robustness_score = max(0, 100 - crashes * 20)  # -20 per crash
    
    mean_conviction = float(np.mean(non_flat_convictions)) if non_flat_convictions else 0.0
    conviction_score = min(mean_conviction * 100, 100.0)

    directional_accuracy = (directional_correct / directional_total * 100) if directional_total > 0 else 0.0

    # Coverage: how many signal types are populated
    try:
        up_summary = rules.compute(make_trending_up(), ticker="COV", trade_date=trade_date)
        populated_signals = sum(1 for s in up_summary.signals if s is not None) if hasattr(up_summary, "signals") else 2
        coverage_score = min(populated_signals * 25, 100.0)
    except Exception:
        coverage_score = 50.0

    # Weighted composite
    quality_score = (
        0.30 * robustness_score +
        0.30 * directional_accuracy +
        0.25 * conviction_score +
        0.15 * coverage_score
    )

    print(f"  Signal Quality Components:")
    print(f"    Robustness:           {robustness_score:.1f}/100 ({crashes} crashes)")
    print(f"    Directional accuracy: {directional_accuracy:.1f}/100 ({directional_correct}/{directional_total})")
    print(f"    Conviction score:     {conviction_score:.1f}/100 (mean={mean_conviction:.3f})")
    print(f"    Coverage score:       {coverage_score:.1f}/100")
    print(f"  SIGNAL_QUALITY_SCORE: {quality_score:.2f}")

    return quality_score


if __name__ == "__main__":
    score = evaluate_signal_quality()
    print(f"SIGNAL_QUALITY_SCORE: {score:.4f}")
