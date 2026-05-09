"""
Out-of-Sample Validation — Dual-Regime Strategy
================================================
Tests the DualRegimeStrategy across 4 years of BTC and ETH hourly data
using walk-forward analysis across 4 market regimes.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from tradingagents.research.dual_regime_strategy import DualRegimeStrategy

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS = {
    "BTC": "data/historical/BTC_USD_1h_2022-01-01_2026-01-01.parquet",
    "ETH": "data/historical/ETH_USD_1h_2022-01-01_2026-01-01.parquet",
}
TRANSACTION_COST = 0.001   # 0.1% per trade (Bybit taker fee)
PERIODS = [
    ("2022-01-01", "2022-12-31", "Bear Market 2022"),
    ("2023-01-01", "2023-12-31", "Recovery 2023"),
    ("2024-01-01", "2024-12-31", "Bull Market 2024"),
    ("2025-01-01", "2025-12-31", "Post-ATH 2025"),
]
WFA_FOLDS = 12   # 12 monthly walk-forward folds


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    return df


def backtest_period(df: pd.DataFrame, strategy: DualRegimeStrategy, label: str) -> dict:
    """Run a vectorised backtest on a period slice."""
    if len(df) < 200:
        return {"period": label, "error": "insufficient data"}

    signals = strategy.generate_signals(df)
    close   = df["close"]

    # Vectorised P&L
    returns = close.pct_change().fillna(0)
    # Position is held from signal bar to next bar
    pos_returns = signals.shift(1).fillna(0) * returns

    # Apply transaction costs on signal changes
    signal_changes = (signals != signals.shift(1)).astype(int)
    pos_returns -= signal_changes * TRANSACTION_COST

    # Metrics
    total_return   = (1 + pos_returns).prod() - 1
    n_weeks        = len(df) / (24 * 7)
    weekly_return  = (1 + total_return) ** (1 / max(n_weeks, 1)) - 1
    sharpe         = (pos_returns.mean() / pos_returns.std() * np.sqrt(8760)) if pos_returns.std() > 0 else 0
    cum_ret        = (1 + pos_returns).cumprod()
    drawdown       = (cum_ret / cum_ret.cummax() - 1).min()
    n_trades       = signal_changes.sum()
    win_rate       = (pos_returns[pos_returns != 0] > 0).mean() if n_trades > 0 else 0

    # Regime distribution
    regime = strategy.get_regime(df)
    regime_dist = regime.value_counts(normalize=True).to_dict()

    return {
        "period": label,
        "total_return_pct": round(total_return * 100, 2),
        "weekly_return_pct": round(weekly_return * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(abs(drawdown) * 100, 2),
        "n_trades": int(n_trades),
        "win_rate_pct": round(float(win_rate) * 100, 1),
        "regime_trending_pct": round(regime_dist.get("TRENDING", 0) * 100, 1),
        "regime_ranging_pct": round(regime_dist.get("RANGING", 0) * 100, 1),
        "regime_transition_pct": round(regime_dist.get("TRANSITION", 0) * 100, 1),
    }


def walk_forward_analysis(df: pd.DataFrame, strategy: DualRegimeStrategy) -> dict:
    """Monthly walk-forward analysis across the full dataset."""
    df = df.sort_index()
    monthly_groups = df.groupby(pd.Grouper(freq="ME"))
    monthly_returns = []

    for month_end, month_df in monthly_groups:
        if len(month_df) < 100:
            continue
        result = backtest_period(month_df, strategy, str(month_end.date()))
        if "error" not in result:
            monthly_returns.append(result["weekly_return_pct"])

    if not monthly_returns:
        return {"wfa_avg_weekly_return": 0, "wfa_profitable_folds_pct": 0, "wfa_sharpe": 0}

    arr = np.array(monthly_returns)
    return {
        "wfa_avg_weekly_return_pct": round(float(arr.mean()), 3),
        "wfa_profitable_folds_pct": round(float((arr > 0).mean() * 100), 1),
        "wfa_sharpe": round(float(arr.mean() / arr.std()) if arr.std() > 0 else 0, 3),
        "wfa_n_folds": len(arr),
        "wfa_best_month_pct": round(float(arr.max()), 2),
        "wfa_worst_month_pct": round(float(arr.min()), 2),
    }


def compute_score(results: list, wfa: dict) -> float:
    """
    Composite OOS score (0–100):
      40% Walk-forward profitable folds (target: >50%)
      30% Average WFA weekly return (target: >1%)
      20% Max drawdown control (target: <20%)
      10% Sharpe ratio (target: >0.5)
    """
    profitable_folds = wfa.get("wfa_profitable_folds_pct", 0)
    avg_weekly       = wfa.get("wfa_avg_weekly_return_pct", 0)
    avg_drawdown     = np.mean([r.get("max_drawdown_pct", 100) for r in results if "error" not in r])
    avg_sharpe       = np.mean([r.get("sharpe", 0) for r in results if "error" not in r])

    score_folds    = min(profitable_folds / 60 * 40, 40)
    score_return   = min(max(avg_weekly / 2.0 * 30, 0), 30)
    score_drawdown = min(max((30 - avg_drawdown) / 30 * 20, 0), 20)
    score_sharpe   = min(max(avg_sharpe / 1.0 * 10, 0), 10)

    return round(score_folds + score_return + score_drawdown + score_sharpe, 2)


def main():
    print("=" * 70)
    print("DUAL-REGIME STRATEGY — OUT-OF-SAMPLE VALIDATION")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    strategy = DualRegimeStrategy()
    all_results = {}

    for symbol, path in SYMBOLS.items():
        print(f"\n{'='*35} {symbol} {'='*35}")
        df = load_data(path)
        print(f"Loaded {len(df)} bars | {df.index[0]} → {df.index[-1]}")

        period_results = []
        for start, end, label in PERIODS:
            period_df = df.loc[start:end]
            result = backtest_period(period_df, strategy, label)
            period_results.append(result)
            if "error" not in result:
                print(
                    f"  {label:<22} | return={result['total_return_pct']:+6.1f}% "
                    f"| weekly={result['weekly_return_pct']:+5.2f}% "
                    f"| Sharpe={result['sharpe']:+.2f} "
                    f"| DD={result['max_drawdown_pct']:.1f}% "
                    f"| trades={result['n_trades']} "
                    f"| T:{result['regime_trending_pct']:.0f}% R:{result['regime_ranging_pct']:.0f}%"
                )
            else:
                print(f"  {label}: {result['error']}")

        # Walk-forward analysis on full dataset
        wfa = walk_forward_analysis(df, strategy)
        print(f"\n  Walk-Forward Analysis ({wfa.get('wfa_n_folds', 0)} monthly folds):")
        print(f"    Avg weekly return: {wfa.get('wfa_avg_weekly_return_pct', 0):+.3f}%")
        print(f"    Profitable folds:  {wfa.get('wfa_profitable_folds_pct', 0):.1f}%")
        print(f"    WFA Sharpe:        {wfa.get('wfa_sharpe', 0):.3f}")
        print(f"    Best month:        {wfa.get('wfa_best_month_pct', 0):+.2f}%")
        print(f"    Worst month:       {wfa.get('wfa_worst_month_pct', 0):+.2f}%")

        score = compute_score(period_results, wfa)
        print(f"\n  OOS COMPOSITE SCORE: {score:.2f}/100")

        all_results[symbol] = {
            "periods": period_results,
            "wfa": wfa,
            "score": score,
        }

    # Overall score
    avg_score = np.mean([v["score"] for v in all_results.values()])
    print(f"\n{'='*70}")
    print(f"OVERALL OOS SCORE: {avg_score:.2f}/100")
    print(f"{'='*70}")

    # Save results
    output = {
        "strategy": "DualRegimeStrategy",
        "run_date": datetime.now().isoformat(),
        "results": all_results,
        "avg_score": avg_score,
    }
    Path("data/dual_regime_oos_results.json").write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to data/dual_regime_oos_results.json")

    return avg_score


if __name__ == "__main__":
    main()
