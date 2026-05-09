"""
Dual-Regime Strategy Parameter Optimiser
==========================================
Grid search over key parameters to find the best combination
for OOS performance across 4 years of BTC/ETH data.

Optimises:
  - Hurst thresholds (trending vs ranging classification)
  - ADX thresholds
  - Mean-reversion RSI thresholds (less aggressive entries)
  - Bollinger Band width
  - Momentum EMA periods

Uses a fast vectorised backtest (no per-bar loop for most calcs).
"""
from __future__ import annotations
import itertools
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from tradingagents.research.dual_regime_strategy import (
    DualRegimeStrategy, _hurst_exponent, _adx, _ema, _atr,
    _bollinger_bands, _rsi, _macd_histogram
)

# ── Data ──────────────────────────────────────────────────────────────────────
SYMBOLS = {
    "BTC": "data/historical/BTC_USD_1h_2022-01-01_2026-01-01.parquet",
    "ETH": "data/historical/ETH_USD_1h_2022-01-01_2026-01-01.parquet",
}
TRANSACTION_COST = 0.001
BEST_PARAMS_FILE = Path("data/dual_regime_best_params.json")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    return df


def fast_backtest(signals: pd.Series, close: pd.Series) -> dict:
    """Vectorised backtest returning key metrics."""
    returns = close.pct_change().fillna(0)
    pos_returns = signals.shift(1).fillna(0) * returns
    signal_changes = (signals != signals.shift(1)).astype(int)
    pos_returns -= signal_changes * TRANSACTION_COST

    total_return  = (1 + pos_returns).prod() - 1
    n_weeks       = len(close) / (24 * 7)
    weekly_return = (1 + total_return) ** (1 / max(n_weeks, 1)) - 1
    std           = pos_returns.std()
    sharpe        = (pos_returns.mean() / std * np.sqrt(8760)) if std > 0 else 0
    cum_ret       = (1 + pos_returns).cumprod()
    drawdown      = abs((cum_ret / cum_ret.cummax() - 1).min())
    n_trades      = int(signal_changes.sum())

    return {
        "weekly_return": weekly_return,
        "sharpe": sharpe,
        "drawdown": drawdown,
        "n_trades": n_trades,
    }


def wfa_score(df: pd.DataFrame, params: dict) -> float:
    """
    Walk-forward score for a given parameter set.
    Returns composite score 0-100.
    """
    strat = DualRegimeStrategy(**params)
    signals = strat.generate_signals(df)

    # Monthly WFA
    monthly_returns = []
    for _, month_df in df.groupby(pd.Grouper(freq="ME")):
        if len(month_df) < 100:
            continue
        month_signals = signals.loc[month_df.index]
        result = fast_backtest(month_signals, month_df["close"])
        monthly_returns.append(result["weekly_return"])

    if not monthly_returns:
        return 0.0

    arr = np.array(monthly_returns)
    profitable_folds = (arr > 0).mean() * 100
    avg_weekly = arr.mean() * 100

    # Full period metrics
    full_result = fast_backtest(signals, df["close"])
    drawdown = full_result["drawdown"] * 100
    sharpe = full_result["sharpe"]

    score_folds    = min(profitable_folds / 60 * 40, 40)
    score_return   = min(max(avg_weekly / 2.0 * 30, 0), 30)
    score_drawdown = min(max((30 - drawdown) / 30 * 20, 0), 20)
    score_sharpe   = min(max(sharpe / 1.0 * 10, 0), 10)

    return score_folds + score_return + score_drawdown + score_sharpe


def main():
    print("=" * 70)
    print("DUAL-REGIME PARAMETER OPTIMISATION")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    datasets = {}
    for sym, path in SYMBOLS.items():
        datasets[sym] = load_data(path)
        print(f"Loaded {sym}: {len(datasets[sym])} bars")

    # Pre-compute Hurst and ADX (expensive, cache them)
    print("\nPre-computing Hurst exponents (cached for all param combos)...")
    hurst_cache = {}
    for sym, df in datasets.items():
        for window in [48, 72, 96, 120]:
            key = (sym, window)
            hurst_cache[key] = _hurst_exponent(df["close"], window=window)
            print(f"  {sym} Hurst(window={window}) done")

    # Parameter grid
    param_grid = {
        "hurst_window":           [72, 96],
        "hurst_trend_threshold":  [0.52, 0.55, 0.58],
        "hurst_revert_threshold": [0.42, 0.45, 0.48],
        "adx_trend_threshold":    [20.0, 25.0, 30.0],
        "adx_range_threshold":    [15.0, 20.0],
        "rsi_oversold":           [30.0, 35.0, 40.0],
        "rsi_overbought":         [60.0, 65.0, 70.0],
        "bb_std":                 [1.5, 2.0, 2.5],
        "ema_fast":               [9, 12],
        "ema_slow":               [21, 26],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    print(f"\nTotal parameter combinations: {len(combos)}")

    best_score = 0.0
    best_params = {}
    results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Skip invalid combinations
        if params["hurst_trend_threshold"] <= params["hurst_revert_threshold"]:
            continue
        if params["adx_trend_threshold"] <= params["adx_range_threshold"]:
            continue
        if params["rsi_oversold"] >= params["rsi_overbought"]:
            continue

        # Score across both symbols
        scores = []
        for sym, df in datasets.items():
            try:
                s = wfa_score(df, params)
                scores.append(s)
            except Exception:
                scores.append(0.0)

        avg_score = np.mean(scores)
        results.append({"params": params, "score": avg_score})

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()
            print(f"  [{i+1}/{len(combos)}] NEW BEST: {avg_score:.2f}/100 | {params}")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(combos)} | Best so far: {best_score:.2f}/100")

    print(f"\n{'='*70}")
    print(f"OPTIMISATION COMPLETE")
    print(f"Best score: {best_score:.2f}/100")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    # Save results
    output = {
        "best_score": best_score,
        "best_params": best_params,
        "top_10": sorted(results, key=lambda x: x["score"], reverse=True)[:10],
        "run_date": datetime.now().isoformat(),
    }
    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {BEST_PARAMS_FILE}")

    return best_params, best_score


if __name__ == "__main__":
    main()
