#!/usr/bin/env python3
"""
Multi-Asset Walk-Forward Validation
====================================
Runs walk-forward on all 8 assets individually and on the risk-parity portfolio.
Tests whether optimised parameters generalise across time windows.

Usage:
    python3 scripts/walk_forward_multi.py
    python3 scripts/walk_forward_multi.py --symbols BTCUSDT ETHUSDT SOLUSDT
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import numpy as np
import pandas as pd
import time
import argparse
import importlib

from scripts.eval_multi_asset import (
    load_data, _atr_vec,
    generate_atr_expansion_entries, generate_donchian_entries,
    simulate_positions, compute_metrics, compute_risk_parity_weights,
    INITIAL_CAPITAL, FEE, WARMUP
)

WINDOW_MONTHS = 6   # Each test window
N_WINDOWS = 6       # Number of walk-forward windows


def walk_forward_single(symbol: str, cfg: dict):
    """Run walk-forward on a single asset. Returns per-window metrics."""
    df = load_data(symbol)
    n = len(df)
    window_size = n // (N_WINDOWS + 1)  # Reserve first window as initial train

    results = []
    for w in range(N_WINDOWS):
        # Test window
        test_start = (w + 1) * window_size
        test_end = min((w + 2) * window_size, n)
        if test_end - test_start < 500:
            continue

        test_df = df.iloc[max(0, test_start - WARMUP):test_end].copy()
        close = test_df['close'].values.astype(float)
        high = test_df['high'].values.astype(float)
        low = test_df['low'].values.astype(float)
        atr = _atr_vec(high, low, close, cfg.get('atr_period', 14))

        strategy = cfg.get('strategy', 'ATR_EXPANSION')
        if strategy == 'ATR_EXPANSION':
            entries = generate_atr_expansion_entries(test_df, cfg)
        else:
            entries = generate_donchian_entries(test_df, cfg)

        pnl, trades = simulate_positions(
            entries, close, high, low, atr,
            cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])

        if len(trades) < 3:
            results.append({'window': w, 'sharpe': 0, 'return': 0, 'max_dd': 0, 'trades': 0})
            continue

        metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
        results.append({
            'window': w,
            'sharpe': metrics['sharpe'],
            'return': metrics['total_return'],
            'max_dd': metrics['max_dd'],
            'trades': metrics['n_trades'],
            'pnl': pnl,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=None)
    args = parser.parse_args()

    t0 = time.time()

    mod = importlib.import_module('tradingagents.research.per_asset_router')
    importlib.reload(mod)
    ASSET_CONFIG = mod.ASSET_CONFIG

    symbols = args.symbols or list(ASSET_CONFIG.keys())
    symbols = [s.upper() for s in symbols]

    print("=" * 90)
    print(f"MULTI-ASSET WALK-FORWARD VALIDATION ({N_WINDOWS} windows, ~{WINDOW_MONTHS}mo each)")
    print(f"Assets: {len(symbols)}")
    print("=" * 90)

    # Per-asset walk-forward
    all_wf = {}
    for sym in symbols:
        cfg = ASSET_CONFIG.get(sym)
        if cfg is None:
            continue
        wf = walk_forward_single(sym, cfg)
        all_wf[sym] = wf

    # Print per-asset summary
    print(f"\n{'Symbol':10s} {'FullSharpe':>10s} {'AvgWF':>8s} {'MinWF':>8s} {'MaxWF':>8s} "
          f"{'Pos/Total':>10s} {'OverfitR':>10s}")
    print("-" * 70)

    from scripts.eval_multi_asset import eval_single_asset

    for sym in symbols:
        if sym not in all_wf:
            continue
        wf = all_wf[sym]
        cfg = ASSET_CONFIG[sym]
        full_metrics = eval_single_asset(sym, cfg)
        full_sharpe = full_metrics['sharpe']

        sharpes = [w['sharpe'] for w in wf]
        avg_sharpe = np.mean(sharpes) if sharpes else 0
        min_sharpe = min(sharpes) if sharpes else 0
        max_sharpe = max(sharpes) if sharpes else 0
        positive = sum(1 for s in sharpes if s > 0)
        overfit = (1 - avg_sharpe / full_sharpe) * 100 if full_sharpe > 0 else 0

        print(f"{sym:10s} {full_sharpe:10.3f} {avg_sharpe:8.3f} {min_sharpe:8.3f} {max_sharpe:8.3f} "
              f"  {positive}/{len(sharpes)}      {overfit:9.1f}%")

    # Portfolio walk-forward (risk-parity across all assets per window)
    print(f"\n{'─' * 90}")
    print("PORTFOLIO WALK-FORWARD (risk-parity per window)")
    print(f"{'─' * 90}")

    # Find the minimum number of windows across all assets
    min_windows = min(len(wf) for wf in all_wf.values())
    port_sharpes = []

    for w in range(min_windows):
        # Collect PnL arrays for this window from all assets
        window_pnl = {}
        for sym in symbols:
            if sym not in all_wf:
                continue
            wf = all_wf[sym]
            if w < len(wf) and 'pnl' in wf[w]:
                window_pnl[sym] = wf[w]['pnl']

        if len(window_pnl) < 2:
            port_sharpes.append(0)
            print(f"  Window {w}: insufficient data")
            continue

        # Risk-parity weights for this window
        weights = compute_risk_parity_weights(window_pnl)
        min_len = min(len(p) for p in window_pnl.values())
        port_pnl = np.zeros(min_len)
        for sym, pnl in window_pnl.items():
            port_pnl += weights[sym] * pnl[:min_len]

        port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_pnl)
        port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100
        port_sharpe = float(np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760)) if np.std(port_pnl) > 0 else 0
        peak = np.maximum.accumulate(port_equity)
        port_dd = float(np.max((peak - port_equity) / peak) * 100)

        port_sharpes.append(port_sharpe)
        print(f"  Window {w}: Sharpe {port_sharpe:6.3f} | Return {port_return:7.1f}% | MaxDD {port_dd:5.1f}%")

    # Portfolio summary
    full_port_sharpe = 2.78  # From full eval
    avg_port_sharpe = np.mean(port_sharpes) if port_sharpes else 0
    positive_windows = sum(1 for s in port_sharpes if s > 0)
    overfit_ratio = (1 - avg_port_sharpe / full_port_sharpe) * 100 if full_port_sharpe > 0 else 0

    print(f"\n{'=' * 90}")
    print("PORTFOLIO WALK-FORWARD SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Full-period Sharpe:  {full_port_sharpe:.3f}")
    print(f"  Avg WF Sharpe:       {avg_port_sharpe:.3f}")
    print(f"  Min WF Sharpe:       {min(port_sharpes):.3f}" if port_sharpes else "  Min WF Sharpe: N/A")
    print(f"  Max WF Sharpe:       {max(port_sharpes):.3f}" if port_sharpes else "  Max WF Sharpe: N/A")
    print(f"  Positive windows:    {positive_windows}/{len(port_sharpes)}")
    print(f"  Overfit ratio:       {overfit_ratio:.1f}%")

    elapsed = time.time() - t0
    print(f"\n  Eval time: {elapsed:.1f}s")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
