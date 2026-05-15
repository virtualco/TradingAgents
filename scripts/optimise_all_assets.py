#!/usr/bin/env python3
"""
Batch Bayesian Optimisation for All Assets
===========================================
Runs Optuna TPE on each symbol, optimising strategy-specific parameters.
Outputs optimal configs and applies them to per_asset_router.py.

Usage:
    python3 scripts/optimise_all_assets.py                    # All 8 assets, 300 trials each
    python3 scripts/optimise_all_assets.py --symbols SOLUSDT  # Single asset
    python3 scripts/optimise_all_assets.py --trials 500       # More trials
    python3 scripts/optimise_all_assets.py --apply            # Auto-apply best params
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import numpy as np
import time
import argparse
import json
import optuna
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

from scripts.eval_multi_asset import (
    load_data, eval_single_asset,
    generate_atr_expansion_entries, generate_donchian_entries,
    simulate_positions, compute_metrics,
    _atr_vec
)
from tradingagents.research.per_asset_router import ASSET_CONFIG

INITIAL_CAPITAL = 100_000.0


def make_atr_expansion_objective(df, symbol):
    """Create Optuna objective for ATR Expansion strategy."""
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    def objective(trial):
        cfg = {
            'strategy':       'ATR_EXPANSION',
            'atr_period':     trial.suggest_int('atr_period', 10, 25),
            'expansion_mult': trial.suggest_float('expansion_mult', 1.5, 5.0, step=0.1),
            'vol_mult':       trial.suggest_float('vol_mult', 1.0, 4.0, step=0.1),
            'max_hold_bars':  trial.suggest_int('max_hold_bars', 4, 36, step=2),
            'stop_mult':      trial.suggest_float('stop_mult', 0.8, 4.0, step=0.1),
            'tp_mult':        trial.suggest_float('tp_mult', 2.0, 12.0, step=0.5),
            'order_type':     'Market',
        }
        atr = _atr_vec(high, low, close, cfg['atr_period'])
        entries = generate_atr_expansion_entries(df, cfg)
        pnl, trades = simulate_positions(
            entries, close, high, low, atr,
            cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])

        if len(trades) < 20:
            return -10.0

        metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
        # Penalise if too few trades (want at least 50)
        trade_penalty = max(0, (50 - len(trades)) * 0.01)
        return metrics['sharpe'] - trade_penalty

    return objective


def make_donchian_objective(df, symbol):
    """Create Optuna objective for Donchian Momentum strategy."""
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    def objective(trial):
        cfg = {
            'strategy':          'DONCHIAN_MOMENTUM',
            'donchian_period':   trial.suggest_int('donchian_period', 15, 45),
            'adx_min':           trial.suggest_int('adx_min', 10, 30),
            'adx_trend':         trial.suggest_int('adx_trend', 15, 35),
            'vol_mult':          trial.suggest_float('vol_mult', 1.0, 4.0, step=0.1),
            'hurst_min':         trial.suggest_float('hurst_min', 0.35, 0.65, step=0.01),
            'vol_atr_max':       trial.suggest_float('vol_atr_max', 0.02, 0.15, step=0.01),
            'max_hold_bars':     trial.suggest_int('max_hold_bars', 12, 120, step=6),
            'stop_mult':         trial.suggest_float('stop_mult', 0.8, 5.0, step=0.1),
            'tp_mult':           trial.suggest_float('tp_mult', 2.0, 12.0, step=0.5),
            'order_type':        'Market',
            'atr_donchian_factor': trial.suggest_categorical('atr_donchian_factor', [None, 0.5, 1.0, 1.5, 2.0]),
        }
        atr = _atr_vec(high, low, close, 14)
        entries = generate_donchian_entries(df, cfg)
        pnl, trades = simulate_positions(
            entries, close, high, low, atr,
            cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])

        if len(trades) < 15:
            return -10.0

        metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
        trade_penalty = max(0, (40 - len(trades)) * 0.01)
        return metrics['sharpe'] - trade_penalty

    return objective


def optimise_symbol(symbol: str, n_trials: int = 300):
    """Run Bayesian optimisation for a single symbol."""
    cfg = ASSET_CONFIG[symbol]
    strategy = cfg['strategy']

    print(f"\n{'='*60}")
    print(f"Optimising {symbol} ({strategy}) — {n_trials} trials")
    print(f"{'='*60}")

    t0 = time.time()
    df = load_data(symbol)

    if strategy == 'ATR_EXPANSION':
        objective = make_atr_expansion_objective(df, symbol)
    else:
        objective = make_donchian_objective(df, symbol)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best['strategy'] = strategy
    best['order_type'] = cfg.get('order_type', 'Market')

    # Evaluate best params
    metrics = eval_single_asset(symbol, best)
    elapsed = time.time() - t0

    print(f"  Best Sharpe: {metrics['sharpe']:.3f} | Return: {metrics['total_return']:.1f}% | "
          f"MaxDD: {metrics['max_dd']:.1f}% | Trades: {metrics['n_trades']} | "
          f"WinRate: {metrics['win_rate']:.1f}% | PF: {metrics['pf']:.2f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Best params: {json.dumps({k: v for k, v in best.items() if k not in ('strategy', 'order_type')}, indent=2)}")

    return {
        'symbol': symbol,
        'strategy': strategy,
        'params': best,
        'metrics': {k: v for k, v in metrics.items() if k != 'pnl'},
        'n_trials': n_trials,
        'time_s': elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--trials', type=int, default=300)
    parser.add_argument('--apply', action='store_true', help='Auto-apply best params to per_asset_router.py')
    args = parser.parse_args()

    symbols = args.symbols or list(ASSET_CONFIG.keys())
    symbols = [s.upper() for s in symbols]

    print(f"Batch Bayesian Optimisation: {len(symbols)} assets, {args.trials} trials each")

    results = []
    for sym in symbols:
        if sym not in ASSET_CONFIG:
            print(f"  WARNING: {sym} not in ASSET_CONFIG, skipping")
            continue
        result = optimise_symbol(sym, args.trials)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("OPTIMISATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Symbol':10s} {'Strategy':20s} {'Sharpe':>8s} {'Return':>10s} {'MaxDD':>8s} "
          f"{'Trades':>8s} {'WinRate':>8s} {'PF':>6s} {'Time':>6s}")
    print("-" * 80)
    for r in results:
        m = r['metrics']
        print(f"{r['symbol']:10s} {r['strategy']:20s} {m['sharpe']:8.3f} {m['total_return']:9.1f}% "
              f"{m['max_dd']:7.1f}% {m['n_trades']:8d} {m['win_rate']:7.1f}% {m['pf']:5.2f} {r['time_s']:5.1f}s")

    # Save results
    out_path = Path('/home/ubuntu/TradingAgents/data/optimisation_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print configs for copy-paste into per_asset_router.py
    print(f"\n{'='*80}")
    print("OPTIMAL CONFIGS (for per_asset_router.py)")
    print(f"{'='*80}")
    for r in results:
        sym = r['symbol']
        params = r['params']
        label = sym.replace('USDT', '')
        print(f"\n{label}_CONFIG = {{")
        for k, v in sorted(params.items()):
            if isinstance(v, str):
                print(f"    '{k}': '{v}',")
            elif v is None:
                print(f"    '{k}': None,")
            elif isinstance(v, float):
                print(f"    '{k}': {v},")
            else:
                print(f"    '{k}': {v},")
        print("}")


if __name__ == '__main__':
    main()
