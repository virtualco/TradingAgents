#!/usr/bin/env python3
"""
Walk-Forward Validation — Anti-Overfitting Guard
==================================================
Splits the 4-year hourly dataset into rolling train/test windows.
Optimises parameters on each training window using Optuna, then evaluates
on the subsequent out-of-sample test window.

Reports the average OOS Sharpe across all test windows — a much more
robust metric than single-period OOS Sharpe.

Usage:
  python3 scripts/walk_forward.py [--train-months 12] [--test-months 6] [--trials 100]
  python3 scripts/walk_forward.py --validate-only  # test current params on all windows
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')
import argparse
import json
import time
import numpy as np
import pandas as pd
import optuna
from scripts.eval_per_asset_oos import (
    _atr_vec, _adx_vec, _hurst_fast_vec, _rolling_mean,
    generate_btc_entries, generate_eth_entries,
    simulate_positions, compute_metrics,
    INITIAL_CAPITAL, BTC_WEIGHT, ETH_WEIGHT, WARMUP
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_full_data(symbol: str) -> pd.DataFrame:
    path = f'/home/ubuntu/TradingAgents/data/historical/{symbol}_USD_1h_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ── Window Evaluation ─────────────────────────────────────────────────────────
def eval_window(btc_df: pd.DataFrame, eth_df: pd.DataFrame,
                btc_cfg: dict, eth_cfg: dict) -> dict:
    """Evaluate a BTC+ETH portfolio on a data window. Returns metrics dict."""
    btc_close = btc_df['close'].values.astype(float)
    btc_high  = btc_df['high'].values.astype(float)
    btc_low   = btc_df['low'].values.astype(float)
    eth_close = eth_df['close'].values.astype(float)
    eth_high  = eth_df['high'].values.astype(float)
    eth_low   = eth_df['low'].values.astype(float)

    btc_atr = _atr_vec(btc_high, btc_low, btc_close, btc_cfg.get('atr_period', 14))
    eth_atr = _atr_vec(eth_high, eth_low, eth_close, 14)

    btc_entries = generate_btc_entries(btc_df, btc_cfg)
    eth_entries = generate_eth_entries(eth_df, eth_cfg)

    btc_pnl, btc_trades = simulate_positions(
        btc_entries, btc_close, btc_high, btc_low, btc_atr,
        btc_cfg['stop_mult'], btc_cfg['tp_mult'], btc_cfg['max_hold_bars'])
    eth_pnl, eth_trades = simulate_positions(
        eth_entries, eth_close, eth_high, eth_low, eth_atr,
        eth_cfg['stop_mult'], eth_cfg['tp_mult'], eth_cfg['max_hold_bars'])

    min_len = min(len(btc_pnl), len(eth_pnl))
    if min_len < WARMUP + 100:
        return {'sharpe': 0.0, 'total_return': 0.0, 'max_dd': 100.0,
                'n_trades': 0, 'win_rate': 0.0}

    port_pnl = BTC_WEIGHT * btc_pnl[:min_len] + ETH_WEIGHT * eth_pnl[:min_len]
    port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_pnl)
    port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100

    sharpe = 0.0
    if np.std(port_pnl) > 0:
        sharpe = float(np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760))

    peak = np.maximum.accumulate(port_equity)
    dd = (peak - port_equity) / peak
    max_dd = float(np.max(dd) * 100)

    all_trades = btc_trades + eth_trades
    n_trades = len(all_trades)
    win_rate = 0.0
    if n_trades > 0:
        win_rate = float(np.sum(np.array(all_trades) > 0) / n_trades * 100)

    return {
        'sharpe': sharpe,
        'total_return': port_return,
        'max_dd': max_dd,
        'n_trades': n_trades,
        'win_rate': win_rate,
    }


# ── Optuna Objective for a Training Window ────────────────────────────────────
def make_objective(btc_train: pd.DataFrame, eth_train: pd.DataFrame):
    """Create an Optuna objective function for a specific training window."""
    def objective(trial: optuna.Trial) -> float:
        btc_cfg = {
            'atr_period':     trial.suggest_int('btc_atr_period', 10, 20),
            'expansion_mult': trial.suggest_float('btc_expansion_mult', 2.0, 4.5, step=0.1),
            'vol_mult':       trial.suggest_float('btc_vol_mult', 0.8, 2.5, step=0.1),
            'max_hold_bars':  trial.suggest_int('btc_max_hold_bars', 6, 36, step=2),
            'stop_mult':      trial.suggest_float('btc_stop_mult', 0.8, 3.0, step=0.1),
            'tp_mult':        trial.suggest_float('btc_tp_mult', 2.0, 8.0, step=0.5),
        }
        use_atr_max = trial.suggest_categorical('eth_use_atr_max', [True, False])
        eth_cfg = {
            'donchian_period':    trial.suggest_int('eth_donchian_period', 15, 35),
            'adx_min':            trial.suggest_int('eth_adx_min', 8, 22),
            'adx_trend':          trial.suggest_int('eth_adx_trend', 14, 28),
            'vol_mult':           trial.suggest_float('eth_vol_mult', 0.8, 3.0, step=0.1),
            'hurst_min':          trial.suggest_float('eth_hurst_min', 0.30, 0.55, step=0.02),
            'vol_atr_max':        trial.suggest_float('eth_vol_atr_max', 0.02, 0.10, step=0.01) if use_atr_max else None,
            'max_hold_bars':      trial.suggest_int('eth_max_hold_bars', 12, 96, step=6),
            'stop_mult':          trial.suggest_float('eth_stop_mult', 0.8, 3.5, step=0.1),
            'tp_mult':            trial.suggest_float('eth_tp_mult', 3.0, 10.0, step=0.5),
            'atr_donchian_factor': None,
        }
        result = eval_window(btc_train, eth_train, btc_cfg, eth_cfg)
        return result['sharpe']
    return objective


def extract_configs(params: dict) -> tuple:
    """Extract BTC and ETH configs from Optuna trial params."""
    btc_cfg = {
        'atr_period':     params['btc_atr_period'],
        'expansion_mult': params['btc_expansion_mult'],
        'vol_mult':       params['btc_vol_mult'],
        'max_hold_bars':  params['btc_max_hold_bars'],
        'stop_mult':      params['btc_stop_mult'],
        'tp_mult':        params['btc_tp_mult'],
    }
    use_atr_max = params.get('eth_use_atr_max', True)
    eth_cfg = {
        'donchian_period':    params['eth_donchian_period'],
        'adx_min':            params['eth_adx_min'],
        'adx_trend':          params['eth_adx_trend'],
        'vol_mult':           params['eth_vol_mult'],
        'hurst_min':          params['eth_hurst_min'],
        'vol_atr_max':        params.get('eth_vol_atr_max') if use_atr_max else None,
        'max_hold_bars':      params['eth_max_hold_bars'],
        'stop_mult':          params['eth_stop_mult'],
        'tp_mult':            params['eth_tp_mult'],
        'atr_donchian_factor': None,
    }
    return btc_cfg, eth_cfg


# ── Walk-Forward Engine ───────────────────────────────────────────────────────
def run_walk_forward(train_months: int, test_months: int, n_trials: int,
                     validate_only: bool = False):
    """
    Rolling walk-forward: train on [t, t+train], test on [t+train, t+train+test].
    Step forward by test_months each iteration.
    """
    print("Loading full dataset...")
    btc_full = load_full_data('BTC')
    eth_full = load_full_data('ETH')

    start = btc_full.index.min()
    end = btc_full.index.max()
    print(f"Data range: {start.date()} to {end.date()}")
    print(f"Train: {train_months}mo, Test: {test_months}mo, Trials: {n_trials}")

    if validate_only:
        print("\n[VALIDATE-ONLY] Testing current params on all windows...")
        from tradingagents.research.per_asset_router import BTC_CONFIG, ETH_CONFIG
        btc_cfg = dict(BTC_CONFIG)
        eth_cfg = dict(ETH_CONFIG)
        print(f"BTC: {btc_cfg}")
        print(f"ETH: {eth_cfg}")

    windows = []
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        windows.append((cursor, train_end, test_end))
        cursor = cursor + pd.DateOffset(months=test_months)

    print(f"\n{len(windows)} walk-forward windows:\n")
    print(f"{'Window':<8} {'Train':>22} {'Test':>22} {'Train Sharpe':>13} {'Test Sharpe':>12} {'Test Ret':>10} {'Test DD':>9} {'Trades':>8}")
    print("-" * 110)

    results = []
    for i, (train_start, train_end, test_end) in enumerate(windows):
        btc_train = btc_full[train_start:train_end]
        eth_train = eth_full[train_start:train_end]
        btc_test = btc_full[train_end:test_end]
        eth_test = eth_full[train_end:test_end]

        if validate_only:
            train_m = eval_window(btc_train, eth_train, btc_cfg, eth_cfg)
            test_m = eval_window(btc_test, eth_test, btc_cfg, eth_cfg)
        else:
            # Optimise on training window
            objective = make_objective(btc_train, eth_train)
            study = optuna.create_study(direction='maximize',
                                         sampler=optuna.samplers.TPESampler(seed=42 + i))
            study.optimize(objective, n_trials=n_trials)

            btc_cfg, eth_cfg = extract_configs(study.best_params)
            train_m = {'sharpe': study.best_value}
            test_m = eval_window(btc_test, eth_test, btc_cfg, eth_cfg)

        results.append({
            'window': i + 1,
            'train_period': f"{train_start.date()} → {train_end.date()}",
            'test_period': f"{train_end.date()} → {test_end.date()}",
            'train_sharpe': train_m['sharpe'],
            'test_sharpe': test_m['sharpe'],
            'test_return': test_m['total_return'],
            'test_max_dd': test_m['max_dd'],
            'test_trades': test_m['n_trades'],
        })

        print(f"  {i+1:<6} {train_start.date()} → {train_end.date()}  "
              f"{train_end.date()} → {test_end.date()}  "
              f"{train_m['sharpe']:>11.3f}  {test_m['sharpe']:>10.3f}  "
              f"{test_m['total_return']:>8.1f}%  {test_m['max_dd']:>7.1f}%  "
              f"{test_m['n_trades']:>6}")

    # Summary
    test_sharpes = [r['test_sharpe'] for r in results]
    test_returns = [r['test_return'] for r in results]
    test_dds = [r['test_max_dd'] for r in results]

    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(f"  Windows:           {len(results)}")
    print(f"  Avg Test Sharpe:   {np.mean(test_sharpes):.3f} ± {np.std(test_sharpes):.3f}")
    print(f"  Min Test Sharpe:   {np.min(test_sharpes):.3f}")
    print(f"  Max Test Sharpe:   {np.max(test_sharpes):.3f}")
    print(f"  Avg Test Return:   {np.mean(test_returns):.1f}%")
    print(f"  Avg Test MaxDD:    {np.mean(test_dds):.1f}%")
    print(f"  Positive windows:  {sum(1 for s in test_sharpes if s > 0)}/{len(test_sharpes)}")

    # Overfitting ratio
    train_sharpes = [r['train_sharpe'] for r in results]
    if np.mean(train_sharpes) > 0:
        overfit_ratio = 1 - (np.mean(test_sharpes) / np.mean(train_sharpes))
        print(f"  Overfit ratio:     {overfit_ratio:.1%} (lower is better, <30% is good)")
    print(f"{'='*70}")

    # Save results
    output_path = '/home/ubuntu/TradingAgents/data/walk_forward_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'train_months': train_months,
                'test_months': test_months,
                'n_trials': n_trials,
                'validate_only': validate_only,
            },
            'windows': results,
            'summary': {
                'avg_test_sharpe': float(np.mean(test_sharpes)),
                'std_test_sharpe': float(np.std(test_sharpes)),
                'min_test_sharpe': float(np.min(test_sharpes)),
                'max_test_sharpe': float(np.max(test_sharpes)),
                'positive_windows': sum(1 for s in test_sharpes if s > 0),
                'total_windows': len(test_sharpes),
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"\nMETRIC: {np.mean(test_sharpes):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation')
    parser.add_argument('--train-months', type=int, default=12,
                        help='Training window size in months')
    parser.add_argument('--test-months', type=int, default=6,
                        help='Test window size in months')
    parser.add_argument('--trials', type=int, default=100,
                        help='Optuna trials per training window')
    parser.add_argument('--validate-only', action='store_true',
                        help='Test current params on all windows (no optimisation)')
    args = parser.parse_args()

    t0 = time.time()
    run_walk_forward(args.train_months, args.test_months, args.trials, args.validate_only)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
