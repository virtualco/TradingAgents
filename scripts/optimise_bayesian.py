#!/usr/bin/env python3
"""
Bayesian Optimisation — Optuna TPE Sampler
============================================
Sample-efficient parameter search using Tree-structured Parzen Estimator.
Explores the BTC/ETH parameter space intelligently, focusing on promising regions.

Usage:
  python3 scripts/optimise_bayesian.py [--trials 200] [--timeout 600] [--asset both|btc|eth]
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')
import argparse
import json
import time
import numpy as np
import optuna
from scripts.eval_per_asset_oos import (
    load_data, _atr_vec, generate_btc_entries, generate_eth_entries,
    simulate_positions, compute_metrics,
    INITIAL_CAPITAL, BTC_WEIGHT, ETH_WEIGHT
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Data (loaded once) ────────────────────────────────────────────────────────
print("Loading data...")
btc_df = load_data('BTC')
eth_df = load_data('ETH')
btc_close = btc_df['close'].values.astype(float)
btc_high  = btc_df['high'].values.astype(float)
btc_low   = btc_df['low'].values.astype(float)
eth_close = eth_df['close'].values.astype(float)
eth_high  = eth_df['high'].values.astype(float)
eth_low   = eth_df['low'].values.astype(float)
btc_atr   = _atr_vec(btc_high, btc_low, btc_close, 14)
eth_atr   = _atr_vec(eth_high, eth_low, eth_close, 14)
print("Data loaded.\n")


def eval_btc(cfg: dict) -> dict:
    """Evaluate BTC strategy with given config, return metrics dict."""
    entries = generate_btc_entries(btc_df, cfg)
    pnl, trades = simulate_positions(
        entries, btc_close, btc_high, btc_low, btc_atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])
    return compute_metrics(pnl, trades, INITIAL_CAPITAL * BTC_WEIGHT)


def eval_eth(cfg: dict) -> dict:
    """Evaluate ETH strategy with given config, return metrics dict."""
    entries = generate_eth_entries(eth_df, cfg)
    pnl, trades = simulate_positions(
        entries, eth_close, eth_high, eth_low, eth_atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])
    return compute_metrics(pnl, trades, INITIAL_CAPITAL * ETH_WEIGHT)


def eval_portfolio(btc_cfg: dict, eth_cfg: dict) -> float:
    """Evaluate combined portfolio Sharpe."""
    btc_entries = generate_btc_entries(btc_df, btc_cfg)
    btc_pnl, btc_trades = simulate_positions(
        btc_entries, btc_close, btc_high, btc_low, btc_atr,
        btc_cfg['stop_mult'], btc_cfg['tp_mult'], btc_cfg['max_hold_bars'])

    eth_entries = generate_eth_entries(eth_df, eth_cfg)
    eth_pnl, eth_trades = simulate_positions(
        eth_entries, eth_close, eth_high, eth_low, eth_atr,
        eth_cfg['stop_mult'], eth_cfg['tp_mult'], eth_cfg['max_hold_bars'])

    min_len = min(len(btc_pnl), len(eth_pnl))
    port_pnl = BTC_WEIGHT * btc_pnl[:min_len] + ETH_WEIGHT * eth_pnl[:min_len]
    if np.std(port_pnl) > 0:
        return float(np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760))
    return 0.0


# ── Optuna Objectives ─────────────────────────────────────────────────────────
def btc_objective(trial: optuna.Trial) -> float:
    cfg = {
        'atr_period':     trial.suggest_int('atr_period', 10, 20),
        'expansion_mult': trial.suggest_float('expansion_mult', 2.0, 4.0, step=0.1),
        'vol_mult':       trial.suggest_float('vol_mult', 0.8, 2.5, step=0.1),
        'max_hold_bars':  trial.suggest_int('max_hold_bars', 6, 36, step=2),
        'stop_mult':      trial.suggest_float('stop_mult', 0.8, 3.0, step=0.1),
        'tp_mult':        trial.suggest_float('tp_mult', 2.0, 8.0, step=0.5),
    }
    m = eval_btc(cfg)
    # Multi-objective: maximise Sharpe, penalise extreme drawdown
    sharpe = m['sharpe']
    if m['max_dd'] > 40:
        sharpe -= 0.2 * (m['max_dd'] - 40) / 10
    if m['n_trades'] < 100:
        sharpe -= 0.5  # penalise too few trades
    return sharpe


def eth_objective(trial: optuna.Trial) -> float:
    use_atr_max = trial.suggest_categorical('use_atr_max', [True, False])
    use_adaptive = trial.suggest_categorical('use_adaptive', [True, False])

    cfg = {
        'donchian_period':    trial.suggest_int('donchian_period', 15, 35),
        'adx_min':            trial.suggest_int('adx_min', 8, 22),
        'adx_trend':          trial.suggest_int('adx_trend', 14, 28),
        'vol_mult':           trial.suggest_float('vol_mult', 0.8, 3.0, step=0.1),
        'hurst_min':          trial.suggest_float('hurst_min', 0.30, 0.55, step=0.02),
        'vol_atr_max':        trial.suggest_float('vol_atr_max', 0.02, 0.10, step=0.01) if use_atr_max else None,
        'max_hold_bars':      trial.suggest_int('max_hold_bars', 12, 96, step=6),
        'stop_mult':          trial.suggest_float('stop_mult', 0.8, 3.5, step=0.1),
        'tp_mult':            trial.suggest_float('tp_mult', 3.0, 10.0, step=0.5),
        'atr_donchian_factor': trial.suggest_float('atr_donchian_factor', 0.1, 1.0, step=0.1) if use_adaptive else None,
    }
    m = eval_eth(cfg)
    sharpe = m['sharpe']
    if m['max_dd'] > 50:
        sharpe -= 0.2 * (m['max_dd'] - 50) / 10
    if m['n_trades'] < 100:
        sharpe -= 0.5
    return sharpe


def portfolio_objective(trial: optuna.Trial) -> float:
    """Joint optimisation of both assets simultaneously."""
    btc_cfg = {
        'atr_period':     trial.suggest_int('btc_atr_period', 10, 20),
        'expansion_mult': trial.suggest_float('btc_expansion_mult', 2.0, 4.0, step=0.1),
        'vol_mult':       trial.suggest_float('btc_vol_mult', 0.8, 2.5, step=0.1),
        'max_hold_bars':  trial.suggest_int('btc_max_hold_bars', 6, 36, step=2),
        'stop_mult':      trial.suggest_float('btc_stop_mult', 0.8, 3.0, step=0.1),
        'tp_mult':        trial.suggest_float('btc_tp_mult', 2.0, 8.0, step=0.5),
    }
    use_atr_max = trial.suggest_categorical('eth_use_atr_max', [True, False])
    use_adaptive = trial.suggest_categorical('eth_use_adaptive', [True, False])
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
        'atr_donchian_factor': trial.suggest_float('eth_atr_donchian_factor', 0.1, 1.0, step=0.1) if use_adaptive else None,
    }
    return eval_portfolio(btc_cfg, eth_cfg)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Bayesian optimisation for trading strategy')
    parser.add_argument('--trials', type=int, default=200, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Max seconds for study')
    parser.add_argument('--asset', choices=['btc', 'eth', 'both', 'portfolio'], default='portfolio',
                        help='Which asset(s) to optimise')
    args = parser.parse_args()

    t0 = time.time()

    if args.asset == 'btc':
        study = optuna.create_study(direction='maximize', study_name='btc_optimisation',
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(btc_objective, n_trials=args.trials, timeout=args.timeout)
        best = study.best_trial
        print(f"\n{'='*70}")
        print(f"BEST BTC: Sharpe={best.value:.3f}")
        print(f"Params: {json.dumps(best.params, indent=2)}")

    elif args.asset == 'eth':
        study = optuna.create_study(direction='maximize', study_name='eth_optimisation',
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(eth_objective, n_trials=args.trials, timeout=args.timeout)
        best = study.best_trial
        print(f"\n{'='*70}")
        print(f"BEST ETH: Sharpe={best.value:.3f}")
        print(f"Params: {json.dumps(best.params, indent=2)}")

    elif args.asset == 'both':
        # Optimise each independently then combine
        print("Phase 1: Optimising BTC...")
        btc_study = optuna.create_study(direction='maximize', study_name='btc',
                                         sampler=optuna.samplers.TPESampler(seed=42))
        btc_study.optimize(btc_objective, n_trials=args.trials // 2, timeout=args.timeout)
        btc_best = btc_study.best_trial
        print(f"  BTC best: Sharpe={btc_best.value:.3f}")

        print("\nPhase 2: Optimising ETH...")
        eth_study = optuna.create_study(direction='maximize', study_name='eth',
                                         sampler=optuna.samplers.TPESampler(seed=42))
        eth_study.optimize(eth_objective, n_trials=args.trials // 2, timeout=args.timeout)
        eth_best = eth_study.best_trial
        print(f"  ETH best: Sharpe={eth_best.value:.3f}")

        # Combine
        btc_cfg = {k: v for k, v in btc_best.params.items()}
        eth_params = {k: v for k, v in eth_best.params.items()}
        eth_cfg = {}
        for k, v in eth_params.items():
            if k == 'use_atr_max':
                continue
            elif k == 'use_adaptive':
                continue
            elif k == 'vol_atr_max' and not eth_params.get('use_atr_max', True):
                eth_cfg['vol_atr_max'] = None
            elif k == 'atr_donchian_factor' and not eth_params.get('use_adaptive', False):
                eth_cfg['atr_donchian_factor'] = None
            else:
                eth_cfg[k] = v
        if 'use_atr_max' in eth_params and not eth_params['use_atr_max']:
            eth_cfg['vol_atr_max'] = None
        if 'use_adaptive' in eth_params and not eth_params['use_adaptive']:
            eth_cfg['atr_donchian_factor'] = None

        port_sharpe = eval_portfolio(btc_cfg, eth_cfg)
        print(f"\n{'='*70}")
        print(f"PORTFOLIO Sharpe: {port_sharpe:.3f}")
        print(f"BTC config: {json.dumps(btc_cfg, indent=2)}")
        print(f"ETH config: {json.dumps(eth_cfg, indent=2)}")

    else:  # portfolio — joint optimisation
        study = optuna.create_study(direction='maximize', study_name='portfolio_joint',
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(portfolio_objective, n_trials=args.trials, timeout=args.timeout)
        best = study.best_trial
        print(f"\n{'='*70}")
        print(f"BEST PORTFOLIO: Sharpe={best.value:.3f}")
        print(f"Params: {json.dumps(best.params, indent=2)}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"METRIC: {study.best_value:.6f}" if 'study' in dir() else "")


if __name__ == '__main__':
    main()
