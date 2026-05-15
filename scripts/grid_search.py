"""
Grid Search — Systematic Parameter Optimisation
=================================================
Tests parameter combinations for BTC and ETH strategies independently,
then combines the best of each for portfolio-level Sharpe.
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')
import numpy as np
import pandas as pd
import itertools
import time
from scripts.eval_per_asset_oos import (
    load_data, _atr_vec, _adx_vec, _hurst_fast_vec, _rolling_mean,
    generate_btc_entries, generate_eth_entries,
    simulate_positions, compute_metrics,
    FEE, INITIAL_CAPITAL, BTC_WEIGHT, ETH_WEIGHT, WARMUP
)

# Load data once
print("Loading data...")
btc_df = load_data('BTC')
eth_df = load_data('ETH')

btc_close = btc_df['close'].values.astype(float)
btc_high = btc_df['high'].values.astype(float)
btc_low = btc_df['low'].values.astype(float)

eth_close = eth_df['close'].values.astype(float)
eth_high = eth_df['high'].values.astype(float)
eth_low = eth_df['low'].values.astype(float)

print("Data loaded. Starting grid search...\n")

# ── BTC Grid Search ──────────────────────────────────────────────────────────
btc_params = {
    'expansion_mult': [2.5, 2.8, 3.0, 3.2],
    'vol_mult': [1.0, 1.2, 1.5, 1.8],
    'max_hold_bars': [8, 12, 16, 24],
    'stop_mult': [1.5, 2.0, 2.5],
    'tp_mult': [3.0, 4.0, 5.0, 6.0],
    'atr_period': [14],
}

print("=" * 70)
print("BTC GRID SEARCH")
print("=" * 70)

btc_results = []
btc_keys = list(btc_params.keys())
btc_combos = list(itertools.product(*[btc_params[k] for k in btc_keys]))
print(f"Testing {len(btc_combos)} BTC combinations...")

t0 = time.time()
for i, combo in enumerate(btc_combos):
    cfg = dict(zip(btc_keys, combo))
    
    btc_atr = _atr_vec(btc_high, btc_low, btc_close, cfg['atr_period'])
    entries = generate_btc_entries(btc_df, cfg)
    pnl, trades = simulate_positions(
        entries, btc_close, btc_high, btc_low, btc_atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])
    
    metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
    btc_results.append({**cfg, **metrics})
    
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(btc_combos)} done ({elapsed:.1f}s)")

# Sort by Sharpe
btc_results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"\nTop 5 BTC configs:")
print(f"{'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'WR':>6} {'Trades':>7} | exp  vol  hold stop  tp")
for r in btc_results[:5]:
    print(f"{r['sharpe']:>8.3f} {r['total_return']:>7.1f}% {r['max_dd']:>7.1f}% {r['win_rate']:>5.1f}% {r['n_trades']:>7} | "
          f"{r['expansion_mult']:.1f} {r['vol_mult']:.1f}  {r['max_hold_bars']:>3}  {r['stop_mult']:.1f}  {r['tp_mult']:.1f}")

best_btc = btc_results[0]

# ── ETH Grid Search ──────────────────────────────────────────────────────────
eth_params = {
    'donchian_period': [18, 20, 22, 25, 30],
    'adx_min': [12, 15, 18],
    'adx_trend': [16, 18, 20, 22],
    'vol_mult': [1.0, 1.3, 1.5, 2.0],
    'hurst_min': [0.38, 0.42, 0.45, 0.48],
    'vol_atr_max': [None, 0.05, 0.08],
    'max_hold_bars': [24, 36, 48, 72],
    'stop_mult': [1.5, 2.0, 2.5],
    'tp_mult': [4.0, 5.0, 6.0, 7.0],
    'atr_donchian_factor': [None],  # Skip adaptive for speed
}

print("\n" + "=" * 70)
print("ETH GRID SEARCH")
print("=" * 70)

eth_keys = list(eth_params.keys())
eth_combos = list(itertools.product(*[eth_params[k] for k in eth_keys]))
print(f"Testing {len(eth_combos)} ETH combinations...")

# Sample if too many
MAX_ETH = 2000
if len(eth_combos) > MAX_ETH:
    np.random.seed(42)
    indices = np.random.choice(len(eth_combos), MAX_ETH, replace=False)
    eth_combos = [eth_combos[i] for i in indices]
    print(f"  (Sampled {MAX_ETH} from {len(list(itertools.product(*[eth_params[k] for k in eth_keys])))})")

eth_results = []
t0 = time.time()
for i, combo in enumerate(eth_combos):
    cfg = dict(zip(eth_keys, combo))
    
    eth_atr = _atr_vec(eth_high, eth_low, eth_close, 14)
    entries = generate_eth_entries(eth_df, cfg)
    pnl, trades = simulate_positions(
        entries, eth_close, eth_high, eth_low, eth_atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])
    
    metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
    eth_results.append({**cfg, **metrics})
    
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(eth_combos)} done ({elapsed:.1f}s)")

# Sort by Sharpe
eth_results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"\nTop 5 ETH configs:")
print(f"{'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'WR':>6} {'Trades':>7} | dp  adx_m adx_t vol  hurst  atr_max hold stop  tp")
for r in eth_results[:5]:
    print(f"{r['sharpe']:>8.3f} {r['total_return']:>7.1f}% {r['max_dd']:>7.1f}% {r['win_rate']:>5.1f}% {r['n_trades']:>7} | "
          f"{r['donchian_period']:>2} {r['adx_min']:>4} {r['adx_trend']:>4} {r['vol_mult']:.1f}  {r['hurst_min']:.2f}  "
          f"{str(r['vol_atr_max']):>5} {r['max_hold_bars']:>3}  {r['stop_mult']:.1f}  {r['tp_mult']:.1f}")

best_eth = eth_results[0]

# ── Portfolio Combination ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PORTFOLIO COMBINATION (Best BTC + Best ETH)")
print("=" * 70)

# Recalculate with best params
btc_cfg_best = {k: best_btc[k] for k in btc_keys}
eth_cfg_best = {k: best_eth[k] for k in eth_keys}

btc_atr = _atr_vec(btc_high, btc_low, btc_close, btc_cfg_best['atr_period'])
btc_entries = generate_btc_entries(btc_df, btc_cfg_best)
btc_pnl, btc_trades = simulate_positions(
    btc_entries, btc_close, btc_high, btc_low, btc_atr,
    btc_cfg_best['stop_mult'], btc_cfg_best['tp_mult'], btc_cfg_best['max_hold_bars'])

eth_atr = _atr_vec(eth_high, eth_low, eth_close, 14)
eth_entries = generate_eth_entries(eth_df, eth_cfg_best)
eth_pnl, eth_trades = simulate_positions(
    eth_entries, eth_close, eth_high, eth_low, eth_atr,
    eth_cfg_best['stop_mult'], eth_cfg_best['tp_mult'], eth_cfg_best['max_hold_bars'])

# Portfolio
min_len = min(len(btc_pnl), len(eth_pnl))
port_pnl = BTC_WEIGHT * btc_pnl[:min_len] + ETH_WEIGHT * eth_pnl[:min_len]
port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_pnl)
port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100
port_sharpe = np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760) if np.std(port_pnl) > 0 else 0

btc_m = compute_metrics(btc_pnl, btc_trades, INITIAL_CAPITAL * BTC_WEIGHT)
eth_m = compute_metrics(eth_pnl, eth_trades, INITIAL_CAPITAL * ETH_WEIGHT)

print(f"\nBest BTC config: {btc_cfg_best}")
print(f"  Sharpe={btc_m['sharpe']:.3f}, Return={btc_m['total_return']:.1f}%, DD={btc_m['max_dd']:.1f}%, WR={btc_m['win_rate']:.1f}%, Trades={btc_m['n_trades']}")
print(f"\nBest ETH config: {eth_cfg_best}")
print(f"  Sharpe={eth_m['sharpe']:.3f}, Return={eth_m['total_return']:.1f}%, DD={eth_m['max_dd']:.1f}%, WR={eth_m['win_rate']:.1f}%, Trades={eth_m['n_trades']}")
print(f"\nPORTFOLIO: Sharpe={port_sharpe:.3f}, Return={port_return:.1f}%")
print(f"\nMETRIC: {port_sharpe:.6f}")
