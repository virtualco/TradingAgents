"""
Focused ETH Grid Search — Reduced parameter space for speed.
BTC best: expansion=3.0, vol=1.2, hold=12, stop=1.5, tp=4.0 (Sharpe 1.09)
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')
import numpy as np
import pandas as pd
import itertools
import time
from scripts.eval_per_asset_oos import (
    load_data, _atr_vec, _adx_vec, _hurst_fast_vec, _rolling_mean,
    generate_eth_entries, simulate_positions, compute_metrics,
    FEE, INITIAL_CAPITAL, WARMUP
)

print("Loading ETH data...")
eth_df = load_data('ETH')
eth_close = eth_df['close'].values.astype(float)
eth_high = eth_df['high'].values.astype(float)
eth_low = eth_df['low'].values.astype(float)
eth_atr = _atr_vec(eth_high, eth_low, eth_close, 14)
print("Data loaded.\n")

# Focused ETH parameter grid — key levers only
eth_params = {
    'donchian_period': [18, 20, 22, 25],
    'adx_min': [12, 15, 18],
    'adx_trend': [16, 18, 20, 22],
    'vol_mult': [1.0, 1.3, 1.5, 2.0],
    'hurst_min': [0.38, 0.42, 0.45, 0.48],
    'vol_atr_max': [None, 0.05],
    'max_hold_bars': [24, 36, 48],
    'stop_mult': [1.5, 2.0, 2.5],
    'tp_mult': [4.0, 5.0, 6.0],
    'atr_donchian_factor': [None],
}

eth_keys = list(eth_params.keys())
all_combos = list(itertools.product(*[eth_params[k] for k in eth_keys]))
print(f"Total ETH combinations: {len(all_combos)}")

# Sample 500 for speed (each takes ~1.5s due to Hurst)
MAX_COMBOS = 500
if len(all_combos) > MAX_COMBOS:
    np.random.seed(42)
    indices = np.random.choice(len(all_combos), MAX_COMBOS, replace=False)
    combos = [all_combos[i] for i in indices]
    print(f"Sampled {MAX_COMBOS} combinations")
else:
    combos = all_combos

results = []
t0 = time.time()
for i, combo in enumerate(combos):
    cfg = dict(zip(eth_keys, combo))
    
    entries = generate_eth_entries(eth_df, cfg)
    pnl, trades = simulate_positions(
        entries, eth_close, eth_high, eth_low, eth_atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])
    
    metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
    results.append({**cfg, **metrics})
    
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        best_so_far = max(r['sharpe'] for r in results)
        print(f"  {i+1}/{len(combos)} done ({elapsed:.1f}s) | Best Sharpe so far: {best_so_far:.3f}")

# Sort by Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"\n{'='*70}")
print("TOP 10 ETH CONFIGS:")
print(f"{'='*70}")
print(f"{'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'WR':>6} {'Trades':>7} | dp adx_m adx_t vol  hurst atr_max hold stop tp")
for r in results[:10]:
    atr_max_str = f"{r['vol_atr_max']}" if r['vol_atr_max'] is not None else "None"
    print(f"{r['sharpe']:>8.3f} {r['total_return']:>7.1f}% {r['max_dd']:>7.1f}% {r['win_rate']:>5.1f}% {r['n_trades']:>7} | "
          f"{r['donchian_period']:>2} {r['adx_min']:>4} {r['adx_trend']:>4} {r['vol_mult']:.1f}  {r['hurst_min']:.2f} "
          f"{atr_max_str:>5} {r['max_hold_bars']:>3}  {r['stop_mult']:.1f} {r['tp_mult']:.1f}")

# Now compute portfolio with best BTC + best ETH
print(f"\n{'='*70}")
print("PORTFOLIO: Best BTC (Sharpe 1.09) + Best ETH")
print(f"{'='*70}")

from scripts.eval_per_asset_oos import generate_btc_entries
btc_df = load_data('BTC')
btc_close = btc_df['close'].values.astype(float)
btc_high = btc_df['high'].values.astype(float)
btc_low = btc_df['low'].values.astype(float)

best_btc_cfg = {'atr_period': 14, 'expansion_mult': 3.0, 'vol_mult': 1.2,
                'max_hold_bars': 12, 'stop_mult': 1.5, 'tp_mult': 4.0}
best_eth_cfg = {k: results[0][k] for k in eth_keys}

btc_atr = _atr_vec(btc_high, btc_low, btc_close, 14)
btc_entries = generate_btc_entries(btc_df, best_btc_cfg)
btc_pnl, btc_trades = simulate_positions(
    btc_entries, btc_close, btc_high, btc_low, btc_atr,
    best_btc_cfg['stop_mult'], best_btc_cfg['tp_mult'], best_btc_cfg['max_hold_bars'])

eth_entries = generate_eth_entries(eth_df, best_eth_cfg)
eth_pnl, eth_trades = simulate_positions(
    eth_entries, eth_close, eth_high, eth_low, eth_atr,
    best_eth_cfg['stop_mult'], best_eth_cfg['tp_mult'], best_eth_cfg['max_hold_bars'])

BTC_WEIGHT = 0.5
ETH_WEIGHT = 0.5
min_len = min(len(btc_pnl), len(eth_pnl))
port_pnl = BTC_WEIGHT * btc_pnl[:min_len] + ETH_WEIGHT * eth_pnl[:min_len]
port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_pnl)
port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100
port_sharpe = np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760) if np.std(port_pnl) > 0 else 0

peak = np.maximum.accumulate(port_equity)
dd = (peak - port_equity) / peak
port_max_dd = np.max(dd) * 100

all_trades = btc_trades + eth_trades
all_arr = np.array(all_trades)
port_wr = np.sum(all_arr > 0) / len(all_arr) * 100

btc_m = compute_metrics(btc_pnl, btc_trades, INITIAL_CAPITAL * BTC_WEIGHT)
eth_m = compute_metrics(eth_pnl, eth_trades, INITIAL_CAPITAL * ETH_WEIGHT)

print(f"\nBTC: Sharpe={btc_m['sharpe']:.3f}, Return={btc_m['total_return']:.1f}%, DD={btc_m['max_dd']:.1f}%, WR={btc_m['win_rate']:.1f}%, Trades={btc_m['n_trades']}")
print(f"ETH: Sharpe={eth_m['sharpe']:.3f}, Return={eth_m['total_return']:.1f}%, DD={eth_m['max_dd']:.1f}%, WR={eth_m['win_rate']:.1f}%, Trades={eth_m['n_trades']}")
print(f"\nPORTFOLIO: Sharpe={port_sharpe:.3f}, Return={port_return:.1f}%, MaxDD={port_max_dd:.1f}%, WR={port_wr:.1f}%, Trades={len(all_trades)}")
print(f"\nBest ETH config: {best_eth_cfg}")
print(f"Best BTC config: {best_btc_cfg}")
print(f"\nMETRIC: {port_sharpe:.6f}")
