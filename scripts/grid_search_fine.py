"""
Fine-tuning Grid Search — Narrow ranges around best parameters.
Best so far: Portfolio Sharpe 1.05
BTC: exp=3.0, vol=1.2, hold=12, stop=1.5, tp=4.0 (Sharpe 1.09)
ETH: dp=25, adx_min=15, adx_trend=22, vol=2.0, hurst=0.42, atr_max=0.05, hold=48, stop=1.5, tp=6.0 (Sharpe 0.62)
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')
import numpy as np
import pandas as pd
import itertools
import time
from scripts.eval_per_asset_oos import (
    load_data, _atr_vec, generate_btc_entries, generate_eth_entries,
    simulate_positions, compute_metrics,
    INITIAL_CAPITAL, BTC_WEIGHT, ETH_WEIGHT
)

print("Loading data...")
btc_df = load_data('BTC')
eth_df = load_data('ETH')
btc_close = btc_df['close'].values.astype(float)
btc_high = btc_df['high'].values.astype(float)
btc_low = btc_df['low'].values.astype(float)
eth_close = eth_df['close'].values.astype(float)
eth_high = eth_df['high'].values.astype(float)
eth_low = eth_df['low'].values.astype(float)
btc_atr = _atr_vec(btc_high, btc_low, btc_close, 14)
eth_atr = _atr_vec(eth_high, eth_low, eth_close, 14)
print("Data loaded.\n")

# Fine-tune BTC around best
btc_fine = {
    'atr_period': [14],
    'expansion_mult': [2.8, 3.0, 3.2],
    'vol_mult': [1.0, 1.1, 1.2, 1.3],
    'max_hold_bars': [10, 12, 14, 16],
    'stop_mult': [1.2, 1.5, 1.8],
    'tp_mult': [3.5, 4.0, 4.5, 5.0],
}

# Fine-tune ETH around best
eth_fine = {
    'donchian_period': [22, 25, 28],
    'adx_min': [12, 15, 18],
    'adx_trend': [20, 22, 24],
    'vol_mult': [1.5, 1.8, 2.0, 2.2],
    'hurst_min': [0.38, 0.40, 0.42, 0.44],
    'vol_atr_max': [0.04, 0.05, 0.06, None],
    'max_hold_bars': [36, 48, 60],
    'stop_mult': [1.2, 1.5, 1.8],
    'tp_mult': [5.0, 6.0, 7.0, 8.0],
    'atr_donchian_factor': [None],
}

# BTC fine-tune
print("=" * 70)
print("BTC FINE-TUNE")
print("=" * 70)
btc_keys = list(btc_fine.keys())
btc_combos = list(itertools.product(*[btc_fine[k] for k in btc_keys]))
print(f"Testing {len(btc_combos)} BTC combinations...")

btc_results = []
t0 = time.time()
for i, combo in enumerate(btc_combos):
    cfg = dict(zip(btc_keys, combo))
    entries = generate_btc_entries(btc_df, cfg)
    pnl, trades = simulate_positions(
        entries, btc_close, btc_high, btc_low, btc_atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])
    metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
    btc_results.append({**cfg, **metrics})

btc_results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"Done in {time.time()-t0:.1f}s")
print(f"\nTop 5 BTC:")
for r in btc_results[:5]:
    print(f"  Sharpe={r['sharpe']:.3f} Ret={r['total_return']:.1f}% DD={r['max_dd']:.1f}% WR={r['win_rate']:.1f}% T={r['n_trades']} | "
          f"exp={r['expansion_mult']} vol={r['vol_mult']} hold={r['max_hold_bars']} stop={r['stop_mult']} tp={r['tp_mult']}")

# ETH fine-tune (sample 300 for speed)
print("\n" + "=" * 70)
print("ETH FINE-TUNE")
print("=" * 70)
eth_keys = list(eth_fine.keys())
all_eth = list(itertools.product(*[eth_fine[k] for k in eth_keys]))
print(f"Total ETH: {len(all_eth)}")
MAX = 300
if len(all_eth) > MAX:
    np.random.seed(123)
    indices = np.random.choice(len(all_eth), MAX, replace=False)
    eth_combos = [all_eth[i] for i in indices]
    print(f"Sampled {MAX}")
else:
    eth_combos = all_eth

eth_results = []
t0 = time.time()
for i, combo in enumerate(eth_combos):
    cfg = dict(zip(eth_keys, combo))
    entries = generate_eth_entries(eth_df, cfg)
    pnl, trades = simulate_positions(
        entries, eth_close, eth_high, eth_low, eth_atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])
    metrics = compute_metrics(pnl, trades, INITIAL_CAPITAL)
    eth_results.append({**cfg, **metrics})
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(eth_combos)} ({time.time()-t0:.0f}s) best={max(r['sharpe'] for r in eth_results):.3f}")

eth_results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"\nTop 5 ETH:")
for r in eth_results[:5]:
    atr_max_str = str(r['vol_atr_max']) if r['vol_atr_max'] is not None else "None"
    print(f"  Sharpe={r['sharpe']:.3f} Ret={r['total_return']:.1f}% DD={r['max_dd']:.1f}% WR={r['win_rate']:.1f}% T={r['n_trades']} | "
          f"dp={r['donchian_period']} adx_m={r['adx_min']} adx_t={r['adx_trend']} vol={r['vol_mult']} "
          f"hurst={r['hurst_min']} atr_max={atr_max_str} hold={r['max_hold_bars']} stop={r['stop_mult']} tp={r['tp_mult']}")

# Portfolio
print("\n" + "=" * 70)
print("BEST PORTFOLIO")
print("=" * 70)

best_btc = {k: btc_results[0][k] for k in btc_keys}
best_eth = {k: eth_results[0][k] for k in eth_keys}

btc_entries = generate_btc_entries(btc_df, best_btc)
btc_pnl, btc_trades = simulate_positions(
    btc_entries, btc_close, btc_high, btc_low, btc_atr,
    best_btc['stop_mult'], best_btc['tp_mult'], best_btc['max_hold_bars'])

eth_entries = generate_eth_entries(eth_df, best_eth)
eth_pnl, eth_trades = simulate_positions(
    eth_entries, eth_close, eth_high, eth_low, eth_atr,
    best_eth['stop_mult'], best_eth['tp_mult'], best_eth['max_hold_bars'])

min_len = min(len(btc_pnl), len(eth_pnl))
port_pnl = BTC_WEIGHT * btc_pnl[:min_len] + ETH_WEIGHT * eth_pnl[:min_len]
port_sharpe = np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760) if np.std(port_pnl) > 0 else 0
port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_pnl)
port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100
peak = np.maximum.accumulate(port_equity)
port_max_dd = np.max((peak - port_equity) / peak) * 100

btc_m = compute_metrics(btc_pnl, btc_trades, INITIAL_CAPITAL * BTC_WEIGHT)
eth_m = compute_metrics(eth_pnl, eth_trades, INITIAL_CAPITAL * ETH_WEIGHT)

print(f"\nBTC: Sharpe={btc_m['sharpe']:.3f}, Return={btc_m['total_return']:.1f}%, DD={btc_m['max_dd']:.1f}%, WR={btc_m['win_rate']:.1f}%, Trades={btc_m['n_trades']}")
print(f"ETH: Sharpe={eth_m['sharpe']:.3f}, Return={eth_m['total_return']:.1f}%, DD={eth_m['max_dd']:.1f}%, WR={eth_m['win_rate']:.1f}%, Trades={eth_m['n_trades']}")
print(f"\nPORTFOLIO: Sharpe={port_sharpe:.3f}, Return={port_return:.1f}%, MaxDD={port_max_dd:.1f}%")
print(f"\nBest BTC config: {best_btc}")
print(f"Best ETH config: {best_eth}")
print(f"\nMETRIC: {port_sharpe:.6f}")

# Also test with different BTC/ETH weights
print("\n" + "=" * 70)
print("WEIGHT SENSITIVITY")
print("=" * 70)
for bw in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ew = 1.0 - bw
    p = bw * btc_pnl[:min_len] + ew * eth_pnl[:min_len]
    s = np.mean(p) / np.std(p) * np.sqrt(8760) if np.std(p) > 0 else 0
    eq = INITIAL_CAPITAL * np.cumprod(1 + p)
    ret = (eq[-1] / INITIAL_CAPITAL - 1) * 100
    pk = np.maximum.accumulate(eq)
    dd = np.max((pk - eq) / pk) * 100
    print(f"  BTC={bw:.0%} ETH={ew:.0%}: Sharpe={s:.3f}, Return={ret:.1f}%, MaxDD={dd:.1f}%")
