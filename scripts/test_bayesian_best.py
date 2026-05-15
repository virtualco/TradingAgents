#!/usr/bin/env python3
"""
Test the Bayesian-optimal parameters for both BTC and ETH.
Evaluates portfolio with different weight combinations.
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')
import numpy as np
from scripts.eval_per_asset_oos import (
    load_data, _atr_vec, generate_btc_entries, generate_eth_entries,
    simulate_positions, compute_metrics,
    INITIAL_CAPITAL, WARMUP
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
print("Data loaded.\n")

# Bayesian-optimal BTC params (500 trials, Sharpe 1.39)
BAYESIAN_BTC = {
    'atr_period': 15,
    'expansion_mult': 3.7,
    'vol_mult': 2.5,
    'max_hold_bars': 12,
    'stop_mult': 1.5,
    'tp_mult': 8.0,
}

# Bayesian-optimal ETH params (500 trials, Sharpe 1.16)
BAYESIAN_ETH = {
    'donchian_period': 32,
    'adx_min': 21,
    'adx_trend': 22,
    'vol_mult': 2.0,
    'hurst_min': 0.52,
    'vol_atr_max': 0.08,
    'max_hold_bars': 90,
    'stop_mult': 3.0,
    'tp_mult': 9.0,
    'atr_donchian_factor': 1.0,
}

# Current best params (grid-search)
CURRENT_BTC = {
    'atr_period': 14,
    'expansion_mult': 3.0,
    'vol_mult': 1.2,
    'max_hold_bars': 12,
    'stop_mult': 1.5,
    'tp_mult': 4.0,
}

CURRENT_ETH = {
    'donchian_period': 28,
    'adx_min': 12,
    'adx_trend': 24,
    'vol_mult': 1.8,
    'hurst_min': 0.42,
    'vol_atr_max': 0.04,
    'max_hold_bars': 60,
    'stop_mult': 1.2,
    'tp_mult': 6.0,
    'atr_donchian_factor': None,
}


def eval_portfolio(btc_cfg, eth_cfg, btc_weight=0.5):
    eth_weight = 1.0 - btc_weight
    btc_atr = _atr_vec(btc_high, btc_low, btc_close, btc_cfg.get('atr_period', 14))
    eth_atr = _atr_vec(eth_high, eth_low, eth_close, 14)

    btc_entries = generate_btc_entries(btc_df, btc_cfg)
    btc_pnl, btc_trades = simulate_positions(
        btc_entries, btc_close, btc_high, btc_low, btc_atr,
        btc_cfg['stop_mult'], btc_cfg['tp_mult'], btc_cfg['max_hold_bars'])
    btc_m = compute_metrics(btc_pnl, btc_trades, INITIAL_CAPITAL * btc_weight)

    eth_entries = generate_eth_entries(eth_df, eth_cfg)
    eth_pnl, eth_trades = simulate_positions(
        eth_entries, eth_close, eth_high, eth_low, eth_atr,
        eth_cfg['stop_mult'], eth_cfg['tp_mult'], eth_cfg['max_hold_bars'])
    eth_m = compute_metrics(eth_pnl, eth_trades, INITIAL_CAPITAL * eth_weight)

    min_len = min(len(btc_pnl), len(eth_pnl))
    port_pnl = btc_weight * btc_pnl[:min_len] + eth_weight * eth_pnl[:min_len]
    port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_pnl)
    port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100
    sharpe = float(np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760)) if np.std(port_pnl) > 0 else 0
    peak = np.maximum.accumulate(port_equity)
    max_dd = float(np.max((peak - port_equity) / peak) * 100)
    all_trades = btc_trades + eth_trades
    n_trades = len(all_trades)
    win_rate = float(np.sum(np.array(all_trades) > 0) / n_trades * 100) if n_trades > 0 else 0

    return {
        'sharpe': sharpe, 'total_return': port_return, 'max_dd': max_dd,
        'n_trades': n_trades, 'win_rate': win_rate,
        'btc_sharpe': btc_m['sharpe'], 'eth_sharpe': eth_m['sharpe'],
    }


print("=" * 70)
print("COMPARISON: Current vs Bayesian-Optimal Parameters")
print("=" * 70)

# Current params (50/50)
curr = eval_portfolio(CURRENT_BTC, CURRENT_ETH, 0.5)
print(f"\n[CURRENT] 50/50 BTC/ETH:")
print(f"  Portfolio: Sharpe={curr['sharpe']:.3f}, Return={curr['total_return']:.1f}%, DD={curr['max_dd']:.1f}%, WR={curr['win_rate']:.1f}%, Trades={curr['n_trades']}")
print(f"  BTC: Sharpe={curr['btc_sharpe']:.3f} | ETH: Sharpe={curr['eth_sharpe']:.3f}")

# Bayesian params (50/50)
bay = eval_portfolio(BAYESIAN_BTC, BAYESIAN_ETH, 0.5)
print(f"\n[BAYESIAN] 50/50 BTC/ETH:")
print(f"  Portfolio: Sharpe={bay['sharpe']:.3f}, Return={bay['total_return']:.1f}%, DD={bay['max_dd']:.1f}%, WR={bay['win_rate']:.1f}%, Trades={bay['n_trades']}")
print(f"  BTC: Sharpe={bay['btc_sharpe']:.3f} | ETH: Sharpe={bay['eth_sharpe']:.3f}")

# Hybrid: Bayesian BTC + Current ETH
hyb1 = eval_portfolio(BAYESIAN_BTC, CURRENT_ETH, 0.5)
print(f"\n[HYBRID-1] Bayesian BTC + Current ETH (50/50):")
print(f"  Portfolio: Sharpe={hyb1['sharpe']:.3f}, Return={hyb1['total_return']:.1f}%, DD={hyb1['max_dd']:.1f}%, WR={hyb1['win_rate']:.1f}%, Trades={hyb1['n_trades']}")

# Hybrid: Current BTC + Bayesian ETH
hyb2 = eval_portfolio(CURRENT_BTC, BAYESIAN_ETH, 0.5)
print(f"\n[HYBRID-2] Current BTC + Bayesian ETH (50/50):")
print(f"  Portfolio: Sharpe={hyb2['sharpe']:.3f}, Return={hyb2['total_return']:.1f}%, DD={hyb2['max_dd']:.1f}%, WR={hyb2['win_rate']:.1f}%, Trades={hyb2['n_trades']}")

# Weight sensitivity with Bayesian params
print(f"\n{'='*70}")
print("WEIGHT SENSITIVITY (Bayesian Params)")
print(f"{'='*70}")
for bw in [0.3, 0.4, 0.5, 0.6, 0.7]:
    r = eval_portfolio(BAYESIAN_BTC, BAYESIAN_ETH, bw)
    print(f"  BTC={bw:.0%} ETH={1-bw:.0%}: Sharpe={r['sharpe']:.3f}, Return={r['total_return']:.1f}%, DD={r['max_dd']:.1f}%, Trades={r['n_trades']}")

# Best combo with 60/40
print(f"\n{'='*70}")
print("BEST CONFIGURATION: Bayesian 60/40 BTC/ETH")
print(f"{'='*70}")
best = eval_portfolio(BAYESIAN_BTC, BAYESIAN_ETH, 0.6)
print(f"  Sharpe={best['sharpe']:.3f}, Return={best['total_return']:.1f}%, DD={best['max_dd']:.1f}%, WR={best['win_rate']:.1f}%, Trades={best['n_trades']}")
print(f"  BTC: Sharpe={best['btc_sharpe']:.3f} | ETH: Sharpe={best['eth_sharpe']:.3f}")
print(f"\nMETRIC: {best['sharpe']:.6f}")
