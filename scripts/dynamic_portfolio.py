#!/usr/bin/env python3
"""
Dynamic Portfolio: Monthly Rebalancing + Regime-Aware Allocation
================================================================
Implements:
1. Monthly rolling risk-parity rebalancing (not static full-period weights)
2. Regime detection (trending vs ranging) per asset
3. Regime-aware allocation: increase weight for trending assets, reduce for ranging

Usage:
    python3 scripts/dynamic_portfolio.py                    # All 11 assets
    python3 scripts/dynamic_portfolio.py --symbols BTCUSDT ETHUSDT SPY
    python3 scripts/dynamic_portfolio.py --rebalance-days 30
    python3 scripts/dynamic_portfolio.py --no-regime        # Disable regime filter
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import numpy as np
import pandas as pd
import time
import argparse
import importlib

from scripts.eval_multi_asset import (
    load_data, eval_single_asset, get_bars_per_year, get_warmup,
    generate_atr_expansion_entries, generate_donchian_entries,
    simulate_positions, _atr_vec, FEE, INITIAL_CAPITAL, WARMUP
)


# ── Regime Detection ─────────────────────────────────────────────────────────

def detect_regime(returns: np.ndarray, lookback: int = 60) -> float:
    """
    Classify recent returns as trending (>0.5) or ranging (<0.5) using:
    - Efficiency ratio: |cumulative return| / sum(|returns|)
    - Rolling Sharpe sign: positive = trending, negative = ranging
    - Hurst proxy: variance ratio test
    
    Returns single regime score [0, 1] where 1 = strong trend.
    """
    if len(returns) < lookback:
        return 0.5
    
    window = returns[-lookback:]
    if np.std(window) < 1e-10:
        return 0.3  # flat = ranging
    
    # 1. Efficiency ratio: |net move| / sum(|moves|)
    cum_ret = np.cumsum(window)
    net_move = abs(cum_ret[-1])
    gross_move = np.sum(np.abs(window))
    efficiency = net_move / gross_move if gross_move > 0 else 0
    
    # 2. Variance ratio (Hurst proxy): var(2-period returns) / (2 * var(1-period returns))
    # VR > 1 = trending, VR < 1 = mean-reverting
    var1 = np.var(window)
    ret2 = window[:-1] + window[1:]  # 2-period returns
    var2 = np.var(ret2)
    vr = var2 / (2 * var1) if var1 > 1e-12 else 1.0
    vr_score = np.clip((vr - 0.5) / 1.0, 0, 1)  # normalise: 0.5→0, 1.5→1
    
    # 3. Rolling Sharpe direction
    sharpe_sign = 1.0 if np.mean(window) > 0 else 0.3
    
    # Combine: efficiency (40%) + variance ratio (40%) + direction bonus (20%)
    score = 0.4 * efficiency + 0.4 * vr_score + 0.2 * sharpe_sign
    return float(np.clip(score, 0, 1))


MAX_SINGLE_WEIGHT = 0.25  # Cap any single asset at 25%
MIN_SINGLE_WEIGHT = 0.02  # Floor at 2% for diversification


def compute_regime_adjusted_weights(
    daily_returns: dict,
    lookback_vol: int = 120,
    lookback_regime: int = 90,
    regime_boost: float = 1.5,
    regime_dampen: float = 0.5,
    trending_threshold: float = 0.50,
    ranging_threshold: float = 0.35,
) -> dict:
    """
    Compute risk-parity weights adjusted by regime:
    - Trending assets get weight * regime_boost
    - Ranging assets get weight * regime_dampen
    - Neutral assets keep base weight
    - Max single asset capped at 25%, min at 2%
    """
    # Base risk-parity weights
    vols = {}
    regimes = {}
    for sym, ret in daily_returns.items():
        recent = ret[-lookback_vol:] if len(ret) > lookback_vol else ret
        vol = float(np.std(recent))
        vols[sym] = max(vol, 1e-8)
        
        # Regime score for this asset
        regime_data = ret[-lookback_regime:] if len(ret) > lookback_regime else ret
        regimes[sym] = detect_regime(regime_data, min(lookback_regime, len(regime_data)))
    
    inv_vols = {sym: 1.0 / v for sym, v in vols.items()}
    total_iv = sum(inv_vols.values())
    base_weights = {sym: iv / total_iv for sym, iv in inv_vols.items()}
    
    # Apply regime adjustment
    adjusted = {}
    for sym, w in base_weights.items():
        r = regimes[sym]
        if r >= trending_threshold:
            adjusted[sym] = w * regime_boost
        elif r <= ranging_threshold:
            adjusted[sym] = w * regime_dampen
        else:
            adjusted[sym] = w
    
    # Apply caps and floors
    n_assets = len(adjusted)
    for sym in adjusted:
        adjusted[sym] = max(adjusted[sym], MIN_SINGLE_WEIGHT)
    
    # Re-normalise
    total_adj = sum(adjusted.values())
    adjusted = {sym: w / total_adj for sym, w in adjusted.items()}
    
    # Cap at MAX_SINGLE_WEIGHT, redistribute excess
    for _ in range(5):  # iterate to converge
        excess = 0
        n_uncapped = 0
        for sym, w in adjusted.items():
            if w > MAX_SINGLE_WEIGHT:
                excess += w - MAX_SINGLE_WEIGHT
                adjusted[sym] = MAX_SINGLE_WEIGHT
            else:
                n_uncapped += 1
        if excess > 0 and n_uncapped > 0:
            per_asset = excess / n_uncapped
            for sym in adjusted:
                if adjusted[sym] < MAX_SINGLE_WEIGHT:
                    adjusted[sym] += per_asset
    
    # Final normalise
    total_adj = sum(adjusted.values())
    return {sym: w / total_adj for sym, w in adjusted.items()}, regimes


# ── Dynamic Portfolio Simulation ─────────────────────────────────────────────

def simulate_dynamic_portfolio(
    asset_pnl: dict,
    asset_configs: dict,
    rebalance_days: int = 30,
    use_regime: bool = True,
    lookback_vol: int = 60,
    lookback_regime: int = 60,
):
    """
    Simulate portfolio with monthly rebalancing and optional regime-aware allocation.
    
    Returns:
        portfolio_equity: daily equity curve
        weight_history: list of (day_idx, weights_dict)
        regime_history: list of (day_idx, regimes_dict)
    """
    # Convert all PnL to daily returns
    daily_returns = {}
    for sym, pnl in asset_pnl.items():
        bpy = get_bars_per_year(sym, asset_configs.get(sym))
        equity = INITIAL_CAPITAL * np.cumprod(1 + pnl)
        if bpy == 8760:  # hourly → daily
            n_days = len(equity) // 24
            if n_days > 0:
                daily_eq = equity[23::24][:n_days]
                daily_ret = np.diff(daily_eq) / daily_eq[:-1]
                daily_returns[sym] = daily_ret
        else:  # already daily
            daily_ret = np.diff(equity) / equity[:-1]
            daily_returns[sym] = daily_ret
    
    min_len = min(len(r) for r in daily_returns.values())
    n_days = min_len
    
    # Initialise
    portfolio_equity = np.zeros(n_days)
    portfolio_equity[0] = INITIAL_CAPITAL
    weight_history = []
    regime_history = []
    
    # Initial weights (equal until we have enough data)
    current_weights = {sym: 1.0 / len(daily_returns) for sym in daily_returns}
    weight_history.append((0, dict(current_weights)))
    
    for day in range(1, n_days):
        # Rebalance?
        if day % rebalance_days == 0 and day >= lookback_vol:
            # Build lookback window for each asset
            lookback_data = {
                sym: ret[max(0, day - lookback_vol):day]
                for sym, ret in daily_returns.items()
            }
            
            if use_regime:
                current_weights, regimes = compute_regime_adjusted_weights(
                    lookback_data, lookback_vol, lookback_regime
                )
                regime_history.append((day, dict(regimes)))
            else:
                # Pure risk-parity
                vols = {}
                for sym, ret in lookback_data.items():
                    vols[sym] = max(float(np.std(ret)), 1e-8)
                inv_vols = {sym: 1.0 / v for sym, v in vols.items()}
                total = sum(inv_vols.values())
                current_weights = {sym: iv / total for sym, iv in inv_vols.items()}
            
            weight_history.append((day, dict(current_weights)))
        
        # Compute weighted daily return
        port_ret = 0.0
        for sym, ret in daily_returns.items():
            if day < len(ret):
                port_ret += current_weights.get(sym, 0) * ret[day]
        
        portfolio_equity[day] = portfolio_equity[day - 1] * (1 + port_ret)
    
    return portfolio_equity, weight_history, regime_history, daily_returns


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Dynamic Portfolio with Regime-Aware Allocation')
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--rebalance-days', type=int, default=30)
    parser.add_argument('--no-regime', action='store_true')
    parser.add_argument('--compare', action='store_true', help='Compare static vs dynamic')
    args = parser.parse_args()
    
    t0 = time.time()
    
    mod = importlib.import_module('tradingagents.research.per_asset_router')
    importlib.reload(mod)
    ASSET_CONFIG = mod.ASSET_CONFIG
    
    symbols = args.symbols or list(ASSET_CONFIG.keys())
    symbols = [s.upper() for s in symbols]
    
    print("=" * 80)
    print(f"DYNAMIC PORTFOLIO SIMULATION")
    print(f"Assets: {len(symbols)} | Rebalance: every {args.rebalance_days} days | "
          f"Regime: {'OFF' if args.no_regime else 'ON'}")
    print("=" * 80)
    
    # Evaluate each asset
    asset_pnl = {}
    asset_metrics = {}
    for sym in symbols:
        cfg = ASSET_CONFIG.get(sym)
        if cfg is None:
            print(f"  WARNING: {sym} not in ASSET_CONFIG, skipping")
            continue
        try:
            metrics = eval_single_asset(sym, cfg)
            asset_pnl[sym] = metrics['pnl']
            asset_metrics[sym] = metrics
        except Exception as e:
            print(f"  ERROR: {sym}: {e}")
    
    # Print per-asset results
    print(f"\n{'Asset':10s} {'Strategy':20s} {'Sharpe':>8s} {'Return':>10s} {'MaxDD':>8s} "
          f"{'WinRate':>8s} {'Trades':>8s} {'PF':>6s}")
    print("-" * 80)
    for sym in symbols:
        if sym not in asset_metrics:
            continue
        m = asset_metrics[sym]
        cfg = ASSET_CONFIG[sym]
        print(f"{sym:10s} {cfg['strategy']:20s} {m['sharpe']:8.3f} {m['total_return']:9.1f}% "
              f"{m['max_dd']:7.1f}% {m['win_rate']:7.1f}% {m['n_trades']:8d} {m['pf']:5.2f}")
    
    # Dynamic portfolio
    use_regime = not args.no_regime
    equity, weights_hist, regime_hist, daily_rets = simulate_dynamic_portfolio(
        asset_pnl, ASSET_CONFIG,
        rebalance_days=args.rebalance_days,
        use_regime=use_regime,
    )
    
    total_return = (equity[-1] / INITIAL_CAPITAL - 1) * 100
    daily_ret = np.diff(equity) / equity[:-1]
    sharpe = float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if np.std(daily_ret) > 0 else 0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak) * 100)
    total_trades = sum(m['n_trades'] for m in asset_metrics.values())
    
    print(f"\n{'═' * 80}")
    print(f"DYNAMIC PORTFOLIO ({'Regime-Aware' if use_regime else 'Risk-Parity'}, "
          f"{args.rebalance_days}-day rebalance)")
    print(f"{'═' * 80}")
    print(f"  Sharpe:       {sharpe:.3f}")
    print(f"  Total Return: {total_return:.1f}%")
    print(f"  Max Drawdown: {max_dd:.1f}%")
    print(f"  Total Trades: {total_trades}")
    print(f"  Rebalances:   {len(weights_hist)}")
    
    # Show latest weights
    if weights_hist:
        _, latest_w = weights_hist[-1]
        print(f"\n  Latest Weights:")
        for sym, w in sorted(latest_w.items(), key=lambda x: -x[1]):
            regime_str = ""
            if regime_hist:
                _, latest_r = regime_hist[-1]
                r = latest_r.get(sym, 0.5)
                label = "TREND" if r >= 0.55 else ("RANGE" if r <= 0.40 else "NEUTRAL")
                regime_str = f" [{label} {r:.2f}]"
            print(f"    {sym:10s}: {w*100:5.1f}%{regime_str}")
    
    # Compare with static if requested
    if args.compare:
        print(f"\n{'─' * 80}")
        print("COMPARISON: Static vs Dynamic")
        print(f"{'─' * 80}")
        
        # Static risk-parity (no regime)
        equity_static, _, _, _ = simulate_dynamic_portfolio(
            asset_pnl, ASSET_CONFIG,
            rebalance_days=999999,  # never rebalance
            use_regime=False,
        )
        static_ret = (equity_static[-1] / INITIAL_CAPITAL - 1) * 100
        static_dr = np.diff(equity_static) / equity_static[:-1]
        static_sharpe = float(np.mean(static_dr) / np.std(static_dr) * np.sqrt(252)) if np.std(static_dr) > 0 else 0
        static_peak = np.maximum.accumulate(equity_static)
        static_dd = float(np.max((static_peak - equity_static) / static_peak) * 100)
        
        # Dynamic risk-parity (no regime)
        equity_dyn_nore, _, _, _ = simulate_dynamic_portfolio(
            asset_pnl, ASSET_CONFIG,
            rebalance_days=args.rebalance_days,
            use_regime=False,
        )
        dyn_nore_ret = (equity_dyn_nore[-1] / INITIAL_CAPITAL - 1) * 100
        dyn_nore_dr = np.diff(equity_dyn_nore) / equity_dyn_nore[:-1]
        dyn_nore_sharpe = float(np.mean(dyn_nore_dr) / np.std(dyn_nore_dr) * np.sqrt(252)) if np.std(dyn_nore_dr) > 0 else 0
        dyn_nore_peak = np.maximum.accumulate(equity_dyn_nore)
        dyn_nore_dd = float(np.max((dyn_nore_peak - equity_dyn_nore) / dyn_nore_peak) * 100)
        
        print(f"  {'Mode':30s} {'Sharpe':>8s} {'Return':>10s} {'MaxDD':>8s}")
        print(f"  {'-'*60}")
        print(f"  {'Static Risk-Parity':30s} {static_sharpe:8.3f} {static_ret:9.1f}% {static_dd:7.1f}%")
        print(f"  {'Dynamic Risk-Parity':30s} {dyn_nore_sharpe:8.3f} {dyn_nore_ret:9.1f}% {dyn_nore_dd:7.1f}%")
        print(f"  {'Dynamic + Regime':30s} {sharpe:8.3f} {total_return:9.1f}% {max_dd:7.1f}%")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Eval time: {elapsed:.1f}s")
    print(f"METRIC: {sharpe:.6f}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
