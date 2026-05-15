#!/usr/bin/env python3
"""
Multi-Asset OOS Evaluation with Risk-Parity Weighting
======================================================
Evaluates all crypto assets from per_asset_router.py and computes
a risk-parity weighted portfolio Sharpe ratio.

Signal generation logic is identical to eval_per_asset_oos.py (the proven harness).

Usage:
    python3 scripts/eval_multi_asset.py
    python3 scripts/eval_multi_asset.py --symbols BTCUSDT ETHUSDT SOLUSDT
    python3 scripts/eval_multi_asset.py --equal-weight
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import numpy as np
import pandas as pd
import time
import argparse
import importlib

FEE = 0.001
INITIAL_CAPITAL = 100_000.0
WARMUP = 200
DATA_DIR = '/home/ubuntu/TradingAgents/data/historical'

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(symbol: str) -> pd.DataFrame:
    from tradingagents.research.per_asset_router import DATA_FILE_MAP
    entry = DATA_FILE_MAP.get(symbol, symbol.replace('USDT', '_USD'))
    if isinstance(entry, tuple):
        label, tf = entry
    else:
        label, tf = entry, '1h'
    path = f'{DATA_DIR}/{label}_{tf}_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ── Vectorised Indicators (identical to eval_per_asset_oos.py) ───────────────

def _ewm_vec(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = float(arr[0])
    for i in range(1, len(arr)):
        out[i] = alpha * float(arr[i]) + (1 - alpha) * out[i - 1]
    return out

def _atr_vec(high, low, close, period=14):
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return _ewm_vec(tr, period)

def _adx_vec(high, low, close, period=14):
    pc = np.roll(close, 1); pc[0] = close[0]
    ph = np.roll(high, 1);  ph[0] = high[0]
    pl = np.roll(low, 1);   pl[0] = low[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    dm_p = np.where((high - ph) > (pl - low), np.maximum(high - ph, 0.0), 0.0)
    dm_m = np.where((pl - low) > (high - ph), np.maximum(pl - low, 0.0), 0.0)
    atr_s = _ewm_vec(tr, period)
    safe = np.where(atr_s > 0, atr_s, 1e-9)
    di_p = 100.0 * _ewm_vec(dm_p, period) / safe
    di_m = 100.0 * _ewm_vec(dm_m, period) / safe
    denom = np.where((di_p + di_m) > 0, di_p + di_m, 1e-9)
    dx = 100.0 * np.abs(di_p - di_m) / denom
    return _ewm_vec(dx, period), di_p, di_m

def _hurst_fast_vec(close, window=96):
    """Variance ratio Hurst — O(n). Identical to eval_per_asset_oos.py."""
    n = len(close)
    log_ret = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
    log_ret = np.concatenate([[0], log_ret])
    hurst = np.full(n, 0.5)
    k = 16
    for i in range(window, n):
        seg = log_ret[i-window:i]
        var1 = np.var(seg)
        if var1 < 1e-15:
            continue
        n_agg = len(seg) // k
        if n_agg < 2:
            continue
        agg = seg[:n_agg*k].reshape(n_agg, k).sum(axis=1)
        var_k = np.var(agg)
        vr = var_k / (k * var1) if var1 > 0 else 1.0
        if vr > 0:
            hurst[i] = np.clip(np.log(vr) / (2 * np.log(k)) + 0.5, 0, 1)
    return hurst

def _rolling_mean(arr, window):
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0)
    out = np.full(len(arr), np.nan)
    out[window-1:] = (cs[window:] - cs[:-window]) / window
    return out


# ── Entry Signal Generators (identical logic to eval_per_asset_oos.py) ───────

def generate_atr_expansion_entries(df, cfg, warmup=WARMUP):
    """ATR Expansion: uses midpoint for bull/bear (matching proven harness)."""
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)

    atr_period = cfg.get('atr_period', 14)
    expansion_mult = cfg.get('expansion_mult', 3.0)
    vol_mult = cfg.get('vol_mult', 1.5)

    atr = _atr_vec(high, low, close, atr_period)
    prev_atr = np.roll(atr, 1); prev_atr[0] = atr[0]
    bar_range = high - low
    vol_ma = _rolling_mean(volume, 20)
    vol_ma[:20] = volume[:20].mean()

    expansion = bar_range > expansion_mult * prev_atr
    vol_surge = volume > vol_mult * vol_ma
    bar_bull = close > (high + low) / 2  # midpoint — matches proven harness
    bar_bear = close < (high + low) / 2

    entries = np.zeros(n)
    entries[(expansion & vol_surge & bar_bull)] = 1
    entries[(expansion & vol_surge & bar_bear)] = -1
    entries[:warmup] = 0
    return entries


def generate_donchian_entries(df, cfg, warmup=WARMUP):
    """Donchian Momentum: identical to eval_per_asset_oos.py (vectorised)."""
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)

    dp = cfg.get('donchian_period', 25)
    adx_min = cfg.get('adx_min', 18)
    adx_trend = cfg.get('adx_trend', 22)
    vol_mult = cfg.get('vol_mult', 2.0)
    hurst_min = cfg.get('hurst_min', 0.48)
    vol_atr_max = cfg.get('vol_atr_max', None)
    atr_donchian_factor = cfg.get('atr_donchian_factor', None)

    atr = _atr_vec(high, low, close, 14)
    adx, _, _ = _adx_vec(high, low, close, 14)
    hurst = _hurst_fast_vec(close, 96)
    vol_ma = _rolling_mean(volume, 20)
    vol_ma[:20] = volume[:20].mean()

    # Donchian channels — optionally adaptive
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)

    if atr_donchian_factor is not None:
        # Adaptive Donchian period based on ATR ratio
        long_atr_ma = _rolling_mean(atr, dp * 2)
        for i in range(dp + 1, n):
            lt_avg = long_atr_ma[i] if not np.isnan(long_atr_ma[i]) else atr[i]
            if lt_avg > 0:
                vol_ratio = atr[i] / lt_avg
                adp = int(np.clip(dp / (vol_ratio ** atr_donchian_factor), 15, 45))
            else:
                adp = dp
            start = max(0, i - adp - 1)
            dc_upper[i] = high[start:i-1].max() if i > adp else np.nan
            dc_lower[i] = low[start:i-1].min() if i > adp else np.nan
    else:
        # Fixed Donchian period
        for i in range(dp + 1, n):
            dc_upper[i] = high[i-dp-1:i-1].max()
            dc_lower[i] = low[i-dp-1:i-1].min()

    trending = (adx >= adx_trend) & (hurst >= hurst_min)
    adx_ok = adx >= adx_min
    vol_ok = volume >= vol_mult * vol_ma

    # Optional ATR volatility filter
    if vol_atr_max is not None:
        atr_pct = atr / np.maximum(close, 1)
        low_vol = atr_pct <= vol_atr_max
    else:
        low_vol = np.ones(n, dtype=bool)

    long_sig = (close > dc_upper) & adx_ok & vol_ok & low_vol & trending
    short_sig = (close < dc_lower) & adx_ok & vol_ok & low_vol & trending

    entries = np.zeros(n)
    entries[long_sig] = 1
    entries[short_sig] = -1
    entries[:warmup] = 0
    return entries


# ── Position Simulation (identical to eval_per_asset_oos.py) ─────────────────

def simulate_positions(entries, close, high, low, atr, stop_mult, tp_mult, max_hold):
    n = len(close)
    pnl = np.zeros(n)
    trades = []
    pos = 0
    entry_price = 0.0
    bars_held = 0

    for i in range(1, n):
        if pos != 0:
            bars_held += 1
            if pos == 1:
                sl_hit = low[i] <= entry_price - stop_mult * atr[i]
                tp_hit = high[i] >= entry_price + tp_mult * atr[i]
            else:
                sl_hit = high[i] >= entry_price + stop_mult * atr[i]
                tp_hit = low[i] <= entry_price - tp_mult * atr[i]

            exit_price = None
            if sl_hit:
                exit_price = entry_price - pos * stop_mult * atr[i]
            elif tp_hit:
                exit_price = entry_price + pos * tp_mult * atr[i]
            elif bars_held >= max_hold:
                exit_price = close[i]

            if exit_price is not None:
                ret = pos * (exit_price - entry_price) / entry_price - FEE
                pnl[i] = ret
                trades.append(ret)
                pos = 0
                bars_held = 0

        if pos == 0 and entries[i] != 0:
            pos = int(entries[i])
            entry_price = close[i]
            bars_held = 0
            pnl[i] -= FEE

    return pnl, trades


def compute_metrics(pnl, trades, capital, bars_per_year=8760):
    equity = capital * np.cumprod(1 + pnl)
    total_return = (equity[-1] / capital - 1) * 100
    sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(bars_per_year)) if np.std(pnl) > 0 else 0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak) * 100)
    n_trades = len(trades)
    win_rate = float(np.sum(np.array(trades) > 0) / n_trades * 100) if n_trades > 0 else 0
    wins = sum(t for t in trades if t > 0)
    losses = abs(sum(t for t in trades if t < 0))
    pf = float(wins / losses) if losses > 0 else 999
    return {
        'sharpe': sharpe, 'total_return': total_return, 'max_dd': max_dd,
        'n_trades': n_trades, 'win_rate': win_rate, 'pf': pf, 'pnl': pnl,
    }


# ── Risk-Parity Weighting ───────────────────────────────────────────────────

def compute_risk_parity_weights(pnl_dict: dict, lookback: int = 720) -> dict:
    """Inverse-volatility (risk-parity) weights."""
    vols = {}
    for sym, pnl in pnl_dict.items():
        recent = pnl[-lookback:] if len(pnl) > lookback else pnl
        vol = float(np.std(recent))
        vols[sym] = max(vol, 1e-8)

    inv_vols = {sym: 1.0 / v for sym, v in vols.items()}
    total = sum(inv_vols.values())
    return {sym: iv / total for sym, iv in inv_vols.items()}


# ── Single Asset Eval (exported for Bayesian optimiser) ──────────────────────

def get_bars_per_year(symbol: str, cfg: dict = None) -> int:
    """Return annualisation factor: 8760 for hourly, 252 for daily."""
    if cfg and cfg.get('timeframe') == '1d':
        return 252
    from tradingagents.research.per_asset_router import DATA_FILE_MAP
    entry = DATA_FILE_MAP.get(symbol)
    if isinstance(entry, tuple) and entry[1] == '1d':
        return 252
    return 8760


def get_warmup(symbol: str, cfg: dict = None) -> int:
    """Return warmup period: 200 for hourly, 50 for daily."""
    if cfg and cfg.get('timeframe') == '1d':
        return 50
    from tradingagents.research.per_asset_router import DATA_FILE_MAP
    entry = DATA_FILE_MAP.get(symbol)
    if isinstance(entry, tuple) and entry[1] == '1d':
        return 50
    return 200


def eval_single_asset(symbol: str, cfg: dict = None):
    """Evaluate a single asset. Returns metrics dict with 'pnl' array."""
    if cfg is None:
        from tradingagents.research.per_asset_router import ASSET_CONFIG
        cfg = ASSET_CONFIG[symbol]

    df = load_data(symbol)
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    atr = _atr_vec(high, low, close, cfg.get('atr_period', 14))

    strategy = cfg.get('strategy', 'ATR_EXPANSION')
    warmup = get_warmup(symbol, cfg)
    if strategy == 'ATR_EXPANSION':
        entries = generate_atr_expansion_entries(df, cfg, warmup=warmup)
    else:
        entries = generate_donchian_entries(df, cfg, warmup=warmup)

    pnl, trades = simulate_positions(
        entries, close, high, low, atr,
        cfg['stop_mult'], cfg['tp_mult'], cfg['max_hold_bars'])

    bpy = get_bars_per_year(symbol, cfg)
    return compute_metrics(pnl, trades, INITIAL_CAPITAL, bars_per_year=bpy)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Multi-Asset OOS Evaluation')
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--equal-weight', action='store_true')
    args = parser.parse_args()

    t0 = time.time()

    mod = importlib.import_module('tradingagents.research.per_asset_router')
    importlib.reload(mod)
    ASSET_CONFIG = mod.ASSET_CONFIG

    symbols = args.symbols or list(ASSET_CONFIG.keys())
    symbols = [s.upper() for s in symbols]

    print("=" * 80)
    print(f"MULTI-ASSET OOS EVALUATION (2022-2026, 35k bars, {FEE*100:.1f}% fees)")
    print(f"Assets: {len(symbols)} | Weighting: {'Equal' if args.equal_weight else 'Risk-Parity'}")
    print("=" * 80)

    asset_metrics = {}
    asset_pnl = {}
    for sym in symbols:
        cfg = ASSET_CONFIG.get(sym)
        if cfg is None:
            print(f"  WARNING: {sym} not in ASSET_CONFIG, skipping")
            continue
        try:
            metrics = eval_single_asset(sym, cfg)
            asset_metrics[sym] = metrics
            asset_pnl[sym] = metrics['pnl']
        except Exception as e:
            print(f"  ERROR: {sym}: {e}")

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

    if len(asset_pnl) >= 2:
        # Separate crypto (hourly) and traditional (daily) assets
        # Convert all to daily equity curves for fair portfolio combination
        daily_returns = {}
        for sym, pnl in asset_pnl.items():
            bpy = get_bars_per_year(sym, ASSET_CONFIG.get(sym))
            equity = INITIAL_CAPITAL * np.cumprod(1 + pnl)
            if bpy == 8760:  # hourly → resample to daily (24 bars per day)
                n_days = len(equity) // 24
                if n_days > 0:
                    daily_eq = equity[23::24][:n_days]  # end-of-day values
                    daily_ret = np.diff(daily_eq) / daily_eq[:-1]
                    daily_returns[sym] = daily_ret
            else:  # already daily
                daily_ret = np.diff(equity) / equity[:-1]
                daily_returns[sym] = daily_ret

        if args.equal_weight:
            weights = {sym: 1.0 / len(daily_returns) for sym in daily_returns}
        else:
            weights = compute_risk_parity_weights(daily_returns)

        min_len = min(len(r) for r in daily_returns.values())
        port_daily = np.zeros(min_len)
        for sym, ret in daily_returns.items():
            port_daily += weights[sym] * ret[:min_len]

        port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_daily)
        port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100
        port_sharpe = float(np.mean(port_daily) / np.std(port_daily) * np.sqrt(252)) if np.std(port_daily) > 0 else 0
        peak = np.maximum.accumulate(port_equity)
        port_dd = float(np.max((peak - port_equity) / peak) * 100)
        total_trades = sum(m['n_trades'] for m in asset_metrics.values())

        print(f"\n{'─' * 80}")
        print(f"PORTFOLIO ({len(asset_pnl)} assets, {'equal' if args.equal_weight else 'risk-parity'} weighted)")
        print(f"{'─' * 80}")
        print(f"  Sharpe:       {port_sharpe:.3f}")
        print(f"  Total Return: {port_return:.1f}%")
        print(f"  Max Drawdown: {port_dd:.1f}%")
        print(f"  Total Trades: {total_trades}")

        print(f"\n  Weights:")
        for sym in sorted(weights, key=weights.get, reverse=True):
            print(f"    {sym:10s}: {weights[sym]*100:5.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Eval time: {elapsed:.1f}s")
    sharpe_val = port_sharpe if len(asset_pnl) >= 2 else list(asset_metrics.values())[0]['sharpe']
    print(f"METRIC: {sharpe_val:.6f}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
