"""
Evaluation Harness — Per-Asset OOS Strategy (VECTORISED + POSITION MANAGEMENT)
================================================================================
Generates entry signals vectorised using parameters IMPORTED from per_asset_router.py,
then simulates position management with:
  - ATR-based stop-loss
  - ATR-based take-profit
  - Max hold time exit
  - 0.1% transaction costs on entry and exit
Target runtime: <5s.
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')
import importlib
import numpy as np
import pandas as pd
import traceback
import time

FEE = 0.001
INITIAL_CAPITAL = 100_000.0
BTC_WEIGHT = 0.6
ETH_WEIGHT = 0.4
WARMUP = 200

def load_data(symbol: str) -> pd.DataFrame:
    path = f'/home/ubuntu/TradingAgents/data/historical/{symbol}_USD_1h_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# ── Vectorised Indicators ─────────────────────────────────────────────────────
def _ewm_vec(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def _atr_vec(high, low, close, period=14):
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ewm_vec(tr, period)

def _adx_vec(high, low, close, period=14):
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    prev_high = np.roll(high, 1); prev_high[0] = high[0]
    prev_low = np.roll(low, 1); prev_low[0] = low[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    dm_p = np.maximum(high - prev_high, 0)
    dm_m = np.maximum(prev_low - low, 0)
    mask = dm_p > dm_m
    dm_m[mask & (dm_p > 0)] = 0
    mask2 = dm_m > dm_p
    dm_p[mask2 & (dm_m > 0)] = 0
    atr_s = _ewm_vec(tr, period)
    di_p = 100 * _ewm_vec(dm_p, period) / np.maximum(atr_s, 1e-10)
    di_m = 100 * _ewm_vec(dm_m, period) / np.maximum(atr_s, 1e-10)
    dx = 100 * np.abs(di_p - di_m) / np.maximum(di_p + di_m, 1e-10)
    adx = _ewm_vec(dx, period)
    return adx, di_p, di_m

def _hurst_fast_vec(close, window=96):
    """Variance ratio Hurst — O(n)."""
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

# ── Import Strategy Parameters ────────────────────────────────────────────────
def get_strategy_params():
    """
    Import BTC_CONFIG and ETH_CONFIG from per_asset_router.py.
    This is what autoresearch modifies — changes to those dicts affect the eval.
    Uses importlib.reload() to pick up latest changes even within the same process.
    """
    try:
        mod_name = 'tradingagents.research.per_asset_router'
        if mod_name in sys.modules:
            mod = importlib.reload(sys.modules[mod_name])
        else:
            mod = importlib.import_module(mod_name)
        return mod.BTC_CONFIG, mod.ETH_CONFIG
    except Exception as e:
        print(f"PARAM IMPORT ERROR: {e}")
        traceback.print_exc()
        # Fallback defaults (original baseline)
        btc = {'atr_period': 14, 'expansion_mult': 3.0, 'vol_mult': 1.5,
               'max_hold_bars': 12, 'stop_mult': 2.0, 'tp_mult': 4.0}
        eth = {'donchian_period': 25, 'adx_min': 18, 'adx_trend': 22,
               'vol_mult': 2.0, 'hurst_min': 0.48, 'vol_atr_max': 0.05,
               'max_hold_bars': 48, 'stop_mult': 2.5, 'tp_mult': 5.0}
        return btc, eth

# ── Entry Signal Generation (parameterised) ──────────────────────────────────
def generate_btc_entries(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """BTC ATR Expansion: discrete entry signals (+1 long, -1 short, 0 no entry)."""
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
    bar_bull = close > (high + low) / 2
    bar_bear = close < (high + low) / 2
    
    entries = np.zeros(n)
    entries[(expansion & vol_surge & bar_bull)] = 1
    entries[(expansion & vol_surge & bar_bear)] = -1
    entries[:WARMUP] = 0
    return entries

def generate_eth_entries(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """ETH Donchian Momentum: discrete entry signals — reads ALL params from cfg."""
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
    vol_atr_max = cfg.get('vol_atr_max', None)  # None = disabled
    atr_donchian_factor = cfg.get('atr_donchian_factor', None)  # None = no adaptive
    
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
    entries[:WARMUP] = 0
    return entries

# ── Position Management Simulation ───────────────────────────────────────────
def simulate_positions(entries, close, high, low, atr, stop_mult, tp_mult, max_hold):
    """
    Simulate discrete trades with stop-loss, take-profit, and max hold.
    Returns array of per-bar P&L (as fraction of entry price).
    """
    n = len(close)
    pnl_bars = np.zeros(n)
    
    i = 0
    trades = []
    while i < n:
        if entries[i] != 0:
            direction = entries[i]  # +1 or -1
            entry_price = close[i]
            entry_atr = atr[i]
            stop_price = entry_price - direction * stop_mult * entry_atr
            tp_price = entry_price + direction * tp_mult * entry_atr
            
            # Simulate hold
            exit_price = None
            exit_bar = None
            for j in range(i + 1, min(i + max_hold + 1, n)):
                if direction == 1:
                    if low[j] <= stop_price:
                        exit_price = stop_price
                        exit_bar = j
                        break
                    if high[j] >= tp_price:
                        exit_price = tp_price
                        exit_bar = j
                        break
                else:
                    if high[j] >= stop_price:
                        exit_price = stop_price
                        exit_bar = j
                        break
                    if low[j] <= tp_price:
                        exit_price = tp_price
                        exit_bar = j
                        break
            
            if exit_price is None:
                exit_bar = min(i + max_hold, n - 1)
                exit_price = close[exit_bar]
            
            trade_pnl = direction * (exit_price - entry_price) / entry_price
            trade_pnl -= 2 * FEE  # entry + exit fees
            
            pnl_bars[exit_bar] += trade_pnl
            trades.append(trade_pnl)
            
            i = exit_bar + 1
        else:
            i += 1
    
    return pnl_bars, trades

def compute_metrics(pnl_bars, trades, capital):
    """Compute strategy metrics from per-bar P&L."""
    equity = capital * np.cumprod(1 + pnl_bars)
    total_return = (equity[-1] / capital - 1) * 100
    
    if np.std(pnl_bars) > 0:
        sharpe = np.mean(pnl_bars) / np.std(pnl_bars) * np.sqrt(8760)
    else:
        sharpe = 0.0
    
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = np.max(dd) * 100
    
    trades_arr = np.array(trades) if trades else np.array([0])
    win_rate = np.sum(trades_arr > 0) / len(trades_arr) * 100 if len(trades_arr) > 0 else 0
    
    gross_profit = np.sum(trades_arr[trades_arr > 0])
    gross_loss = abs(np.sum(trades_arr[trades_arr < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_trades': len(trades),
        'pnl_bars': pnl_bars,
    }

def main():
    t0 = time.time()
    
    # Import params from per_asset_router (what autoresearch modifies)
    btc_cfg, eth_cfg = get_strategy_params()
    
    # Verify PerAssetRouter interface loads
    try:
        from tradingagents.research.per_asset_router import PerAssetRouter
        router = PerAssetRouter()
        test_df = load_data('ETH').iloc[:250]
        test_result = router.generate_signals(test_df, 'ETHUSDT')
        assert 'signal' in test_result
    except Exception as e:
        print(f"IMPORT/INTERFACE ERROR: {e}")
        traceback.print_exc()
        print("METRIC: 0.0")
        return
    
    try:
        btc_df = load_data('BTC')
        eth_df = load_data('ETH')
    except Exception as e:
        print(f"DATA ERROR: {e}")
        print("METRIC: 0.0")
        return
    
    try:
        # Generate entry signals using IMPORTED parameters
        btc_entries = generate_btc_entries(btc_df, btc_cfg)
        eth_entries = generate_eth_entries(eth_df, eth_cfg)
        
        # Position management params from configs
        btc_stop = btc_cfg.get('stop_mult', 2.0)
        btc_tp = btc_cfg.get('tp_mult', 4.0)
        btc_max_hold = btc_cfg.get('max_hold_bars', 12)
        
        eth_stop = eth_cfg.get('stop_mult', 2.5)
        eth_tp = eth_cfg.get('tp_mult', 5.0)
        eth_max_hold = eth_cfg.get('max_hold_bars', 48)
        
        # Simulate with position management
        btc_close = btc_df['close'].values.astype(float)
        btc_high = btc_df['high'].values.astype(float)
        btc_low = btc_df['low'].values.astype(float)
        btc_atr = _atr_vec(btc_high, btc_low, btc_close, btc_cfg.get('atr_period', 14))
        
        eth_close = eth_df['close'].values.astype(float)
        eth_high = eth_df['high'].values.astype(float)
        eth_low = eth_df['low'].values.astype(float)
        eth_atr = _atr_vec(eth_high, eth_low, eth_close, 14)
        
        btc_pnl, btc_trades = simulate_positions(
            btc_entries, btc_close, btc_high, btc_low, btc_atr,
            btc_stop, btc_tp, btc_max_hold)
        
        eth_pnl, eth_trades = simulate_positions(
            eth_entries, eth_close, eth_high, eth_low, eth_atr,
            eth_stop, eth_tp, eth_max_hold)
        
        btc_results = compute_metrics(btc_pnl, btc_trades, INITIAL_CAPITAL * BTC_WEIGHT)
        eth_results = compute_metrics(eth_pnl, eth_trades, INITIAL_CAPITAL * ETH_WEIGHT)
        
    except Exception as e:
        print(f"BACKTEST ERROR: {e}")
        traceback.print_exc()
        print("METRIC: 0.0")
        return
    
    # Portfolio-level
    min_len = min(len(btc_pnl), len(eth_pnl))
    port_pnl = BTC_WEIGHT * btc_pnl[:min_len] + ETH_WEIGHT * eth_pnl[:min_len]
    
    port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_pnl)
    port_return = (port_equity[-1] / INITIAL_CAPITAL - 1) * 100
    
    if np.std(port_pnl) > 0:
        port_sharpe = np.mean(port_pnl) / np.std(port_pnl) * np.sqrt(8760)
    else:
        port_sharpe = 0.0
    
    peak = np.maximum.accumulate(port_equity)
    dd = (peak - port_equity) / peak
    port_max_dd = np.max(dd) * 100
    
    all_trades = btc_trades + eth_trades
    all_arr = np.array(all_trades) if all_trades else np.array([0])
    port_win_rate = np.sum(all_arr > 0) / len(all_arr) * 100 if len(all_arr) > 0 else 0
    
    elapsed = time.time() - t0
    
    print("=" * 60)
    print("PER-ASSET OOS EVALUATION (2022-2026, 35k bars, 0.1% fees)")
    print(f"Eval time: {elapsed:.1f}s | Position mgmt: SL/TP/MaxHold")
    print(f"Params: BTC(exp={btc_cfg.get('expansion_mult')}, vol={btc_cfg.get('vol_mult')}, "
          f"SL={btc_stop}, TP={btc_tp}, hold={btc_max_hold})")
    print(f"        ETH(dp={eth_cfg.get('donchian_period')}, adx_t={eth_cfg.get('adx_trend')}, "
          f"hurst={eth_cfg.get('hurst_min')}, vol={eth_cfg.get('vol_mult')}, "
          f"SL={eth_stop}, TP={eth_tp}, hold={eth_max_hold})")
    print("=" * 60)
    print(f"\n{'Asset':<10} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8} {'PF':>6}")
    print("-" * 60)
    print(f"{'BTC':<10} {btc_results['sharpe']:>8.3f} {btc_results['total_return']:>9.1f}% {btc_results['max_dd']:>7.1f}% {btc_results['win_rate']:>7.1f}% {btc_results['n_trades']:>8} {btc_results['profit_factor']:>5.2f}")
    print(f"{'ETH':<10} {eth_results['sharpe']:>8.3f} {eth_results['total_return']:>9.1f}% {eth_results['max_dd']:>7.1f}% {eth_results['win_rate']:>7.1f}% {eth_results['n_trades']:>8} {eth_results['profit_factor']:>5.2f}")
    print(f"{'PORTFOLIO':<10} {port_sharpe:>8.3f} {port_return:>9.1f}% {port_max_dd:>7.1f}% {port_win_rate:>7.1f}% {len(all_trades):>8} {'':>6}")
    print("=" * 60)
    print(f"\nMETRIC: {port_sharpe:.6f}")

if __name__ == '__main__':
    main()
