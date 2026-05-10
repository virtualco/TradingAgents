"""
Fast Parameter Optimisation — TradingAgents v5
================================================
Pre-computes all slow indicators (Hurst, ADX, ATR, Volume MA) once,
then runs all 320 parameter combinations in pure NumPy — no pandas overhead.
Expected runtime: ~30 seconds for all 320 combinations on both BTC and ETH.
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import numpy as np
import pandas as pd
import json
from itertools import product

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data(symbol: str) -> pd.DataFrame:
    path = f'/home/ubuntu/TradingAgents/data/historical/{symbol}_USD_1h_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# ── Pre-compute Indicators ────────────────────────────────────────────────────
def precompute(df: pd.DataFrame) -> dict:
    """Compute all slow indicators once and cache as numpy arrays."""
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)

    # ATR(14)
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = _ewm_np(tr, 14)

    # ADX(14)
    dm_plus  = np.maximum(high - np.roll(high, 1), 0)
    dm_minus = np.maximum(np.roll(low, 1) - low, 0)
    dm_plus[0] = dm_minus[0] = 0
    mask = dm_plus <= dm_minus; dm_plus[mask] = 0
    mask = dm_minus <= dm_plus; dm_minus[mask] = 0
    atr_s    = _ewm_np(tr, 14)
    di_plus  = 100 * _ewm_np(dm_plus, 14)  / np.where(atr_s > 0, atr_s, np.nan)
    di_minus = 100 * _ewm_np(dm_minus, 14) / np.where(atr_s > 0, atr_s, np.nan)
    denom = np.where((di_plus + di_minus) > 0, di_plus + di_minus, np.nan)
    dx = 100 * np.abs(di_plus - di_minus) / denom
    adx = _ewm_np(np.nan_to_num(dx, nan=0.0), 14)

    # Volume MA(20)
    vol_ma = np.array([volume[max(0,i-19):i+1].mean() for i in range(n)])

    # Hurst(96) — computed once, reused for all ADX thresholds
    print("  Pre-computing Hurst exponent (slow step, done once)...")
    hurst = _rolling_hurst(close, window=96)

    # Volatility filter: ATR/price <= 3%
    low_vol = (atr / np.where(close > 0, close, np.nan)) <= 0.03
    low_vol = np.nan_to_num(low_vol, nan=False).astype(bool)

    # Years for per-year analysis
    years = np.array([t.year for t in df.index])

    return {
        'close': close, 'high': high, 'low': low, 'volume': volume,
        'atr': atr, 'adx': adx, 'vol_ma': vol_ma,
        'hurst': hurst, 'low_vol': low_vol, 'years': years, 'n': n
    }

def _ewm_np(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential weighted mean — pure numpy, matches pandas ewm(span, adjust=False)."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def _rolling_hurst(close: np.ndarray, window: int = 96) -> np.ndarray:
    """Fast rolling Hurst using variance-of-increments."""
    n = len(close)
    hurst = np.full(n, 0.5)
    lags = [l for l in [2, 4, 8, 16, 32] if l < window // 2]
    if len(lags) < 2:
        return hurst
    log_lags = np.log(lags)
    for i in range(window, n):
        x = np.log(np.abs(close[i-window:i]) + 1e-10)
        vl = [np.var(x[l:] - x[:-l]) for l in lags]
        try:
            slope = np.polyfit(log_lags, np.log(np.array(vl) + 1e-20), 1)[0]
            hurst[i] = float(np.clip(slope / 2.0, 0.0, 1.0))
        except:
            hurst[i] = 0.5
    return hurst

# ── Donchian Channel ──────────────────────────────────────────────────────────
def donchian_channels(high: np.ndarray, low: np.ndarray, period: int):
    """Pre-compute Donchian channels for a given period."""
    n = len(high)
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    for i in range(period, n):
        dc_upper[i] = high[i-period:i].max()  # shifted: use bars [i-period, i-1]
        dc_lower[i] = low[i-period:i].min()
    return dc_upper, dc_lower

# ── Fast Backtest ─────────────────────────────────────────────────────────────
def backtest(cache: dict, dc_upper: np.ndarray, dc_lower: np.ndarray,
             adx_min: float, volume_mult: float, adx_trend_threshold: float,
             hurst_trend_min: float = 0.48, fee: float = 0.001,
             max_hold: int = 72) -> dict:
    """Pure numpy backtest — no pandas overhead."""
    close   = cache['close']
    volume  = cache['volume']
    adx     = cache['adx']
    vol_ma  = cache['vol_ma']
    hurst   = cache['hurst']
    low_vol = cache['low_vol']
    years   = cache['years']
    n       = cache['n']

    # Regime: TRENDING = ADX >= threshold AND Hurst >= hurst_trend_min
    trending = (adx >= adx_trend_threshold) & (hurst >= hurst_trend_min)

    # Entry conditions (already shifted via Donchian)
    vol_ok    = volume >= volume_mult * vol_ma
    adx_ok    = adx >= adx_min
    long_sig  = (close > dc_upper) & adx_ok & vol_ok & low_vol & trending
    short_sig = (close < dc_lower) & adx_ok & vol_ok & low_vol & trending

    # Shift by 1 bar
    long_sig  = np.roll(long_sig, 1); long_sig[0] = False
    short_sig = np.roll(short_sig, 1); short_sig[0] = False

    # Simulate trades
    trades = []
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        if position == 0:
            if long_sig[i]:
                position = 1
                entry_price = close[i] * (1 + fee)
                entry_bar = i
            elif short_sig[i]:
                position = -1
                entry_price = close[i] * (1 - fee)
                entry_bar = i
        else:
            exit_now = (position == 1 and short_sig[i]) or \
                       (position == -1 and long_sig[i]) or \
                       (i - entry_bar > max_hold)
            if exit_now:
                exit_price = close[i] * (1 - fee * position)
                pnl = (exit_price / entry_price - 1) * position
                trades.append({'pnl': pnl, 'year': int(years[entry_bar])})
                position = 0

    if len(trades) < 5:
        return {'sharpe': -99, 'total_return': -99, 'max_dd': 99,
                'n_trades': len(trades), 'win_rate': 0}

    pnls = np.array([t['pnl'] for t in trades])
    eq   = np.cumprod(1 + pnls)
    eq   = np.insert(eq, 0, 1.0)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    max_dd = abs(dd.min())
    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(len(pnls)) if pnls.std() > 0 else 0

    # Per-year breakdown
    by_year = {}
    for yr in sorted(set(t['year'] for t in trades)):
        yr_pnls = np.array([t['pnl'] for t in trades if t['year'] == yr])
        by_year[yr] = {'n': len(yr_pnls), 'pnl': round(yr_pnls.sum()*100, 2),
                       'wr': round((yr_pnls > 0).mean()*100, 1)}

    return {
        'sharpe': round(float(sharpe), 4),
        'total_return': round(float(eq[-1] - 1) * 100, 2),
        'max_dd': round(float(max_dd) * 100, 2),
        'n_trades': len(pnls),
        'win_rate': round(float((pnls > 0).mean()) * 100, 1),
        'by_year': by_year,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading BTC and ETH data...")
    btc_df = load_data('BTC')
    eth_df = load_data('ETH')

    print("Pre-computing BTC indicators...")
    btc_cache = precompute(btc_df)
    print("Pre-computing ETH indicators...")
    eth_cache = precompute(eth_df)

    # Pre-compute Donchian channels for each period
    donchian_periods     = [10, 15, 20, 25, 30]
    adx_mins             = [18, 20, 22, 25]
    volume_mults         = [1.0, 1.2, 1.5, 2.0]
    adx_trend_thresholds = [18, 20, 22, 25]

    print("Pre-computing Donchian channels for all periods...")
    btc_dc = {}
    eth_dc = {}
    for dp in donchian_periods:
        btc_dc[dp] = donchian_channels(btc_cache['high'], btc_cache['low'], dp)
        eth_dc[dp] = donchian_channels(eth_cache['high'], eth_cache['low'], dp)

    total = len(donchian_periods) * len(adx_mins) * len(volume_mults) * len(adx_trend_thresholds)
    print(f"\nRunning {total} combinations at 0.1% fees...")

    results = []
    best_sharpe = -99
    best_params = {}

    for i, (dp, am, vm, att) in enumerate(product(donchian_periods, adx_mins, volume_mults, adx_trend_thresholds)):
        btc_r = backtest(btc_cache, btc_dc[dp][0], btc_dc[dp][1],
                         adx_min=am, volume_mult=vm, adx_trend_threshold=att, fee=0.001)
        eth_r = backtest(eth_cache, eth_dc[dp][0], eth_dc[dp][1],
                         adx_min=am, volume_mult=vm, adx_trend_threshold=att, fee=0.001)

        avg_sharpe = (btc_r['sharpe'] + eth_r['sharpe']) / 2
        avg_return = (btc_r['total_return'] + eth_r['total_return']) / 2
        avg_dd     = (btc_r['max_dd'] + eth_r['max_dd']) / 2

        row = {
            'donchian_period': dp, 'adx_min': am, 'volume_mult': vm,
            'adx_trend_threshold': att,
            'btc_sharpe': btc_r['sharpe'], 'eth_sharpe': eth_r['sharpe'],
            'avg_sharpe': round(avg_sharpe, 4),
            'btc_return': btc_r['total_return'], 'eth_return': eth_r['total_return'],
            'avg_return': round(avg_return, 2),
            'avg_dd': round(avg_dd, 2),
            'btc_trades': btc_r['n_trades'], 'eth_trades': eth_r['n_trades'],
            'btc_wr': btc_r['win_rate'], 'eth_wr': eth_r['win_rate'],
            'btc_by_year': btc_r.get('by_year', {}),
            'eth_by_year': eth_r.get('by_year', {}),
        }
        results.append(row)

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_params = row
            print(f"  [{i+1:>3}/{total}] NEW BEST: dp={dp} adx_min={am} vol={vm} att={att} | "
                  f"Sharpe={avg_sharpe:.4f} Return={avg_return:.1f}% DD={avg_dd:.1f}%")

    results.sort(key=lambda x: x['avg_sharpe'], reverse=True)

    print(f"\n{'='*80}")
    print("TOP 10 PARAMETER COMBINATIONS (0.1% fees)")
    print(f"{'='*80}")
    print(f"{'#':>3}  {'dp':>4}  {'adx_min':>7}  {'vol':>5}  {'att':>5}  "
          f"{'Sharpe':>8}  {'BTC%':>7}  {'ETH%':>7}  {'DD%':>6}  {'Trades':>7}")
    for j, r in enumerate(results[:10]):
        print(f"  {j+1:>2}  {r['donchian_period']:>4}  {r['adx_min']:>7}  {r['volume_mult']:>5}  "
              f"{r['adx_trend_threshold']:>5}  {r['avg_sharpe']:>8.4f}  "
              f"{r['btc_return']:>6.1f}%  {r['eth_return']:>6.1f}%  "
              f"{r['avg_dd']:>5.1f}%  {r['btc_trades']+r['eth_trades']:>7}")

    # Best params detail
    bp = best_params
    print(f"\n{'='*80}")
    print(f"BEST PARAMS: dp={bp['donchian_period']} adx_min={bp['adx_min']} "
          f"vol={bp['volume_mult']} att={bp['adx_trend_threshold']}")
    print(f"  BTC: Return={bp['btc_return']}%  Sharpe={bp['btc_sharpe']}  "
          f"WR={bp['btc_wr']}%  Trades={bp['btc_trades']}")
    if bp.get('btc_by_year'):
        for yr, d in sorted(bp['btc_by_year'].items()):
            print(f"    {yr}: {d['n']} trades  WR={d['wr']}%  PnL={d['pnl']}%")
    print(f"  ETH: Return={bp['eth_return']}%  Sharpe={bp['eth_sharpe']}  "
          f"WR={bp['eth_wr']}%  Trades={bp['eth_trades']}")
    if bp.get('eth_by_year'):
        for yr, d in sorted(bp['eth_by_year'].items()):
            print(f"    {yr}: {d['n']} trades  WR={d['wr']}%  PnL={d['pnl']}%")

    # Slippage stress test at 0.25%
    print(f"\n{'='*80}")
    print("SLIPPAGE STRESS TEST (0.25% fees)")
    btc_s = backtest(btc_cache, btc_dc[bp['donchian_period']][0], btc_dc[bp['donchian_period']][1],
                     adx_min=bp['adx_min'], volume_mult=bp['volume_mult'],
                     adx_trend_threshold=bp['adx_trend_threshold'], fee=0.0025)
    eth_s = backtest(eth_cache, eth_dc[bp['donchian_period']][0], eth_dc[bp['donchian_period']][1],
                     adx_min=bp['adx_min'], volume_mult=bp['volume_mult'],
                     adx_trend_threshold=bp['adx_trend_threshold'], fee=0.0025)
    print(f"  BTC: Return={btc_s['total_return']}%  Sharpe={btc_s['sharpe']}  "
          f"DD={btc_s['max_dd']}%  WR={btc_s['win_rate']}%  Trades={btc_s['n_trades']}")
    print(f"  ETH: Return={eth_s['total_return']}%  Sharpe={eth_s['sharpe']}  "
          f"DD={eth_s['max_dd']}%  WR={eth_s['win_rate']}%  Trades={eth_s['n_trades']}")

    # Save
    output = {
        'top10': results[:10],
        'best_01pct': bp,
        'slippage_025pct': {'btc': btc_s, 'eth': eth_s}
    }
    with open('/tmp/optimise_v5_fast_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → /tmp/optimise_v5_fast_results.json")

if __name__ == '__main__':
    main()
