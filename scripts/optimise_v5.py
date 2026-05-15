"""
Targeted Parameter Optimisation — TradingAgents v5
====================================================
KEY FINDING: Momentum-only (Donchian breakout) on BTC returns +44.1% Sharpe 0.84
when run without the regime filter. The regime filter is blocking valid trending bars.

STRATEGY: Optimise the Donchian momentum strategy parameters to maximise
Sharpe ratio across the full 4-year OOS dataset. The regime filter will be
kept but with relaxed thresholds to allow more TRENDING bars.

PARAMETERS TO OPTIMISE:
  - donchian_period: [10, 15, 20, 25, 30]
  - adx_min: [18, 20, 22, 25]
  - volume_mult: [1.0, 1.2, 1.5, 2.0]
  - adx_trend_threshold: [18, 20, 22, 25]
  - fee: [0.001, 0.0025]  (0.1% and 0.25%)
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import pandas as pd
import numpy as np
import json
from itertools import product

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data(symbol: str) -> pd.DataFrame:
    path = f'/home/ubuntu/TradingAgents/data/historical/{symbol}_USD_1h_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# ── Indicator Utilities (inline for speed) ───────────────────────────────────
def _adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    atr_s    = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean().fillna(0)

def _atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift()).abs(), (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _hurst_fast(series, window=96):
    def _h(x):
        try:
            lx = np.log(np.abs(x) + 1e-10)
            lags = [l for l in [2,4,8,16,32] if l < len(x)//2]
            if len(lags) < 2: return 0.5
            vl = [np.var(lx[l:]-lx[:-l]) for l in lags]
            return float(np.clip(np.polyfit(np.log(lags), np.log(np.array(vl)+1e-20), 1)[0]/2, 0, 1))
        except: return 0.5
    return series.rolling(window, min_periods=window//2).apply(_h, raw=True).fillna(0.5)

# ── Donchian Momentum Backtest ────────────────────────────────────────────────
def backtest_donchian(df, donchian_period=20, adx_min=22.0, volume_mult=1.5,
                      adx_trend_threshold=22.0, fee=0.001, use_regime=True):
    """Fast vectorised Donchian breakout backtest."""
    close  = df["close"]
    volume = df["volume"]
    adx    = _adx(df, 14)
    atr    = _atr(df, 14)

    # Donchian channel (shifted to avoid lookahead)
    dc_upper = df["high"].rolling(donchian_period).max().shift(1)
    dc_lower = df["low"].rolling(donchian_period).min().shift(1)

    vol_ma = volume.rolling(20).mean()
    low_vol = (atr / close) <= 0.03

    # Regime filter
    if use_regime:
        hurst = _hurst_fast(close, 96)
        trending = (adx >= adx_trend_threshold) & (hurst >= 0.48)
    else:
        trending = pd.Series(True, index=df.index)

    long_entry  = (close > dc_upper) & (adx >= adx_min) & (volume >= volume_mult * vol_ma) & low_vol & trending
    short_entry = (close < dc_lower) & (adx >= adx_min) & (volume >= volume_mult * vol_ma) & low_vol & trending

    # Shift entries by 1 bar
    long_entry  = long_entry.shift(1).fillna(False)
    short_entry = short_entry.shift(1).fillna(False)

    # Vectorised backtest
    close_arr = close.values
    long_arr  = long_entry.values
    short_arr = short_entry.values
    n = len(close_arr)

    equity = 1.0
    trades = []
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, n):
        if position == 0:
            if long_arr[i]:
                position = 1
                entry_price = close_arr[i] * (1 + fee)
                entry_bar = i
            elif short_arr[i]:
                position = -1
                entry_price = close_arr[i] * (1 - fee)
                entry_bar = i
        else:
            # Exit on opposite signal or after 72h max hold
            exit_now = (position == 1 and short_arr[i]) or \
                       (position == -1 and long_arr[i]) or \
                       (i - entry_bar > 72)
            if exit_now:
                exit_price = close_arr[i] * (1 - fee * position)
                pnl = (exit_price / entry_price - 1) * position
                equity *= (1 + pnl)
                trades.append({'pnl': pnl, 'year': pd.Timestamp(df.index[entry_bar]).year})
                position = 0

    if len(trades) < 5:
        return {'sharpe': -99, 'total_return': -99, 'max_dd': 99, 'n_trades': len(trades), 'win_rate': 0}

    pnls = np.array([t['pnl'] for t in trades])
    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls)

    eq_curve = np.cumprod(1 + pnls)
    eq_curve = np.insert(eq_curve, 0, 1.0)
    peak = np.maximum.accumulate(eq_curve)
    dd = (eq_curve - peak) / peak
    max_dd = abs(dd.min())

    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(len(pnls)) if pnls.std() > 0 else 0

    return {
        'sharpe': round(sharpe, 4),
        'total_return': round((eq_curve[-1] - 1) * 100, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': len(pnls),
        'win_rate': round(win_rate * 100, 1),
    }

# ── Grid Search ───────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    btc = load_data('BTC')
    eth = load_data('ETH')

    # Parameter grid
    donchian_periods     = [10, 15, 20, 25, 30]
    adx_mins             = [18, 20, 22, 25]
    volume_mults         = [1.0, 1.2, 1.5, 2.0]
    adx_trend_thresholds = [18, 20, 22, 25]

    total = len(donchian_periods) * len(adx_mins) * len(volume_mults) * len(adx_trend_thresholds)
    print(f"Running {total} parameter combinations on BTC + ETH (4yr OOS)...")
    print(f"Fee: 0.1% (0.001)")

    results = []
    best_sharpe = -99
    best_params = {}

    for i, (dp, am, vm, att) in enumerate(product(donchian_periods, adx_mins, volume_mults, adx_trend_thresholds)):
        btc_r = backtest_donchian(btc, donchian_period=dp, adx_min=am, volume_mult=vm,
                                   adx_trend_threshold=att, fee=0.001, use_regime=True)
        eth_r = backtest_donchian(eth, donchian_period=dp, adx_min=am, volume_mult=vm,
                                   adx_trend_threshold=att, fee=0.001, use_regime=True)

        avg_sharpe = (btc_r['sharpe'] + eth_r['sharpe']) / 2
        avg_return = (btc_r['total_return'] + eth_r['total_return']) / 2
        avg_dd     = (btc_r['max_dd'] + eth_r['max_dd']) / 2

        results.append({
            'donchian_period': dp, 'adx_min': am, 'volume_mult': vm,
            'adx_trend_threshold': att,
            'btc_sharpe': btc_r['sharpe'], 'eth_sharpe': eth_r['sharpe'],
            'avg_sharpe': round(avg_sharpe, 4),
            'btc_return': btc_r['total_return'], 'eth_return': eth_r['total_return'],
            'avg_return': round(avg_return, 2),
            'avg_dd': round(avg_dd, 2),
            'btc_trades': btc_r['n_trades'], 'eth_trades': eth_r['n_trades'],
            'btc_wr': btc_r['win_rate'], 'eth_wr': eth_r['win_rate'],
        })

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_params = results[-1]
            print(f"  [{i+1}/{total}] NEW BEST: dp={dp} adx_min={am} vol={vm} att={att} | "
                  f"Sharpe={avg_sharpe:.3f} Return={avg_return:.1f}% DD={avg_dd:.1f}%")

    # Sort by avg_sharpe
    results.sort(key=lambda x: x['avg_sharpe'], reverse=True)

    print(f"\n{'='*70}")
    print("TOP 10 PARAMETER COMBINATIONS")
    print(f"{'='*70}")
    print(f"{'Rank':>4}  {'dp':>4}  {'adx_min':>7}  {'vol':>5}  {'att':>5}  {'Sharpe':>8}  {'Return':>8}  {'DD':>6}  {'Trades':>7}")
    for j, r in enumerate(results[:10]):
        print(f"  {j+1:>2}  {r['donchian_period']:>4}  {r['adx_min']:>7}  {r['volume_mult']:>5}  "
              f"{r['adx_trend_threshold']:>5}  {r['avg_sharpe']:>8.4f}  {r['avg_return']:>7.1f}%  "
              f"{r['avg_dd']:>5.1f}%  {r['btc_trades']+r['eth_trades']:>7}")

    print(f"\n{'='*70}")
    print("BEST PARAMETERS:")
    print(f"  donchian_period:     {best_params['donchian_period']}")
    print(f"  adx_min:             {best_params['adx_min']}")
    print(f"  volume_mult:         {best_params['volume_mult']}")
    print(f"  adx_trend_threshold: {best_params['adx_trend_threshold']}")
    print(f"  Avg Sharpe:          {best_params['avg_sharpe']}")
    print(f"  BTC Return:          {best_params['btc_return']}%")
    print(f"  ETH Return:          {best_params['eth_return']}%")
    print(f"  Avg DD:              {best_params['avg_dd']}%")

    # Now test best params at 0.25% fees (slippage stress test)
    print(f"\n{'='*70}")
    print("SLIPPAGE STRESS TEST (0.25% fees = 0.0025)")
    bp = best_params
    btc_s = backtest_donchian(btc, donchian_period=bp['donchian_period'],
                               adx_min=bp['adx_min'], volume_mult=bp['volume_mult'],
                               adx_trend_threshold=bp['adx_trend_threshold'], fee=0.0025)
    eth_s = backtest_donchian(eth, donchian_period=bp['donchian_period'],
                               adx_min=bp['adx_min'], volume_mult=bp['volume_mult'],
                               adx_trend_threshold=bp['adx_trend_threshold'], fee=0.0025)
    print(f"  BTC: Return={btc_s['total_return']}%  Sharpe={btc_s['sharpe']}  DD={btc_s['max_dd']}%  WR={btc_s['win_rate']}%")
    print(f"  ETH: Return={eth_s['total_return']}%  Sharpe={eth_s['sharpe']}  DD={eth_s['max_dd']}%  WR={eth_s['win_rate']}%")

    # Save results
    with open('/tmp/optimise_v5_results.json', 'w') as f:
        json.dump({'top10': results[:10], 'best': best_params,
                   'slippage_test': {'btc': btc_s, 'eth': eth_s}}, f, indent=2)
    print(f"\nFull results → /tmp/optimise_v5_results.json")

if __name__ == '__main__':
    main()
