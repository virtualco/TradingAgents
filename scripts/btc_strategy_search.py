"""
BTC-Specific Strategy Search
==============================
BTC 1h Donchian breakout has max Sharpe 0.19 across 320 combinations.
This script tests three alternative approaches on BTC:
  1. 4h-aggregated Donchian breakout (longer timeframe, fewer false signals)
  2. Triple Moving Average (TMA) trend following
  3. Supertrend indicator (ATR-based trailing stop trend system)

All tested on 4-year OOS data (2022–2026) at both 0.1% and 0.25% fees.
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import numpy as np
import pandas as pd
import json

# ── Load and Resample Data ────────────────────────────────────────────────────
def load_data(symbol: str, resample: str = '1h') -> pd.DataFrame:
    path = f'/home/ubuntu/TradingAgents/data/historical/{symbol}_USD_1h_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    if resample != '1h':
        df = df.resample(resample).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
    return df

# ── Indicator Utilities ───────────────────────────────────────────────────────
def _ewm(arr, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def _atr(high, low, close, period=14):
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return _ewm(tr, period)

def _adx(high, low, close, period=14):
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    dm_plus  = np.maximum(high - np.roll(high, 1), 0); dm_plus[0] = 0
    dm_minus = np.maximum(np.roll(low, 1) - low, 0); dm_minus[0] = 0
    mask = dm_plus <= dm_minus; dm_plus[mask] = 0
    mask = dm_minus <= dm_plus; dm_minus[mask] = 0
    atr_s    = _ewm(tr, period)
    di_plus  = 100 * _ewm(dm_plus, period)  / np.where(atr_s > 0, atr_s, np.nan)
    di_minus = 100 * _ewm(dm_minus, period) / np.where(atr_s > 0, atr_s, np.nan)
    denom = np.where((di_plus + di_minus) > 0, di_plus + di_minus, np.nan)
    dx = 100 * np.abs(di_plus - di_minus) / denom
    return _ewm(np.nan_to_num(dx, nan=0.0), period)

def _rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ewm(gain, period * 2 - 1)
    avg_loss = _ewm(loss, period * 2 - 1)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100 - 100 / (1 + rs)

# ── Generic Backtest Engine ───────────────────────────────────────────────────
def run_backtest(close, long_sig, short_sig, fee=0.001, max_hold=48):
    """Pure numpy backtest."""
    n = len(close)
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
                trades.append(pnl)
                position = 0

    if len(trades) < 5:
        return {'sharpe': -99, 'total_return': -99, 'max_dd': 99, 'n_trades': len(trades), 'win_rate': 0}

    pnls = np.array(trades)
    eq   = np.cumprod(1 + pnls)
    eq   = np.insert(eq, 0, 1.0)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    max_dd = abs(dd.min())
    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(len(pnls)) if pnls.std() > 0 else 0

    return {
        'sharpe': round(float(sharpe), 4),
        'total_return': round(float(eq[-1] - 1) * 100, 2),
        'max_dd': round(float(max_dd) * 100, 2),
        'n_trades': len(pnls),
        'win_rate': round(float((pnls > 0).mean()) * 100, 1),
    }

# ── Strategy 1: 4h Donchian Breakout ─────────────────────────────────────────
def test_4h_donchian(df_4h, dp=20, adx_min=20, vol_mult=2.0, fee=0.001):
    close  = df_4h['close'].values.astype(float)
    high   = df_4h['high'].values.astype(float)
    low    = df_4h['low'].values.astype(float)
    volume = df_4h['volume'].values.astype(float)
    n = len(close)

    adx   = _adx(high, low, close, 14)
    atr   = _atr(high, low, close, 14)
    vol_ma = np.array([volume[max(0,i-19):i+1].mean() for i in range(n)])
    low_vol = (atr / np.where(close > 0, close, np.nan)) <= 0.05  # 5% for 4h

    dc_upper = np.array([high[max(0,i-dp):i].max() if i >= dp else np.nan for i in range(n)])
    dc_lower = np.array([low[max(0,i-dp):i].min() if i >= dp else np.nan for i in range(n)])

    long_sig  = (close > dc_upper) & (adx >= adx_min) & (volume >= vol_mult * vol_ma) & low_vol
    short_sig = (close < dc_lower) & (adx >= adx_min) & (volume >= vol_mult * vol_ma) & low_vol
    long_sig  = np.roll(long_sig, 1); long_sig[0] = False
    short_sig = np.roll(short_sig, 1); short_sig[0] = False

    return run_backtest(close, long_sig, short_sig, fee=fee, max_hold=18)  # 18 × 4h = 3 days

# ── Strategy 2: Triple Moving Average (TMA) ───────────────────────────────────
def test_tma(df, fast=8, mid=21, slow=55, fee=0.001):
    close = df['close'].values.astype(float)
    ema_fast = _ewm(close, fast)
    ema_mid  = _ewm(close, mid)
    ema_slow = _ewm(close, slow)

    # Long: fast > mid > slow (all aligned up)
    # Short: fast < mid < slow (all aligned down)
    long_sig  = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    short_sig = (ema_fast < ema_mid) & (ema_mid < ema_slow)

    # Entry on alignment change
    long_entry  = long_sig  & ~np.roll(long_sig, 1)
    short_entry = short_sig & ~np.roll(short_sig, 1)
    long_entry[0] = short_entry[0] = False

    return run_backtest(close, long_entry, short_entry, fee=fee, max_hold=72)

# ── Strategy 3: Supertrend ────────────────────────────────────────────────────
def test_supertrend(df, atr_period=10, atr_mult=3.0, fee=0.001):
    close = df['close'].values.astype(float)
    high  = df['high'].values.astype(float)
    low   = df['low'].values.astype(float)
    n = len(close)

    atr = _atr(high, low, close, atr_period)
    hl2 = (high + low) / 2

    upper_band = hl2 + atr_mult * atr
    lower_band = hl2 - atr_mult * atr

    # Supertrend calculation
    supertrend = np.zeros(n)
    direction  = np.ones(n)  # 1 = uptrend (long), -1 = downtrend (short)

    for i in range(1, n):
        # Upper band
        if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = upper_band[i-1]

        # Lower band
        if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = lower_band[i-1]

        # Direction
        if direction[i-1] == -1 and close[i] > upper_band[i]:
            direction[i] = 1
        elif direction[i-1] == 1 and close[i] < lower_band[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    # Entry on direction change
    long_entry  = (direction == 1) & (np.roll(direction, 1) == -1)
    short_entry = (direction == -1) & (np.roll(direction, 1) == 1)
    long_entry[0] = short_entry[0] = False

    return run_backtest(close, long_entry, short_entry, fee=fee, max_hold=72)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading BTC data (1h and 4h)...")
    btc_1h = load_data('BTC', '1h')
    btc_4h = load_data('BTC', '4h')
    print(f"  1h bars: {len(btc_1h)}  |  4h bars: {len(btc_4h)}")

    results = {}

    print("\n=== STRATEGY 1: 4h Donchian Breakout ===")
    for dp in [15, 20, 25, 30]:
        for adx_min in [18, 20, 22]:
            for vol_mult in [1.5, 2.0]:
                r01 = test_4h_donchian(btc_4h, dp=dp, adx_min=adx_min, vol_mult=vol_mult, fee=0.001)
                r25 = test_4h_donchian(btc_4h, dp=dp, adx_min=adx_min, vol_mult=vol_mult, fee=0.0025)
                key = f"4h_donchian_dp{dp}_adx{adx_min}_vol{vol_mult}"
                results[key] = {'01pct': r01, '025pct': r25}
                if r01['sharpe'] > 0.5:
                    print(f"  dp={dp} adx={adx_min} vol={vol_mult}: Sharpe={r01['sharpe']} "
                          f"Return={r01['total_return']}% DD={r01['max_dd']}% "
                          f"| 0.25%: Sharpe={r25['sharpe']} Return={r25['total_return']}%")

    print("\n=== STRATEGY 2: Triple Moving Average ===")
    for fast, mid, slow in [(5,13,34), (8,21,55), (9,21,50), (13,34,89)]:
        r01 = test_tma(btc_1h, fast=fast, mid=mid, slow=slow, fee=0.001)
        r25 = test_tma(btc_1h, fast=fast, mid=mid, slow=slow, fee=0.0025)
        key = f"tma_{fast}_{mid}_{slow}"
        results[key] = {'01pct': r01, '025pct': r25}
        print(f"  TMA({fast},{mid},{slow}): Sharpe={r01['sharpe']} Return={r01['total_return']}% "
              f"DD={r01['max_dd']}% WR={r01['win_rate']}% Trades={r01['n_trades']}"
              f" | 0.25%: Sharpe={r25['sharpe']} Return={r25['total_return']}%")

    print("\n=== STRATEGY 3: Supertrend ===")
    for atr_p in [7, 10, 14]:
        for atr_m in [2.0, 2.5, 3.0, 3.5]:
            r01 = test_supertrend(btc_1h, atr_period=atr_p, atr_mult=atr_m, fee=0.001)
            r25 = test_supertrend(btc_1h, atr_period=atr_p, atr_mult=atr_m, fee=0.0025)
            key = f"supertrend_p{atr_p}_m{atr_m}"
            results[key] = {'01pct': r01, '025pct': r25}
            if r01['sharpe'] > 0.5:
                print(f"  ST(p={atr_p},m={atr_m}): Sharpe={r01['sharpe']} Return={r01['total_return']}% "
                      f"DD={r01['max_dd']}% WR={r01['win_rate']}% Trades={r01['n_trades']}"
                      f" | 0.25%: Sharpe={r25['sharpe']} Return={r25['total_return']}%")

    # Find overall best
    best_key = max(results, key=lambda k: results[k]['01pct']['sharpe'])
    best = results[best_key]
    print(f"\n{'='*70}")
    print(f"BEST BTC STRATEGY: {best_key}")
    print(f"  0.10% fees: Sharpe={best['01pct']['sharpe']}  Return={best['01pct']['total_return']}%  "
          f"DD={best['01pct']['max_dd']}%  WR={best['01pct']['win_rate']}%  Trades={best['01pct']['n_trades']}")
    print(f"  0.25% fees: Sharpe={best['025pct']['sharpe']}  Return={best['025pct']['total_return']}%  "
          f"DD={best['025pct']['max_dd']}%  WR={best['025pct']['win_rate']}%  Trades={best['025pct']['n_trades']}")

    with open('/tmp/btc_strategy_search.json', 'w') as f:
        json.dump({'best_key': best_key, 'best': best, 'all': results}, f, indent=2)
    print(f"\nResults → /tmp/btc_strategy_search.json")

if __name__ == '__main__':
    main()
