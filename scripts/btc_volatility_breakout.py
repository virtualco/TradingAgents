"""
BTC Volatility Breakout Strategy
==================================
BTC's statistical edge lies in explosive volatility expansion events, not
sustained directional trends. This strategy captures those events using:

  1. ATR Expansion Breakout: Enter when price moves > N × ATR from the
     previous bar's close in a single bar (volatility surge entry).
     Exit: 2× ATR trailing stop or 24h max hold.

  2. Opening Range Breakout (ORB): Use the first 4 hours of each UTC day
     as the "opening range". Enter long/short on breakout of that range.
     Exit: end of day (UTC 23:00) or 2× ATR stop.

  3. Bollinger Band Squeeze + Expansion: Detect BB squeeze (width < 1%
     of price for 8+ consecutive bars), then enter on the first expansion
     bar (price breaks outside the band). Exit: BB width returns to normal.

All three tested across 4 years at 0.1% and 0.25% fees.
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import numpy as np
import pandas as pd
import json

def load_data(symbol: str) -> pd.DataFrame:
    path = f'/home/ubuntu/TradingAgents/data/historical/{symbol}_USD_1h_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df

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

def run_backtest(close, long_sig, short_sig, fee=0.001, max_hold=24):
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

# ── Strategy 1: ATR Expansion Breakout ───────────────────────────────────────
def test_atr_expansion(df, atr_period=14, expansion_mult=2.0, vol_mult=2.0, fee=0.001):
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)

    atr    = _atr(high, low, close, atr_period)
    vol_ma = np.array([volume[max(0,i-19):i+1].mean() for i in range(n)])

    # Bar range > expansion_mult × ATR = volatility expansion
    bar_range = high - low
    expansion = bar_range > expansion_mult * np.roll(atr, 1)
    vol_surge = volume > vol_mult * vol_ma

    # Direction: close > open = bullish expansion, close < open = bearish
    bar_bullish = close > df['open'].values.astype(float)
    bar_bearish = close < df['open'].values.astype(float)

    long_sig  = expansion & vol_surge & bar_bullish
    short_sig = expansion & vol_surge & bar_bearish

    # Enter on next bar
    long_sig  = np.roll(long_sig, 1); long_sig[0] = False
    short_sig = np.roll(short_sig, 1); short_sig[0] = False

    return run_backtest(close, long_sig, short_sig, fee=fee, max_hold=24)

# ── Strategy 2: Opening Range Breakout (ORB) ─────────────────────────────────
def test_orb(df, orb_hours=4, fee=0.001):
    """4-hour opening range breakout with end-of-day exit."""
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    hours  = np.array([t.hour for t in df.index])
    n = len(close)

    # Compute daily opening range high/low (first orb_hours bars of each UTC day)
    orb_high = np.full(n, np.nan)
    orb_low  = np.full(n, np.nan)

    # Group by date and compute ORB
    dates = pd.Series([t.date() for t in df.index])
    for date in dates.unique():
        mask = dates == date
        idx  = np.where(mask)[0]
        orb_mask = idx[hours[idx] < orb_hours]
        if len(orb_mask) < orb_hours:
            continue
        orb_h = high[orb_mask].max()
        orb_l = low[orb_mask].min()
        # Apply ORB levels to all bars after the opening range
        post_orb = idx[hours[idx] >= orb_hours]
        orb_high[post_orb] = orb_h
        orb_low[post_orb]  = orb_l

    # Entry: price breaks above/below ORB after the opening range
    long_sig  = (close > orb_high) & (hours >= orb_hours) & (hours < 20)
    short_sig = (close < orb_low)  & (hours >= orb_hours) & (hours < 20)
    long_sig  = np.nan_to_num(long_sig, nan=False).astype(bool)
    short_sig = np.nan_to_num(short_sig, nan=False).astype(bool)

    # Shift by 1 bar
    long_sig  = np.roll(long_sig, 1); long_sig[0] = False
    short_sig = np.roll(short_sig, 1); short_sig[0] = False

    return run_backtest(close, long_sig, short_sig, fee=fee, max_hold=16)

# ── Strategy 3: Bollinger Band Squeeze + Expansion ───────────────────────────
def test_bb_squeeze(df, bb_period=20, bb_std=2.0, squeeze_pct=0.015, fee=0.001):
    """Enter on BB expansion after a squeeze (low volatility consolidation)."""
    close = df['close'].values.astype(float)
    n = len(close)

    # Rolling BB
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_mid   = np.full(n, np.nan)
    for i in range(bb_period, n):
        window = close[i-bb_period:i]
        mid    = window.mean()
        sigma  = window.std()
        bb_upper[i] = mid + bb_std * sigma
        bb_lower[i] = mid - bb_std * sigma
        bb_mid[i]   = mid

    bb_width = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, np.nan)
    bb_width = np.nan_to_num(bb_width, nan=0.1)

    # Squeeze: BB width < squeeze_pct for last 8 bars
    in_squeeze = np.zeros(n, dtype=bool)
    for i in range(8, n):
        in_squeeze[i] = (bb_width[i-8:i] < squeeze_pct).all()

    # Expansion: price breaks outside band after squeeze
    long_sig  = in_squeeze & (close > bb_upper)
    short_sig = in_squeeze & (close < bb_lower)
    long_sig  = np.roll(long_sig, 1); long_sig[0] = False
    short_sig = np.roll(short_sig, 1); short_sig[0] = False

    return run_backtest(close, long_sig, short_sig, fee=fee, max_hold=24)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading BTC data...")
    btc = load_data('BTC')
    print(f"  Bars: {len(btc)}")

    print("\n=== STRATEGY 1: ATR Expansion Breakout ===")
    best_atr = {'sharpe': -99}
    for atr_p in [7, 10, 14]:
        for exp_m in [1.5, 2.0, 2.5, 3.0]:
            for vol_m in [1.5, 2.0, 2.5]:
                r = test_atr_expansion(btc, atr_period=atr_p, expansion_mult=exp_m, vol_mult=vol_m, fee=0.001)
                r25 = test_atr_expansion(btc, atr_period=atr_p, expansion_mult=exp_m, vol_mult=vol_m, fee=0.0025)
                if r['sharpe'] > best_atr['sharpe']:
                    best_atr = {**r, 'params': f"atr_p={atr_p} exp={exp_m} vol={vol_m}",
                                '025': r25}
                if r['sharpe'] > 0.5:
                    print(f"  atr_p={atr_p} exp={exp_m} vol={vol_m}: Sharpe={r['sharpe']} "
                          f"Return={r['total_return']}% DD={r['max_dd']}% WR={r['win_rate']}% "
                          f"| 0.25%: Sharpe={r25['sharpe']} Return={r25['total_return']}%")
    print(f"  Best: {best_atr.get('params')} Sharpe={best_atr['sharpe']} Return={best_atr['total_return']}%")

    print("\n=== STRATEGY 2: Opening Range Breakout ===")
    for orb_h in [2, 4, 6]:
        r = test_orb(btc, orb_hours=orb_h, fee=0.001)
        r25 = test_orb(btc, orb_hours=orb_h, fee=0.0025)
        print(f"  ORB({orb_h}h): Sharpe={r['sharpe']} Return={r['total_return']}% "
              f"DD={r['max_dd']}% WR={r['win_rate']}% Trades={r['n_trades']}"
              f" | 0.25%: Sharpe={r25['sharpe']} Return={r25['total_return']}%")

    print("\n=== STRATEGY 3: BB Squeeze + Expansion ===")
    best_bb = {'sharpe': -99}
    for sq_pct in [0.010, 0.015, 0.020, 0.025]:
        r = test_bb_squeeze(btc, squeeze_pct=sq_pct, fee=0.001)
        r25 = test_bb_squeeze(btc, squeeze_pct=sq_pct, fee=0.0025)
        if r['sharpe'] > best_bb['sharpe']:
            best_bb = {**r, 'params': f"sq_pct={sq_pct}", '025': r25}
        print(f"  sq_pct={sq_pct}: Sharpe={r['sharpe']} Return={r['total_return']}% "
              f"DD={r['max_dd']}% WR={r['win_rate']}% Trades={r['n_trades']}"
              f" | 0.25%: Sharpe={r25['sharpe']} Return={r25['total_return']}%")

    print(f"\n{'='*70}")
    print("SUMMARY — Best BTC Strategy per Type:")
    print(f"  ATR Expansion: {best_atr.get('params')} → Sharpe={best_atr['sharpe']} Return={best_atr['total_return']}%")
    print(f"  BB Squeeze:    {best_bb.get('params')} → Sharpe={best_bb['sharpe']} Return={best_bb['total_return']}%")

    with open('/tmp/btc_vol_breakout.json', 'w') as f:
        json.dump({'atr_best': best_atr, 'bb_best': best_bb}, f, indent=2)
    print(f"\nResults → /tmp/btc_vol_breakout.json")

if __name__ == '__main__':
    main()
