"""
Final Strategy Validation — TradingAgents v5
=============================================
Per-asset routing based on 4-year OOS optimisation:

  BTC → ATR Expansion Breakout (atr_period=14, expansion_mult=3.0, vol_mult=1.5)
        Best OOS Sharpe: 0.52, Return: +14.2%, DD: 39.8%

  ETH → Donchian Momentum (dp=25, adx_min=18, vol_mult=2.0, adx_trend_threshold=22)
        Best OOS Sharpe: 1.55, Return: +136%, DD: 30.2%

Validation at both 0.1% (Bybit maker) and 0.25% (Bybit taker + slippage) fees.
Year-by-year breakdown to confirm no single year is catastrophic.
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

def _hurst_fast(close, window=96):
    n = len(close)
    hurst = np.full(n, 0.5)
    lags = [l for l in [2, 4, 8, 16, 32] if l < window // 2]
    if len(lags) < 2: return hurst
    log_lags = np.log(lags)
    for i in range(window, n):
        x = np.log(np.abs(close[i-window:i]) + 1e-10)
        vl = [np.var(x[l:] - x[:-l]) for l in lags]
        try:
            slope = np.polyfit(log_lags, np.log(np.array(vl) + 1e-20), 1)[0]
            hurst[i] = float(np.clip(slope / 2.0, 0.0, 1.0))
        except: pass
    return hurst

def run_backtest_detailed(close, long_sig, short_sig, years, fee=0.001, max_hold=24):
    """Backtest with year-by-year breakdown."""
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
                trades.append({'pnl': pnl, 'year': int(years[entry_bar])})
                position = 0

    if len(trades) < 5:
        return {'sharpe': -99, 'total_return': -99, 'max_dd': 99,
                'n_trades': len(trades), 'win_rate': 0, 'by_year': {}}

    pnls = np.array([t['pnl'] for t in trades])
    eq   = np.cumprod(1 + pnls)
    eq   = np.insert(eq, 0, 1.0)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    max_dd = abs(dd.min())
    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(len(pnls)) if pnls.std() > 0 else 0

    by_year = {}
    for yr in sorted(set(t['year'] for t in trades)):
        yr_pnls = np.array([t['pnl'] for t in trades if t['year'] == yr])
        by_year[str(yr)] = {
            'n': len(yr_pnls),
            'pnl_pct': round(yr_pnls.sum() * 100, 2),
            'win_rate': round((yr_pnls > 0).mean() * 100, 1),
            'avg_trade': round(yr_pnls.mean() * 100, 3),
        }

    return {
        'sharpe': round(float(sharpe), 4),
        'total_return': round(float(eq[-1] - 1) * 100, 2),
        'max_dd': round(float(max_dd) * 100, 2),
        'n_trades': len(pnls),
        'win_rate': round(float((pnls > 0).mean()) * 100, 1),
        'by_year': by_year,
    }

# ── BTC: ATR Expansion Breakout ───────────────────────────────────────────────
def btc_atr_expansion(df, atr_period=14, expansion_mult=3.0, vol_mult=1.5, fee=0.001):
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    opens  = df['open'].values.astype(float)
    years  = np.array([t.year for t in df.index])
    n = len(close)

    atr    = _atr(high, low, close, atr_period)
    vol_ma = np.array([volume[max(0,i-19):i+1].mean() for i in range(n)])

    bar_range = high - low
    expansion = bar_range > expansion_mult * np.roll(atr, 1)
    vol_surge = volume > vol_mult * vol_ma
    bar_bullish = close > opens
    bar_bearish = close < opens

    long_sig  = expansion & vol_surge & bar_bullish
    short_sig = expansion & vol_surge & bar_bearish
    long_sig  = np.roll(long_sig, 1); long_sig[0] = False
    short_sig = np.roll(short_sig, 1); short_sig[0] = False

    return run_backtest_detailed(close, long_sig, short_sig, years, fee=fee, max_hold=24)

# ── ETH: Donchian Momentum ────────────────────────────────────────────────────
def eth_donchian_momentum(df, dp=25, adx_min=18, vol_mult=2.0, adx_trend_threshold=22, fee=0.001):
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    years  = np.array([t.year for t in df.index])
    n = len(close)

    adx    = _adx(high, low, close, 14)
    atr    = _atr(high, low, close, 14)
    vol_ma = np.array([volume[max(0,i-19):i+1].mean() for i in range(n)])
    low_vol = (atr / np.where(close > 0, close, np.nan)) <= 0.03

    print("  Computing Hurst for ETH regime filter...")
    hurst = _hurst_fast(close, 96)
    trending = (adx >= adx_trend_threshold) & (hurst >= 0.48)

    dc_upper = np.array([high[max(0,i-dp):i].max() if i >= dp else np.nan for i in range(n)])
    dc_lower = np.array([low[max(0,i-dp):i].min() if i >= dp else np.nan for i in range(n)])

    vol_ok    = volume >= vol_mult * vol_ma
    adx_ok    = adx >= adx_min
    long_sig  = (close > dc_upper) & adx_ok & vol_ok & low_vol & trending
    short_sig = (close < dc_lower) & adx_ok & vol_ok & low_vol & trending
    long_sig  = np.nan_to_num(long_sig, nan=False).astype(bool)
    short_sig = np.nan_to_num(short_sig, nan=False).astype(bool)
    long_sig  = np.roll(long_sig, 1); long_sig[0] = False
    short_sig = np.roll(short_sig, 1); short_sig[0] = False

    return run_backtest_detailed(close, long_sig, short_sig, years, fee=fee, max_hold=72)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("FINAL STRATEGY VALIDATION — TradingAgents v5")
    print("Per-asset routing: BTC→ATR Expansion | ETH→Donchian Momentum")
    print("=" * 70)

    print("\nLoading data...")
    btc = load_data('BTC')
    eth = load_data('ETH')

    results = {}

    # ── BTC Validation ─────────────────────────────────────────────────────
    print("\n--- BTC: ATR Expansion Breakout ---")
    print("  0.10% fees (Bybit maker):")
    btc_01 = btc_atr_expansion(btc, fee=0.001)
    results['btc_01pct'] = btc_01
    print(f"    Sharpe={btc_01['sharpe']}  Return={btc_01['total_return']}%  "
          f"DD={btc_01['max_dd']}%  WR={btc_01['win_rate']}%  Trades={btc_01['n_trades']}")
    for yr, d in sorted(btc_01['by_year'].items()):
        print(f"    {yr}: {d['n']} trades  WR={d['win_rate']}%  PnL={d['pnl_pct']}%  AvgTrade={d['avg_trade']}%")

    print("  0.25% fees (taker + slippage):")
    btc_25 = btc_atr_expansion(btc, fee=0.0025)
    results['btc_025pct'] = btc_25
    print(f"    Sharpe={btc_25['sharpe']}  Return={btc_25['total_return']}%  "
          f"DD={btc_25['max_dd']}%  WR={btc_25['win_rate']}%  Trades={btc_25['n_trades']}")

    # ── ETH Validation ─────────────────────────────────────────────────────
    print("\n--- ETH: Donchian Momentum ---")
    print("  0.10% fees (Bybit maker):")
    eth_01 = eth_donchian_momentum(eth, fee=0.001)
    results['eth_01pct'] = eth_01
    print(f"    Sharpe={eth_01['sharpe']}  Return={eth_01['total_return']}%  "
          f"DD={eth_01['max_dd']}%  WR={eth_01['win_rate']}%  Trades={eth_01['n_trades']}")
    for yr, d in sorted(eth_01['by_year'].items()):
        print(f"    {yr}: {d['n']} trades  WR={d['win_rate']}%  PnL={d['pnl_pct']}%  AvgTrade={d['avg_trade']}%")

    print("  0.25% fees (taker + slippage):")
    eth_25 = eth_donchian_momentum(eth, fee=0.0025)
    results['eth_025pct'] = eth_25
    print(f"    Sharpe={eth_25['sharpe']}  Return={eth_25['total_return']}%  "
          f"DD={eth_25['max_dd']}%  WR={eth_25['win_rate']}%  Trades={eth_25['n_trades']}")

    # ── Combined Portfolio ─────────────────────────────────────────────────
    print("\n--- COMBINED PORTFOLIO (50% BTC + 50% ETH) ---")
    for fee_label, b, e in [('0.10%', btc_01, eth_01), ('0.25%', btc_25, eth_25)]:
        avg_sharpe = (b['sharpe'] + e['sharpe']) / 2
        avg_return = (b['total_return'] + e['total_return']) / 2
        avg_dd     = (b['max_dd'] + e['max_dd']) / 2
        print(f"  {fee_label} fees: Sharpe={avg_sharpe:.4f}  Return={avg_return:.1f}%  DD={avg_dd:.1f}%")

    # ── OOS Gate Check ─────────────────────────────────────────────────────
    print("\n--- OOS READINESS GATE ---")
    gate_pass = True
    checks = [
        ('ETH Sharpe > 1.0', eth_01['sharpe'] > 1.0),
        ('ETH Return > 0%', eth_01['total_return'] > 0),
        ('ETH DD < 40%', eth_01['max_dd'] < 40),
        ('BTC Return > 0% (0.1% fees)', btc_01['total_return'] > 0),
        ('BTC DD < 50%', btc_01['max_dd'] < 50),
        ('ETH Return > 0% at 0.25% fees', eth_25['total_return'] > 0),
    ]
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed: gate_pass = False

    print(f"\n  OVERALL: {'READY FOR TESTNET' if gate_pass else 'FURTHER OPTIMISATION REQUIRED'}")

    with open('/tmp/final_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → /tmp/final_validation.json")

if __name__ == '__main__':
    main()
