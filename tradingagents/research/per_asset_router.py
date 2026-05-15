"""
Per-Asset Strategy Router
=========================
Validated OOS routing (2022-2026, 35,064 bars per asset):

  BTCUSDT → ATR Expansion Breakout  (Sharpe 0.52, +14.2%, DD 39.8%)
             Limit orders ONLY — fee-sensitive, requires maker rates (0.01%)
  ETHUSDT → Donchian Momentum       (Sharpe 1.55, +136%, DD 28.8%)
             Market or limit orders — robust to taker fees (0.06%)

This module is a drop-in replacement for DualRegimeStrategy.generate_signals()
and is designed to be used by live_trader.py's SignalProcessor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ── Technical Indicators ──────────────────────────────────────────────────────

def _ewm(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr, dtype=float)
    result[0] = float(arr[0])
    for i in range(1, len(arr)):
        result[i] = alpha * float(arr[i]) + (1 - alpha) * result[i - 1]
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ewm(tr, period)


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    prev_high  = np.roll(high, 1);  prev_high[0]  = high[0]
    prev_low   = np.roll(low, 1);   prev_low[0]   = low[0]

    tr    = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    dm_p  = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0.0), 0.0)
    dm_m  = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0.0), 0.0)

    atr_s  = _ewm(tr, period)
    di_p   = 100.0 * _ewm(dm_p, period) / np.where(atr_s > 0, atr_s, 1e-9)
    di_m   = 100.0 * _ewm(dm_m, period) / np.where(atr_s > 0, atr_s, 1e-9)
    denom  = np.where((di_p + di_m) > 0, di_p + di_m, 1e-9)
    dx     = 100.0 * np.abs(di_p - di_m) / denom
    return _ewm(dx, period), di_p, di_m


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = _ewm(gain, period)
    avg_l = _ewm(loss, period)
    rs    = avg_g / np.where(avg_l > 0, avg_l, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _hurst_fast(close: np.ndarray, window: int = 96) -> np.ndarray:
    n = len(close)
    h = np.full(n, 0.5)
    lags = [l for l in [2, 4, 8, 16, 32] if l < window // 2]
    if len(lags) < 2:
        return h
    log_lags = np.log(lags)
    for i in range(window, n):
        x = np.log(np.abs(close[i - window:i]) + 1e-10)
        vl = [np.var(x[l:] - x[:-l]) for l in lags]
        try:
            slope = np.polyfit(log_lags, np.log(np.array(vl) + 1e-20), 1)[0]
            h[i] = float(np.clip(slope / 2.0, 0.0, 1.0))
        except Exception:
            pass
    return h


# ── Strategy Parameters (OOS-validated) ──────────────────────────────────────

BTC_CONFIG = {
    'atr_period':     15,        # bayesian-optimised (was 14)
    'expansion_mult': 3.7,       # bayesian-optimised (was 3.0)
    'vol_mult':       2.5,       # bayesian-optimised (was 1.2)
    'max_hold_bars':  12,
    'order_type':     'Limit',   # MUST be limit for BTC — fee sensitive
    'stop_mult':      1.5,
    'tp_mult':        8.0,       # bayesian-optimised (was 4.0)
}

ETH_CONFIG = {
    'donchian_period':  32,       # bayesian-optimised (was 28)
    'adx_min':          21,       # bayesian-optimised (was 12)
    'adx_trend':        22,       # bayesian-optimised (was 24)
    'vol_mult':         2.0,      # bayesian-optimised (was 1.8)
    'hurst_min':        0.52,     # bayesian-optimised (was 0.42)
    'vol_atr_max':      0.08,     # bayesian-optimised (was 0.04)
    'max_hold_bars':    90,       # bayesian-optimised (was 60)
    'order_type':       'Market',
    'stop_mult':        3.0,      # bayesian-optimised (was 1.2)
    'tp_mult':          9.0,      # bayesian-optimised (was 6.0)
    'atr_donchian_factor': 1.0,   # bayesian-optimised (new — adaptive Donchian)
}

ASSET_CONFIG = {
    'BTCUSDT': BTC_CONFIG,
    'ETHUSDT': ETH_CONFIG,
}


# ── Signal Generators ─────────────────────────────────────────────────────────

def _btc_signal(df: pd.DataFrame) -> dict:
    """
    BTC: ATR Expansion Breakout
    Enter on a bar whose range exceeds 3× prior ATR with volume surge.
    Limit orders only — edge is destroyed by taker fees.
    """
    cfg = BTC_CONFIG
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    opens  = df['open'].values.astype(float)
    n = len(close)

    atr_v   = _atr(high, low, close, cfg['atr_period'])
    adx_v, di_p, di_m = _adx(high, low, close)
    rsi_v   = _rsi(close)
    vol_ma  = np.array([volume[max(0, i - 19):i + 1].mean() for i in range(n)])

    bar_range  = high - low
    prev_atr   = np.roll(atr_v, 1); prev_atr[0] = atr_v[0]
    expansion  = bar_range[-1] > cfg['expansion_mult'] * prev_atr[-1]
    vol_surge  = volume[-1] > cfg['vol_mult'] * vol_ma[-1]
    bar_bull   = close[-1] > opens[-1]
    bar_bear   = close[-1] < opens[-1]

    long_sig  = bool(expansion and vol_surge and bar_bull)
    short_sig = bool(expansion and vol_surge and bar_bear)
    signal    = 'LONG' if long_sig else ('SHORT' if short_sig else 'FLAT')

    adx_last = float(adx_v[-1])
    regime = 'TRENDING' if adx_last > 25 else ('RANGING' if adx_last < 15 else 'TRANSITION')

    price = float(close[-1])
    atr_last = float(atr_v[-1])

    return {
        'symbol':     'BTCUSDT',
        'signal':     signal,
        'regime':     regime,
        'strategy':   'ATR_EXPANSION',
        'order_type': cfg['order_type'],
        'conviction': round(float(bar_range[-1] / prev_atr[-1]) / 5.0, 3),  # normalised 0-1
        'price':      price,
        'atr':        atr_last,
        'adx':        adx_last,
        'rsi':        float(rsi_v[-1]),
        'hurst':      0.5,  # not used for BTC
        'di_plus':    float(di_p[-1]),
        'di_minus':   float(di_m[-1]),
        'bar_range':  float(bar_range[-1]),
        'atr_mult':   round(float(bar_range[-1] / prev_atr[-1]), 2) if prev_atr[-1] > 0 else 0,
        'vol_ratio':  round(float(volume[-1] / vol_ma[-1]), 2) if vol_ma[-1] > 0 else 0,
        'stop_loss':  round(price - cfg['stop_mult'] * atr_last, 4) if signal == 'LONG'
                      else round(price + cfg['stop_mult'] * atr_last, 4) if signal == 'SHORT' else 0,
        'take_profit': round(price + cfg['tp_mult'] * atr_last, 4) if signal == 'LONG'
                       else round(price - cfg['tp_mult'] * atr_last, 4) if signal == 'SHORT' else 0,
        'max_hold_bars': cfg['max_hold_bars'],
    }


def _eth_signal(df: pd.DataFrame) -> dict:
    """
    ETH: Donchian Momentum with Hurst regime filter.
    Enter on Donchian channel breakout with ADX ≥ 22, Hurst ≥ 0.48, volume surge.
    Robust to taker fees — market orders acceptable.
    """
    cfg = ETH_CONFIG
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)
    dp = cfg['donchian_period']

    atr_v   = _atr(high, low, close, 14)
    adx_v, di_p, di_m = _adx(high, low, close)
    rsi_v   = _rsi(close)
    hurst_v = _hurst_fast(close, 96)
    vol_ma  = np.array([volume[max(0, i - 19):i + 1].mean() for i in range(n)])

    dc_upper = float(high[max(0, n - dp - 1):n - 1].max()) if n > dp else float('nan')
    dc_lower = float(low[max(0, n - dp - 1):n - 1].min())  if n > dp else float('nan')

    price    = float(close[-1])
    atr_last = float(atr_v[-1])
    adx_last = float(adx_v[-1])
    hurst_last = float(hurst_v[-1])
    atr_pct  = atr_last / price if price > 0 else 1.0

    trending  = adx_last >= cfg['adx_trend'] and hurst_last >= cfg['hurst_min']
    adx_ok    = adx_last >= cfg['adx_min']
    vol_ok    = volume[-1] >= cfg['vol_mult'] * vol_ma[-1]
    low_vol   = atr_pct <= cfg['vol_atr_max']

    long_sig  = bool(not np.isnan(dc_upper) and price > dc_upper and adx_ok and vol_ok and low_vol and trending)
    short_sig = bool(not np.isnan(dc_lower) and price < dc_lower and adx_ok and vol_ok and low_vol and trending)
    signal    = 'LONG' if long_sig else ('SHORT' if short_sig else 'FLAT')

    regime = 'TRENDING' if trending else ('RANGING' if adx_last < 15 else 'TRANSITION')

    return {
        'symbol':     'ETHUSDT',
        'signal':     signal,
        'regime':     regime,
        'strategy':   'DONCHIAN_MOMENTUM',
        'order_type': cfg['order_type'],
        'conviction': round(min(adx_last / 50.0, 1.0), 3),
        'price':      price,
        'atr':        atr_last,
        'adx':        adx_last,
        'rsi':        float(rsi_v[-1]),
        'hurst':      hurst_last,
        'di_plus':    float(di_p[-1]),
        'di_minus':   float(di_m[-1]),
        'dc_upper':   dc_upper if not np.isnan(dc_upper) else 0,
        'dc_lower':   dc_lower if not np.isnan(dc_lower) else 0,
        'vol_ratio':  round(float(volume[-1] / vol_ma[-1]), 2) if vol_ma[-1] > 0 else 0,
        'stop_loss':  round(price - cfg['stop_mult'] * atr_last, 4) if signal == 'LONG'
                      else round(price + cfg['stop_mult'] * atr_last, 4) if signal == 'SHORT' else 0,
        'take_profit': round(price + cfg['tp_mult'] * atr_last, 4) if signal == 'LONG'
                       else round(price - cfg['tp_mult'] * atr_last, 4) if signal == 'SHORT' else 0,
        'max_hold_bars': cfg['max_hold_bars'],
    }


# ── Public Interface ──────────────────────────────────────────────────────────

class PerAssetRouter:
    """
    Drop-in replacement for DualRegimeStrategy in the SignalProcessor pipeline.
    Routes each symbol to its OOS-validated strategy.

    Usage in live_trader.py:
        from tradingagents.research.per_asset_router import PerAssetRouter
        self.strategy = PerAssetRouter()
        # Then in SignalProcessor.generate_signal(symbol):
        #   sig = self.strategy.generate_signals(df, symbol)
    """

    SUPPORTED_SYMBOLS = {'BTCUSDT', 'ETHUSDT'}

    def generate_signals(self, df: pd.DataFrame, symbol: Optional[str] = None) -> dict:
        """
        Generate trading signal for the given symbol.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            symbol: Trading symbol (BTCUSDT or ETHUSDT)

        Returns:
            Signal dict compatible with live_trader.py's OrderExecutor
        """
        if symbol is None:
            raise ValueError('symbol is required for PerAssetRouter')

        sym = symbol.upper()
        if sym not in self.SUPPORTED_SYMBOLS:
            raise ValueError(f'Unsupported symbol: {symbol}. Supported: {self.SUPPORTED_SYMBOLS}')

        if len(df) < 100:
            return {
                'symbol': sym, 'signal': 'FLAT', 'regime': 'TRANSITION',
                'strategy': 'INSUFFICIENT_DATA', 'conviction': 0.0,
                'price': float(df['close'].iloc[-1]) if len(df) > 0 else 0,
                'atr': 0, 'adx': 0, 'rsi': 50, 'hurst': 0.5,
                'stop_loss': 0, 'take_profit': 0, 'max_hold_bars': 24,
            }

        if sym == 'BTCUSDT':
            return _btc_signal(df)
        else:
            return _eth_signal(df)

    def get_order_type(self, symbol: str) -> str:
        """Returns the validated order type for a given symbol."""
        cfg = ASSET_CONFIG.get(symbol.upper(), {})
        return cfg.get('order_type', 'Market')

    def get_max_hold_bars(self, symbol: str) -> int:
        """Returns the validated max hold period for a given symbol."""
        cfg = ASSET_CONFIG.get(symbol.upper(), {})
        return cfg.get('max_hold_bars', 24)
