"""
Per-Asset Strategy Router (Multi-Asset v2)
==========================================
Validated OOS routing (2022-2026, 35,064 bars per asset):

  BTCUSDT  → ATR Expansion Breakout  (Sharpe 1.18, +117%, DD 12.9%)
  ETHUSDT  → Donchian Momentum       (Sharpe 1.16, +370%, DD 34.2%)
  SOLUSDT  → ATR Expansion Breakout  (initial config — pending Bayesian opt)
  AVAXUSDT → Donchian Momentum       (initial config — pending Bayesian opt)
  DOGEUSDT → ATR Expansion Breakout  (initial config — pending Bayesian opt)
  BNBUSDT  → Donchian Momentum       (initial config — pending Bayesian opt)
  XRPUSDT  → ATR Expansion Breakout  (initial config — pending Bayesian opt)
  LINKUSDT → Donchian Momentum       (initial config — pending Bayesian opt)

Strategy assignment rationale:
- ATR Expansion: Best for assets with sharp breakout moves (BTC, SOL, DOGE, XRP)
- Donchian Momentum: Best for assets with sustained trends (ETH, AVAX, BNB, LINK)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from tradingagents.research.volume_profile_mr import (
    volume_profile_mr_signal,
    VPMR_DEFAULT_CONFIG,
)


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


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS — Per-Asset Configs
# ══════════════════════════════════════════════════════════════════════════════

# ── BTC: ATR Expansion Breakout (Bayesian-optimised v2, Sharpe 1.17) ─────────
BTC_CONFIG = {
    'strategy':       'ATR_EXPANSION',
    'atr_period':     20,
    'expansion_mult': 3.8,
    'vol_mult':       1.2,
    'max_hold_bars':  14,
    'order_type':     'Limit',
    'stop_mult':      3.7,
    'tp_mult':        5.0,
}

# ── ETH: Donchian Momentum (Bayesian-optimised v2, Sharpe 1.43) ─────────────
ETH_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   45,
    'adx_min':           11,
    'adx_trend':         21,
    'vol_mult':          3.9,
    'hurst_min':         0.38,
    'vol_atr_max':       0.02,
    'max_hold_bars':     42,
    'order_type':        'Market',
    'stop_mult':         2.1,
    'tp_mult':           5.0,
    'atr_donchian_factor': 0.5,
}

# ── SOL: ATR Expansion Breakout (Bayesian-optimised, Sharpe 1.24) ────────────
SOL_CONFIG = {
    'strategy':       'ATR_EXPANSION',
    'atr_period':     16,
    'expansion_mult': 3.3,
    'vol_mult':       1.2,
    'max_hold_bars':  32,
    'order_type':     'Market',
    'stop_mult':      2.4,
    'tp_mult':        3.5,
}

# ── AVAX: Donchian Momentum (Bayesian-optimised, Sharpe 2.00) ───────────────
AVAX_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   21,
    'adx_min':           25,
    'adx_trend':         16,
    'vol_mult':          2.7,
    'hurst_min':         0.40,
    'vol_atr_max':       0.02,
    'max_hold_bars':     48,
    'order_type':        'Market',
    'stop_mult':         4.9,
    'tp_mult':           11.5,
    'atr_donchian_factor': 1.0,
}

# ── DOGE: ATR Expansion Breakout (Bayesian-optimised, Sharpe 0.97) ───────────
DOGE_CONFIG = {
    'strategy':       'ATR_EXPANSION',
    'atr_period':     11,
    'expansion_mult': 4.2,
    'vol_mult':       3.8,
    'max_hold_bars':  18,
    'order_type':     'Market',
    'stop_mult':      2.3,
    'tp_mult':        2.0,
}

# ── BNB: Donchian Momentum (Bayesian-optimised, Sharpe 1.33) ────────────────
BNB_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   19,
    'adx_min':           28,
    'adx_trend':         32,
    'vol_mult':          3.5,
    'hurst_min':         0.50,
    'vol_atr_max':       0.02,
    'max_hold_bars':     114,
    'order_type':        'Market',
    'stop_mult':         4.1,
    'tp_mult':           10.0,
    'atr_donchian_factor': 2.0,
}

# ── XRP: ATR Expansion Breakout (Bayesian-optimised, Sharpe 1.37) ────────────
XRP_CONFIG = {
    'strategy':       'ATR_EXPANSION',
    'atr_period':     11,
    'expansion_mult': 4.9,
    'vol_mult':       3.2,
    'max_hold_bars':  32,
    'order_type':     'Market',
    'stop_mult':      2.9,
    'tp_mult':        2.0,
}

# ── LINK: Donchian Momentum (Bayesian-optimised, Sharpe 1.64) ───────────────
LINK_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   22,
    'adx_min':           15,
    'adx_trend':         21,
    'vol_mult':          2.8,
    'hurst_min':         0.51,
    'vol_atr_max':       0.02,
    'max_hold_bars':     36,
    'order_type':        'Market',
    'stop_mult':         2.2,
    'tp_mult':           9.0,
    'atr_donchian_factor': 0.5,
}

# ── SPY: Donchian Momentum (daily, Bayesian-optimised, Sharpe 1.14) ────────
SPY_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   18,
    'adx_min':           28,
    'adx_trend':         20,
    'vol_mult':          1.2,
    'hurst_min':         0.36,
    'vol_atr_max':       0.04,
    'max_hold_bars':     72,
    'order_type':        'Market',
    'stop_mult':         4.0,
    'tp_mult':           8.5,
    'atr_donchian_factor': 2.0,
    'timeframe':         '1d',
}

# ── QQQ: ATR Expansion (daily, Bayesian-optimised, Sharpe 1.50) ───────────
QQQ_CONFIG = {
    'strategy':       'ATR_EXPANSION',
    'atr_period':     21,
    'expansion_mult': 1.5,
    'vol_mult':       1.5,
    'max_hold_bars':  26,
    'order_type':     'Market',
    'stop_mult':      2.2,
    'tp_mult':        3.0,
    'timeframe':      '1d',
}

# ── GLD: Donchian Momentum (daily, Bayesian-optimised, Sharpe 1.79) ────────
GLD_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   16,
    'adx_min':           22,
    'adx_trend':         18,
    'vol_mult':          1.1,
    'hurst_min':         0.39,
    'vol_atr_max':       0.04,
    'max_hold_bars':     48,
    'order_type':        'Market',
    'stop_mult':         4.0,
    'tp_mult':           2.5,
    'atr_donchian_factor': 1.5,
    'timeframe':         '1d',
}

# ── Forex Pairs (daily) ──────────────────────────────────────────────────────
EURUSD_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   40,
    'adx_min':           15,
    'adx_trend':         27,
    'vol_mult':          1.3,
    'hurst_min':         0.50,
    'vol_atr_max':       0.1,
    'max_hold_bars':     30,
    'order_type':        'Market',
    'stop_mult':         3.2,
    'tp_mult':           8.0,
    'atr_donchian_factor': 0.5,
    'timeframe':         '1d',
}

GBPUSD_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   21,
    'adx_min':           28,
    'adx_trend':         27,
    'vol_mult':          3.8,
    'hurst_min':         0.37,
    'vol_atr_max':       None,
    'max_hold_bars':     42,
    'order_type':        'Market',
    'stop_mult':         2.6,
    'tp_mult':           7.0,
    'atr_donchian_factor': 1.0,
    'timeframe':         '1d',
}

USDJPY_CONFIG = {
    'strategy':          'DONCHIAN_MOMENTUM',
    'donchian_period':   15,
    'adx_min':           30,
    'adx_trend':         31,
    'vol_mult':          4.0,
    'hurst_min':         0.47,
    'vol_atr_max':       None,
    'max_hold_bars':     60,
    'order_type':        'Market',
    'stop_mult':         3.7,
    'tp_mult':           10.5,
    'atr_donchian_factor': 2.0,
    'timeframe':         '1d',
}

# ── Volume-Profile MR Configs (v6.0) ─────────────────────────────────────────────
DOGE_VPMR_CONFIG = {
    'strategy': 'VOLUME_PROFILE_MR',
    'vwap_lookback': 48,
    'deviation_mult': 1.6,
    'poc_bins': 25,
    'vol_imbalance_min': 1.2,
    'max_hold_bars': 14,
    'order_type': 'Limit',
    'stop_mult': 2.3,
    'tp_mult': 2.5,
}

GLD_VPMR_CONFIG = {
    'strategy': 'VOLUME_PROFILE_MR',
    'vwap_lookback': 60,
    'deviation_mult': 1.4,
    'poc_bins': 30,
    'vol_imbalance_min': 1.1,
    'max_hold_bars': 10,
    'order_type': 'Limit',
    'stop_mult': 1.8,
    'tp_mult': 2.5,
    'timeframe': '1d',
}

# ── Master Config Map ────────────────────────────────────────────────────────────────
ASSET_CONFIG = {
    # Crypto (hourly)
    'BTCUSDT':  BTC_CONFIG,
    'ETHUSDT':  ETH_CONFIG,
    'SOLUSDT':  SOL_CONFIG,
    'AVAXUSDT': AVAX_CONFIG,
    'DOGEUSDT': DOGE_VPMR_CONFIG,   # v6.0: switched to Volume-Profile MR (better in ranging DOGE markets)
    'BNBUSDT':  BNB_CONFIG,
    'XRPUSDT':  XRP_CONFIG,
    'LINKUSDT': LINK_CONFIG,
    # Traditional (daily)
    'SPY':      SPY_CONFIG,
    'QQQ':      QQQ_CONFIG,
    'GLD':      GLD_VPMR_CONFIG,    # v6.0: Gold is mean-reverting, ideal for Volume-Profile MR
    # Forex (daily)
    'EURUSD':   EURUSD_CONFIG,
    'GBPUSD':   GBPUSD_CONFIG,
    'USDJPY':   USDJPY_CONFIG,
}

# Data file mapping (symbol → parquet filename prefix + timeframe)
DATA_FILE_MAP = {
    'BTCUSDT':  ('BTC_USD', '1h'),
    'ETHUSDT':  ('ETH_USD', '1h'),
    'SOLUSDT':  ('SOL_USD', '1h'),
    'AVAXUSDT': ('AVAX_USD', '1h'),
    'DOGEUSDT': ('DOGE_USD', '1h'),
    'BNBUSDT':  ('BNB_USD', '1h'),
    'XRPUSDT':  ('XRP_USD', '1h'),
    'LINKUSDT': ('LINK_USD', '1h'),
    'SPY':      ('SPY', '1d'),
    'QQQ':      ('QQQ', '1d'),
    'GLD':      ('GLD', '1d'),
    'EURUSD':   ('EURUSD', '1d'),
    'GBPUSD':   ('GBPUSD', '1d'),
    'USDJPY':   ('USDJPY', '1d'),
}
# ── Signal Generators ─────────────────────────────────────────────────────────

def _atr_expansion_signal(df: pd.DataFrame, cfg: dict, symbol: str) -> dict:
    """
    ATR Expansion Breakout strategy.
    Enter on a bar whose range exceeds expansion_mult × prior ATR with volume surge.
    Used for: BTC, SOL, DOGE, XRP
    """
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
        'symbol':     symbol,
        'signal':     signal,
        'regime':     regime,
        'strategy':   'ATR_EXPANSION',
        'order_type': cfg['order_type'],
        'conviction': round(float(bar_range[-1] / prev_atr[-1]) / 5.0, 3),
        'price':      price,
        'atr':        atr_last,
        'adx':        adx_last,
        'rsi':        float(rsi_v[-1]),
        'hurst':      0.5,
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


def _donchian_momentum_signal(df: pd.DataFrame, cfg: dict, symbol: str) -> dict:
    """
    Donchian Momentum with Hurst regime filter.
    Enter on Donchian channel breakout with ADX + Hurst + volume confirmation.
    Used for: ETH, AVAX, BNB, LINK
    """
    close  = df['close'].values.astype(float)
    high   = df['high'].values.astype(float)
    low    = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)
    dp = cfg['donchian_period']

    # Adaptive Donchian period if factor is set
    if cfg.get('atr_donchian_factor') is not None:
        atr_pct = _atr(high, low, close, 14)[-1] / close[-1] if close[-1] > 0 else 0.01
        dp = max(10, min(50, int(dp * (1 + cfg['atr_donchian_factor'] * (atr_pct - 0.02) / 0.02))))

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
    low_vol   = (cfg.get('vol_atr_max') is None) or (atr_pct <= cfg['vol_atr_max'])

    long_sig  = bool(not np.isnan(dc_upper) and price > dc_upper and adx_ok and vol_ok and low_vol and trending)
    short_sig = bool(not np.isnan(dc_lower) and price < dc_lower and adx_ok and vol_ok and low_vol and trending)
    signal    = 'LONG' if long_sig else ('SHORT' if short_sig else 'FLAT')

    regime = 'TRENDING' if trending else ('RANGING' if adx_last < 15 else 'TRANSITION')

    return {
        'symbol':     symbol,
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
    Multi-asset strategy router. Routes each symbol to its OOS-validated strategy.

    Usage:
        from tradingagents.research.per_asset_router import PerAssetRouter
        router = PerAssetRouter()
        sig = router.generate_signals(df, 'BTCUSDT')
    """

    SUPPORTED_SYMBOLS = set(ASSET_CONFIG.keys())

    def generate_signals(self, df: pd.DataFrame, symbol: Optional[str] = None) -> dict:
        """
        Generate trading signal for the given symbol.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            symbol: Trading symbol (e.g., BTCUSDT, ETHUSDT, SOLUSDT, etc.)

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

        cfg = ASSET_CONFIG[sym]
        strategy = cfg['strategy']

        if strategy == 'ATR_EXPANSION':
            return _atr_expansion_signal(df, cfg, sym)
        elif strategy == 'DONCHIAN_MOMENTUM':
            return _donchian_momentum_signal(df, cfg, sym)
        elif strategy == 'VOLUME_PROFILE_MR':
            return volume_profile_mr_signal(df, cfg, sym)
        else:
            raise ValueError(f'Unknown strategy: {strategy}')

    def get_order_type(self, symbol: str) -> str:
        """Returns the validated order type for a given symbol."""
        cfg = ASSET_CONFIG.get(symbol.upper(), {})
        return cfg.get('order_type', 'Market')

    def get_max_hold_bars(self, symbol: str) -> int:
        """Returns the validated max hold period for a given symbol."""
        cfg = ASSET_CONFIG.get(symbol.upper(), {})
        return cfg.get('max_hold_bars', 24)

    def get_config(self, symbol: str) -> dict:
        """Returns the full config dict for a symbol."""
        return ASSET_CONFIG.get(symbol.upper(), {})

    @classmethod
    def get_all_symbols(cls) -> list:
        """Returns all supported symbols."""
        return list(ASSET_CONFIG.keys())
