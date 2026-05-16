"""
Volume-Profile Mean Reversion Signal Generator
================================================
Third strategy type for TradingAgents alongside ATR_EXPANSION and DONCHIAN_MOMENTUM.

Concept:
  - Computes VWAP (Volume-Weighted Average Price) as fair value anchor
  - Identifies Point of Control (POC) — price level with highest volume concentration
  - Generates LONG when price deviates below VWAP - σ at high-volume node (mean reversion entry)
  - Generates SHORT when price deviates above VWAP + σ at high-volume node
  - Uses volume imbalance ratio and distance-from-POC as conviction modifiers

Parameters (Bayesian-optimisable):
  - vwap_lookback: bars for VWAP calculation (default 48)
  - deviation_mult: σ multiplier for entry threshold (default 1.5)
  - poc_bins: number of price bins for volume profile (default 30)
  - vol_imbalance_min: minimum buy/sell volume ratio for confirmation (default 1.3)
  - max_hold_bars: maximum bars to hold position
  - stop_mult: ATR multiplier for stop loss
  - tp_mult: ATR multiplier for take profit

Best suited for: ranging/mean-reverting markets (Hurst < 0.45, low ADX)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def _vwap(close: np.ndarray, volume: np.ndarray, lookback: int) -> np.ndarray:
    """Compute rolling VWAP over lookback window."""
    n = len(close)
    vwap = np.full(n, close[-1])
    for i in range(lookback, n):
        window_close = close[i - lookback:i + 1]
        window_vol = volume[i - lookback:i + 1]
        total_vol = window_vol.sum()
        if total_vol > 0:
            vwap[i] = np.sum(window_close * window_vol) / total_vol
        else:
            vwap[i] = window_close.mean()
    return vwap


def _rolling_std(close: np.ndarray, vwap: np.ndarray, lookback: int) -> np.ndarray:
    """Compute rolling standard deviation of price from VWAP."""
    n = len(close)
    std = np.full(n, 0.01)
    for i in range(lookback, n):
        deviations = close[i - lookback:i + 1] - vwap[i - lookback:i + 1]
        std[i] = max(np.std(deviations), 1e-8)
    return std


def _point_of_control(close: np.ndarray, volume: np.ndarray, lookback: int, bins: int) -> float:
    """
    Find the Point of Control (POC) — price level with highest volume concentration.
    Uses the most recent `lookback` bars.
    """
    window_close = close[-lookback:]
    window_vol = volume[-lookback:]

    if len(window_close) < 10:
        return float(close[-1])

    price_min = window_close.min()
    price_max = window_close.max()

    if price_max - price_min < 1e-8:
        return float(close[-1])

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)

    for i, price in enumerate(window_close):
        bin_idx = int((price - price_min) / (price_max - price_min) * (bins - 1))
        bin_idx = min(bin_idx, bins - 1)
        bin_volumes[bin_idx] += window_vol[i]

    poc_bin = np.argmax(bin_volumes)
    poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0
    return float(poc_price)


def _volume_imbalance(volume: np.ndarray, close: np.ndarray, opens: np.ndarray, lookback: int = 20) -> float:
    """
    Compute buy/sell volume imbalance ratio over recent bars.
    Buy volume = volume on bars where close > open.
    """
    recent_close = close[-lookback:]
    recent_open = opens[-lookback:]
    recent_vol = volume[-lookback:]

    buy_mask = recent_close > recent_open
    buy_vol = recent_vol[buy_mask].sum()
    sell_vol = recent_vol[~buy_mask].sum()

    if sell_vol < 1e-9:
        return 2.0
    return float(buy_vol / sell_vol)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    atr[:period] = tr[:period].mean()
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Compute latest ADX value."""
    n = len(close)
    if n < period * 2:
        return 20.0

    dm_p = np.zeros(n)
    dm_m = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        dm_p[i] = up if (up > dn and up > 0) else 0
        dm_m[i] = dn if (dn > up and dn > 0) else 0

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    alpha = 2.0 / (period + 1)
    atr_s = np.zeros(n)
    di_p_s = np.zeros(n)
    di_m_s = np.zeros(n)
    atr_s[period] = tr[1:period + 1].mean()
    di_p_s[period] = dm_p[1:period + 1].mean()
    di_m_s[period] = dm_m[1:period + 1].mean()

    for i in range(period + 1, n):
        atr_s[i] = alpha * tr[i] + (1 - alpha) * atr_s[i - 1]
        di_p_s[i] = alpha * dm_p[i] + (1 - alpha) * di_p_s[i - 1]
        di_m_s[i] = alpha * dm_m[i] + (1 - alpha) * di_m_s[i - 1]

    denom = atr_s[-1] if atr_s[-1] > 0 else 1e-9
    di_plus = 100.0 * di_p_s[-1] / denom
    di_minus = 100.0 * di_m_s[-1] / denom
    dx = 100.0 * abs(di_plus - di_minus) / max(di_plus + di_minus, 1e-9)
    return float(dx)


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute latest RSI value."""
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 2.0 / (period + 1)
    avg_g = gain[:period].mean()
    avg_l = loss[:period].mean()
    for i in range(period, len(gain)):
        avg_g = alpha * gain[i] + (1 - alpha) * avg_g
        avg_l = alpha * loss[i] + (1 - alpha) * avg_l
    if avg_l < 1e-9:
        return 100.0
    rs = avg_g / avg_l
    return float(100.0 - (100.0 / (1.0 + rs)))


def _hurst_fast(close: np.ndarray, window: int = 96) -> float:
    """Compute latest Hurst exponent."""
    n = len(close)
    if n < window:
        return 0.5
    lags = [l for l in [2, 4, 8, 16, 32] if l < window // 2]
    if len(lags) < 2:
        return 0.5
    x = np.log(np.abs(close[-window:]) + 1e-10)
    log_lags = np.log(lags)
    vl = [np.var(x[l:] - x[:-l]) for l in lags]
    try:
        slope = np.polyfit(log_lags, np.log(np.array(vl) + 1e-20), 1)[0]
        return float(np.clip(slope / 2.0, 0.0, 1.0))
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME_PROFILE_MR Signal Generator
# ══════════════════════════════════════════════════════════════════════════════

def volume_profile_mr_signal(df: pd.DataFrame, cfg: dict, symbol: str) -> dict:
    """
    Volume-Profile Mean Reversion strategy.

    Enter LONG when price is below VWAP - deviation_mult*σ near a high-volume node (POC).
    Enter SHORT when price is above VWAP + deviation_mult*σ near a high-volume node.

    Best in RANGING regimes (Hurst < 0.45, ADX < 20).
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    opens = df['open'].values.astype(float)
    n = len(close)

    # Parameters
    vwap_lookback = cfg.get('vwap_lookback', 48)
    deviation_mult = cfg.get('deviation_mult', 1.5)
    poc_bins = cfg.get('poc_bins', 30)
    vol_imbalance_min = cfg.get('vol_imbalance_min', 1.3)
    stop_mult = cfg.get('stop_mult', 2.0)
    tp_mult = cfg.get('tp_mult', 3.0)
    max_hold_bars = cfg.get('max_hold_bars', 24)

    # Compute indicators
    vwap = _vwap(close, volume, vwap_lookback)
    std = _rolling_std(close, vwap, vwap_lookback)
    poc = _point_of_control(close, volume, vwap_lookback, poc_bins)
    vol_imbalance = _volume_imbalance(volume, close, opens)
    atr_v = _atr(high, low, close, 14)
    adx_val = _adx(high, low, close)
    rsi_val = _rsi(close)
    hurst_val = _hurst_fast(close)

    # Current values
    price = float(close[-1])
    vwap_last = float(vwap[-1])
    std_last = float(std[-1])
    atr_last = float(atr_v[-1])

    # Deviation from VWAP in σ units
    deviation = (price - vwap_last) / std_last if std_last > 0 else 0.0

    # POC proximity (how close price is to POC, normalised)
    poc_distance = abs(price - poc) / atr_last if atr_last > 0 else 999.0
    near_poc = poc_distance < 2.0  # within 2 ATR of POC

    # Signal logic — mean reversion
    # LONG: price below VWAP - threshold, near POC, buy volume imbalance
    long_condition = (
        deviation < -deviation_mult and
        near_poc and
        vol_imbalance > vol_imbalance_min and
        hurst_val < 0.50  # mean-reverting regime
    )

    # SHORT: price above VWAP + threshold, near POC, sell volume imbalance
    sell_imbalance = 1.0 / vol_imbalance if vol_imbalance > 0 else 0
    short_condition = (
        deviation > deviation_mult and
        near_poc and
        sell_imbalance > vol_imbalance_min and
        hurst_val < 0.50
    )

    signal = 'LONG' if long_condition else ('SHORT' if short_condition else 'FLAT')

    # Regime classification (using statistical method for now, ML override comes in v6.0)
    regime = 'RANGING' if adx_val < 20 else ('TRENDING' if adx_val > 30 else 'TRANSITION')

    # Conviction: based on deviation magnitude and volume confirmation
    conviction = min(abs(deviation) / (deviation_mult * 2.0), 1.0) * (0.7 + 0.3 * min(vol_imbalance / 2.0, 1.0))
    conviction = round(float(conviction), 3)

    return {
        'symbol': symbol,
        'signal': signal,
        'regime': regime,
        'strategy': 'VOLUME_PROFILE_MR',
        'order_type': cfg.get('order_type', 'Limit'),
        'conviction': conviction,
        'price': price,
        'atr': atr_last,
        'adx': adx_val,
        'rsi': rsi_val,
        'hurst': hurst_val,
        'vwap': vwap_last,
        'vwap_deviation': round(deviation, 3),
        'poc': poc,
        'poc_distance': round(poc_distance, 2),
        'vol_imbalance': round(vol_imbalance, 3),
        'stop_loss': round(price - stop_mult * atr_last, 4) if signal == 'LONG'
                     else round(price + stop_mult * atr_last, 4) if signal == 'SHORT' else 0,
        'take_profit': round(price + tp_mult * atr_last, 4) if signal == 'LONG'
                       else round(price - tp_mult * atr_last, 4) if signal == 'SHORT' else 0,
        'max_hold_bars': max_hold_bars,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Default Configs for Volume-Profile Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

# These are initial configs — will be Bayesian-optimised per asset
VPMR_BTC_CONFIG = {
    'strategy': 'VOLUME_PROFILE_MR',
    'vwap_lookback': 48,
    'deviation_mult': 1.8,
    'poc_bins': 30,
    'vol_imbalance_min': 1.2,
    'max_hold_bars': 12,
    'order_type': 'Limit',
    'stop_mult': 2.5,
    'tp_mult': 3.0,
}

VPMR_ETH_CONFIG = {
    'strategy': 'VOLUME_PROFILE_MR',
    'vwap_lookback': 36,
    'deviation_mult': 1.5,
    'poc_bins': 25,
    'vol_imbalance_min': 1.3,
    'max_hold_bars': 16,
    'order_type': 'Limit',
    'stop_mult': 2.0,
    'tp_mult': 3.5,
}

VPMR_DEFAULT_CONFIG = {
    'strategy': 'VOLUME_PROFILE_MR',
    'vwap_lookback': 48,
    'deviation_mult': 1.5,
    'poc_bins': 30,
    'vol_imbalance_min': 1.3,
    'max_hold_bars': 18,
    'order_type': 'Limit',
    'stop_mult': 2.0,
    'tp_mult': 3.0,
}
