"""
ML-Enhanced Regime Detection — LightGBM Classifier
====================================================
Replaces the simple ADX-threshold regime classifier with a gradient-boosted
model trained on engineered features from historical price data.

Architecture:
  - Feature engineering: 8 technical features computed from OHLCV
  - LightGBM multi-class classifier (TRENDING=0, RANGING=1, TRANSITION=2)
  - Calibrated probability outputs for regime confidence
  - Graceful fallback to statistical classifier if model unavailable

Model lifecycle:
  - Training: scripts/train_regime_model.py (offline, uses 4-year historical data)
  - Serialization: joblib dump to data/models/regime_gbm_v1.joblib
  - Inference: this module loads model once, predicts per-bar

Features (8 dimensions):
  1. Hurst exponent (96-bar window)
  2. ADX (14-period)
  3. RSI (14-period)
  4. Volatility ratio (short/long ATR)
  5. Volume trend (20-bar slope)
  6. Price momentum (ROC 20-bar)
  7. ATR percentile (rank in 200-bar window)
  8. Directional movement balance (DI+ - DI-)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / 'data' / 'models' / 'regime_gbm_v1.joblib'

# Regime labels
REGIME_LABELS = ['TRENDING', 'RANGING', 'TRANSITION']
REGIME_MAP = {0: 'TRENDING', 1: 'RANGING', 2: 'TRANSITION'}

# ── Feature Engineering ───────────────────────────────────────────────────────

def _hurst(close: np.ndarray, window: int = 96) -> float:
    """Compute Hurst exponent for the most recent window."""
    if len(close) < window:
        return 0.5
    x = np.log(np.abs(close[-window:]) + 1e-10)
    lags = [l for l in [2, 4, 8, 16, 32] if l < window // 2]
    if len(lags) < 2:
        return 0.5
    log_lags = np.log(lags)
    vl = [np.var(x[l:] - x[:-l]) for l in lags]
    try:
        slope = np.polyfit(log_lags, np.log(np.array(vl) + 1e-20), 1)[0]
        return float(np.clip(slope / 2.0, 0.0, 1.0))
    except Exception:
        return 0.5


def _adx_full(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
    """Compute ADX, DI+, DI-."""
    n = len(close)
    if n < period * 2:
        return 20.0, 15.0, 15.0

    dm_p = np.zeros(n)
    dm_m = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        dm_p[i] = up if (up > dn and up > 0) else 0
        dm_m[i] = dn if (dn > up and dn > 0) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    alpha = 2.0 / (period + 1)
    atr_s = tr[1:period + 1].mean()
    di_p_s = dm_p[1:period + 1].mean()
    di_m_s = dm_m[1:period + 1].mean()

    for i in range(period + 1, n):
        atr_s = alpha * tr[i] + (1 - alpha) * atr_s
        di_p_s = alpha * dm_p[i] + (1 - alpha) * di_p_s
        di_m_s = alpha * dm_m[i] + (1 - alpha) * di_m_s

    denom = atr_s if atr_s > 0 else 1e-9
    di_plus = 100.0 * di_p_s / denom
    di_minus = 100.0 * di_m_s / denom
    dx = 100.0 * abs(di_plus - di_minus) / max(di_plus + di_minus, 1e-9)
    return float(dx), float(di_plus), float(di_minus)


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI."""
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


def _atr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ATR array."""
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


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the 8-dimensional feature vector for regime classification.

    Returns: np.ndarray of shape (8,)
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)

    # 1. Hurst exponent
    hurst = _hurst(close)

    # 2-3. ADX and DI balance
    adx, di_plus, di_minus = _adx_full(high, low, close)
    di_balance = di_plus - di_minus  # Feature 8

    # 4. RSI
    rsi = _rsi(close)

    # 5. Volatility ratio (short ATR / long ATR)
    atr_short = _atr_array(high, low, close, 7)
    atr_long = _atr_array(high, low, close, 28)
    vol_ratio = float(atr_short[-1] / atr_long[-1]) if atr_long[-1] > 0 else 1.0

    # 6. Volume trend (20-bar linear regression slope, normalised)
    if n >= 20:
        vol_window = volume[-20:]
        x = np.arange(20)
        try:
            slope = np.polyfit(x, vol_window / (vol_window.mean() + 1e-9), 1)[0]
            vol_trend = float(slope)
        except Exception:
            vol_trend = 0.0
    else:
        vol_trend = 0.0

    # 7. Price momentum (ROC 20-bar)
    if n >= 21:
        momentum = float((close[-1] - close[-21]) / (close[-21] + 1e-9) * 100)
    else:
        momentum = 0.0

    # 8. ATR percentile (rank in 200-bar window)
    atr_full = _atr_array(high, low, close, 14)
    window = min(200, n)
    atr_window = atr_full[-window:]
    atr_pct = float(np.searchsorted(np.sort(atr_window), atr_full[-1]) / len(atr_window))

    return np.array([hurst, adx, rsi, vol_ratio, vol_trend, momentum, atr_pct, di_balance])


# ── ML Model Loader ──────────────────────────────────────────────────────────

_model = None
_model_loaded = False


def _load_model():
    """Lazy-load the trained LightGBM model."""
    global _model, _model_loaded
    if _model_loaded:
        return _model

    _model_loaded = True
    if not MODEL_PATH.exists():
        log.warning(f"ML regime model not found at {MODEL_PATH} — using statistical fallback")
        return None

    try:
        import joblib
        _model = joblib.load(MODEL_PATH)
        log.info(f"ML regime model loaded from {MODEL_PATH}")
        return _model
    except Exception as e:
        log.error(f"Failed to load ML regime model: {e}")
        return None


# ── Statistical Fallback ──────────────────────────────────────────────────────

def _statistical_regime(adx: float, hurst: float) -> Tuple[str, float]:
    """
    Original statistical regime classifier (fallback).
    Returns (regime_label, confidence).
    """
    if adx > 30 and hurst > 0.55:
        return 'TRENDING', min(0.7 + (adx - 30) / 50, 0.95)
    elif adx < 18 and hurst < 0.45:
        return 'RANGING', min(0.7 + (18 - adx) / 30, 0.95)
    else:
        return 'TRANSITION', 0.5


# ── Public API ────────────────────────────────────────────────────────────────

def classify_regime(df: pd.DataFrame) -> dict:
    """
    Classify the current market regime using ML model (with statistical fallback).

    Returns:
        {
            'regime': str,           # TRENDING | RANGING | TRANSITION
            'confidence': float,     # 0.0 - 1.0 calibrated probability
            'probabilities': dict,   # {TRENDING: p, RANGING: p, TRANSITION: p}
            'method': str,           # 'ml_gbm' or 'statistical'
            'features': dict,        # feature values for transparency
            'model_version': str,    # model identifier
        }
    """
    features = compute_features(df)
    feature_names = ['hurst', 'adx', 'rsi', 'vol_ratio', 'vol_trend', 'momentum', 'atr_pct', 'di_balance']
    feature_dict = {name: round(float(val), 4) for name, val in zip(feature_names, features)}

    model = _load_model()

    if model is not None:
        try:
            X = features.reshape(1, -1)
            proba = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            regime = REGIME_MAP[pred_idx]
            confidence = float(proba[pred_idx])

            return {
                'regime': regime,
                'confidence': round(confidence, 3),
                'probabilities': {
                    'TRENDING': round(float(proba[0]), 3),
                    'RANGING': round(float(proba[1]), 3),
                    'TRANSITION': round(float(proba[2]), 3),
                },
                'method': 'ml_gbm',
                'features': feature_dict,
                'model_version': 'regime_gbm_v1',
            }
        except Exception as e:
            log.warning(f"ML prediction failed, using fallback: {e}")

    # Statistical fallback
    regime, confidence = _statistical_regime(features[1], features[0])  # adx, hurst
    return {
        'regime': regime,
        'confidence': round(confidence, 3),
        'probabilities': {
            'TRENDING': round(confidence if regime == 'TRENDING' else (1 - confidence) / 2, 3),
            'RANGING': round(confidence if regime == 'RANGING' else (1 - confidence) / 2, 3),
            'TRANSITION': round(confidence if regime == 'TRANSITION' else (1 - confidence) / 2, 3),
        },
        'method': 'statistical',
        'features': feature_dict,
        'model_version': 'statistical_v1',
    }


def get_model_info() -> dict:
    """Return information about the current regime model."""
    model = _load_model()
    return {
        'model_available': model is not None,
        'model_path': str(MODEL_PATH),
        'model_version': 'regime_gbm_v1' if model else 'statistical_v1',
        'feature_count': 8,
        'feature_names': ['hurst', 'adx', 'rsi', 'vol_ratio', 'vol_trend', 'momentum', 'atr_pct', 'di_balance'],
        'classes': REGIME_LABELS,
    }
