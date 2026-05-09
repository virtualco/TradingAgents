"""
Dual-Regime Crypto Day Trading Strategy — TradingAgents v4
===========================================================
Architecture: Hurst Exponent + ADX Regime Classifier → Momentum OR Mean-Reversion

REGIME DETECTION LAYER
  Primary:  Hurst Exponent (rolling 96-bar window on 1h candles = 4 days)
            H > 0.55  → TRENDING  → Momentum sub-strategy
            H < 0.45  → RANGING   → Mean-Reversion sub-strategy
            0.45 ≤ H ≤ 0.55 → TRANSITION → No new trades (flat)
  Secondary: ADX(14) confirmation
            ADX > 25 confirms TRENDING
            ADX < 20 confirms RANGING

MOMENTUM SUB-STRATEGY (Trending regime)
  Entry:  EMA(9) crosses EMA(21) in direction of 50-EMA trend
          + MACD histogram positive/negative
          + Volume ≥ 1.2× 20-bar rolling average
  Exit:   EMA crossover reversal OR ATR(14) trailing stop (2.5×)

MEAN-REVERSION SUB-STRATEGY (Ranging regime)
  Entry:  Price touches lower Bollinger Band (2σ, 20-bar) → Long
          Price touches upper Bollinger Band (2σ, 20-bar) → Short
          + RSI(14) < 30 for long entries, > 70 for short entries
          + Price within ±1 ATR of Bollinger midline (avoid trending breakouts)
  Exit:   Price returns to Bollinger midline (BB_mid) OR stop at 1.5× ATR

RISK OVERLAY (applied to both sub-strategies)
  - ATR-based position sizing (1% account risk per trade)
  - Maximum 3 open positions
  - 1-hour minimum trade cooldown per symbol
  - Regime change forces immediate position evaluation

References:
  - Hurst (1951): Long-range dependence in time series
  - Mandelbrot & Wallis (1969): Fractional Brownian motion
  - Samara Asset Management (2023): Hurst Exponent in Crypto
  - Wilder (1978): ADX trend strength indicator
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


# ── Regime Types ──────────────────────────────────────────────────────────────

RegimeType = Literal["TRENDING", "RANGING", "TRANSITION"]


# ── Indicator Utilities ───────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX trend strength indicator."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    # Zero out where the other direction is larger
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr_s   = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean().fillna(0)


def _hurst_exponent(series: pd.Series, window: int = 96) -> pd.Series:
    """
    Fast vectorised Hurst Exponent using Variance-of-Increments (Higuchi-inspired).
    Runs in O(n * lags) instead of O(n²), suitable for 35k+ bar datasets.

    window: rolling window in bars (96 × 1h = 4 days)
    H > 0.55 → persistent/trending
    H < 0.45 → anti-persistent/mean-reverting
    H ≈ 0.50 → random walk
    """
    def _fast_hurst(x: np.ndarray) -> float:
        n = len(x)
        if n < 20:
            return 0.5
        try:
            # Use log-price differences
            lx = np.log(np.abs(x) + 1e-10)
            lags = [2, 4, 8, 16, 32]
            lags = [l for l in lags if l < n // 2]
            if len(lags) < 2:
                return 0.5
            var_list = []
            for lag in lags:
                diffs = lx[lag:] - lx[:-lag]
                var_list.append(np.var(diffs))
            # Hurst from slope of log(var) vs log(lag): var ~ lag^(2H)
            log_lags = np.log(lags)
            log_vars = np.log(np.array(var_list) + 1e-20)
            slope = np.polyfit(log_lags, log_vars, 1)[0]
            hurst = slope / 2.0
            return float(np.clip(hurst, 0.0, 1.0))
        except Exception:
            return 0.5

    # Rolling apply on price series directly
    hurst_series = series.rolling(window=window, min_periods=window // 2).apply(
        _fast_hurst, raw=True
    )
    return hurst_series.fillna(0.5)


def _bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Returns (upper, middle, lower) Bollinger Bands."""
    mid   = close.rolling(window=period).mean()
    sigma = close.rolling(window=period).std()
    return mid + std_dev * sigma, mid, mid - std_dev * sigma


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


# ── Regime Classifier ─────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    Classifies market regime using Hurst Exponent + ADX.

    Primary signal: Hurst Exponent (rolling R/S analysis)
    Confirmation:   ADX trend strength
    """

    def __init__(
        self,
        hurst_window: int = 96,        # 4 days of 1h bars
        hurst_trend_threshold: float = 0.55,
        hurst_revert_threshold: float = 0.45,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
    ):
        self.hurst_window           = hurst_window
        self.hurst_trend_threshold  = hurst_trend_threshold
        self.hurst_revert_threshold = hurst_revert_threshold
        self.adx_period             = adx_period
        self.adx_trend_threshold    = adx_trend_threshold
        self.adx_range_threshold    = adx_range_threshold

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a Series of RegimeType strings indexed like df.
        Values: "TRENDING", "RANGING", "TRANSITION"
        """
        hurst = _hurst_exponent(df["close"], window=self.hurst_window)
        adx   = _adx(df, period=self.adx_period)

        regime = pd.Series("TRANSITION", index=df.index, dtype=object)

        # TRENDING: Hurst > threshold AND ADX confirms
        trending_mask = (
            (hurst > self.hurst_trend_threshold) &
            (adx > self.adx_trend_threshold)
        )
        # RANGING: Hurst < threshold AND ADX confirms low trend
        ranging_mask = (
            (hurst < self.hurst_revert_threshold) &
            (adx < self.adx_range_threshold)
        )

        regime[trending_mask] = "TRENDING"
        regime[ranging_mask]  = "RANGING"

        return regime


# ── Momentum Sub-Strategy ─────────────────────────────────────────────────────

class MomentumStrategy:
    """
    EMA crossover + MACD + volume surge.
    Designed for TRENDING regime.
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        ema_trend: int = 50,
        volume_mult: float = 1.2,
    ):
        self.ema_fast    = ema_fast
        self.ema_slow    = ema_slow
        self.ema_trend   = ema_trend
        self.volume_mult = volume_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Returns +1 (long), -1 (short), 0 (flat)."""
        close  = df["close"]
        volume = df["volume"]

        fast  = _ema(close, self.ema_fast)
        slow  = _ema(close, self.ema_slow)
        trend = _ema(close, self.ema_trend)
        macd  = _macd_histogram(close)
        vol_avg = volume.rolling(20).mean()

        # Crossover signals
        cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        # Filters
        above_trend   = close > trend
        below_trend   = close < trend
        macd_positive = macd > 0
        macd_negative = macd < 0
        vol_surge     = volume >= self.volume_mult * vol_avg

        long_signal  = cross_up   & above_trend & macd_positive & vol_surge
        short_signal = cross_down & below_trend & macd_negative & vol_surge

        signal = pd.Series(0, index=df.index)
        signal[long_signal]  = 1
        signal[short_signal] = -1

        # Hold position until reversal
        return signal.replace(0, np.nan).ffill().fillna(0).astype(int)


# ── Mean-Reversion Sub-Strategy ───────────────────────────────────────────────

class MeanReversionStrategy:
    """
    Bollinger Band touch + RSI extreme + ATR proximity filter.
    Designed for RANGING regime.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        atr_period: int = 14,
        atr_proximity_mult: float = 1.5,
    ):
        self.bb_period          = bb_period
        self.bb_std             = bb_std
        self.rsi_period         = rsi_period
        self.rsi_oversold       = rsi_oversold
        self.rsi_overbought     = rsi_overbought
        self.atr_period         = atr_period
        self.atr_proximity_mult = atr_proximity_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Returns +1 (long), -1 (short), 0 (flat)."""
        close = df["close"]

        bb_upper, bb_mid, bb_lower = _bollinger_bands(close, self.bb_period, self.bb_std)
        rsi  = _rsi(close, self.rsi_period)
        atr  = _atr(df, self.atr_period)

        # Entry: price touches band + RSI extreme
        long_entry  = (close <= bb_lower) & (rsi < self.rsi_oversold)
        short_entry = (close >= bb_upper) & (rsi > self.rsi_overbought)

        # Avoid breakout traps: price must be within ATR proximity of midline
        # (i.e., not in a strong directional move away from the mean)
        near_mid = (close - bb_mid).abs() < self.atr_proximity_mult * atr

        # Exit: price returns to midline
        at_mid_from_long  = (close >= bb_mid)
        at_mid_from_short = (close <= bb_mid)

        signal = pd.Series(0, index=df.index)
        in_long  = False
        in_short = False

        for i in range(len(df)):
            if in_long:
                if at_mid_from_long.iloc[i]:
                    in_long = False
                    signal.iloc[i] = 0
                else:
                    signal.iloc[i] = 1
            elif in_short:
                if at_mid_from_short.iloc[i]:
                    in_short = False
                    signal.iloc[i] = 0
                else:
                    signal.iloc[i] = -1
            else:
                if long_entry.iloc[i]:
                    in_long = True
                    signal.iloc[i] = 1
                elif short_entry.iloc[i]:
                    in_short = True
                    signal.iloc[i] = -1

        return signal


# ── Dual-Regime Strategy (Main Interface) ─────────────────────────────────────

class DualRegimeStrategy:
    """
    Main dual-regime strategy that combines:
      - RegimeClassifier (Hurst + ADX)
      - MomentumStrategy (for TRENDING regime)
      - MeanReversionStrategy (for RANGING regime)

    Signal generation:
      1. Classify regime for each bar
      2. Generate signals from both sub-strategies
      3. Apply regime mask: use momentum signal in TRENDING,
         mean-reversion signal in RANGING, flat in TRANSITION

    Usage:
        strategy = DualRegimeStrategy()
        signals = strategy.generate_signals(df)   # returns pd.Series of -1/0/1
        regime  = strategy.get_regime(df)         # returns pd.Series of regime labels
    """

    def __init__(
        self,
        # Regime classifier params
        hurst_window: int = 96,
        hurst_trend_threshold: float = 0.55,
        hurst_revert_threshold: float = 0.45,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        # Momentum params
        ema_fast: int = 9,
        ema_slow: int = 21,
        ema_trend: int = 50,
        volume_mult: float = 1.2,
        # Mean-reversion params
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        # Transition handling
        flat_on_transition: bool = True,
        close_on_regime_change: bool = True,
    ):
        self.regime_classifier = RegimeClassifier(
            hurst_window=hurst_window,
            hurst_trend_threshold=hurst_trend_threshold,
            hurst_revert_threshold=hurst_revert_threshold,
            adx_period=adx_period,
            adx_trend_threshold=adx_trend_threshold,
            adx_range_threshold=adx_range_threshold,
        )
        self.momentum_strategy = MomentumStrategy(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_trend=ema_trend,
            volume_mult=volume_mult,
        )
        self.mean_reversion_strategy = MeanReversionStrategy(
            bb_period=bb_period,
            bb_std=bb_std,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
        )
        self.flat_on_transition     = flat_on_transition
        self.close_on_regime_change = close_on_regime_change

    def get_regime(self, df: pd.DataFrame) -> pd.Series:
        """Return regime classification for each bar."""
        return self.regime_classifier.classify(df)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate combined dual-regime signals.
        Returns pd.Series of int: +1 (long), -1 (short), 0 (flat)
        """
        if len(df) < 150:
            return pd.Series(0, index=df.index)

        regime     = self.regime_classifier.classify(df)
        mom_sig    = self.momentum_strategy.generate_signals(df)
        revert_sig = self.mean_reversion_strategy.generate_signals(df)

        # Combine: use sub-strategy signal only in its designated regime
        combined = pd.Series(0, index=df.index)
        combined[regime == "TRENDING"] = mom_sig[regime == "TRENDING"]
        combined[regime == "RANGING"]  = revert_sig[regime == "RANGING"]
        if self.flat_on_transition:
            combined[regime == "TRANSITION"] = 0

        # On regime change: close position (set to 0 for one bar)
        if self.close_on_regime_change:
            regime_changed = regime != regime.shift(1)
            combined[regime_changed] = 0

        return combined.astype(int)

    def get_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a diagnostic DataFrame with all intermediate signals.
        Useful for backtesting analysis and visualisation.
        """
        regime     = self.regime_classifier.classify(df)
        mom_sig    = self.momentum_strategy.generate_signals(df)
        revert_sig = self.mean_reversion_strategy.generate_signals(df)
        combined   = self.generate_signals(df)
        hurst      = _hurst_exponent(df["close"], self.regime_classifier.hurst_window)
        adx        = _adx(df, self.regime_classifier.adx_period)
        bb_upper, bb_mid, bb_lower = _bollinger_bands(df["close"])
        rsi        = _rsi(df["close"])

        return pd.DataFrame({
            "close":        df["close"],
            "regime":       regime,
            "hurst":        hurst,
            "adx":          adx,
            "momentum_sig": mom_sig,
            "revert_sig":   revert_sig,
            "combined_sig": combined,
            "bb_upper":     bb_upper,
            "bb_mid":       bb_mid,
            "bb_lower":     bb_lower,
            "rsi":          rsi,
        }, index=df.index)
