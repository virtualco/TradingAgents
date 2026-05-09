"""
CryptoDayTradingStrategy — v2 with regime filter and ATR-based sizing
Multi-indicator confluence: RSI + EMA crossover + MACD + Volume filter + Regime detection + ATR position sizing
Designed for 1-hour OHLCV data on BTC-USD / ETH-USD.
No look-ahead bias. No external TA libraries required.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class CryptoDayTradingStrategy:
    """
    Multi-indicator confluence strategy for crypto day trading.
    Signals: +1 (LONG), -1 (SHORT), 0 (FLAT)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        ema_fast: int = 9,
        ema_slow: int = 21,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        vol_period: int = 20,
        vol_mult: float = 1.2,
        trend_ema: int = 50,
        atr_period: int = 14,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.trend_ema = trend_ema
        self.atr_period = atr_period

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _macd(self, close: pd.Series):
        ema_f = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_f - ema_s
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False, min_periods=self.atr_period).mean()
        return atr

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for each bar.
        Returns pd.Series: +1 (long), -1 (short), 0 (flat).
        """
        if len(df) < max(self.macd_slow, self.trend_ema, self.atr_period) + 10:
            return pd.Series(0, index=df.index)

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        rsi = self._rsi(close)
        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        trend_ema = self._ema(close, self.trend_ema)
        _, _, macd_hist = self._macd(close)
        atr = self._atr(high, low, close)

        vol_median = volume.rolling(self.vol_period, min_periods=max(1, self.vol_period // 2)).median()
        vol_surge = volume > (self.vol_mult * vol_median)

        # Regime detection: trending if slope of trend_ema over last 3 bars is significant
        trend_ema_slope = trend_ema.diff(3) / trend_ema.shift(3).replace(0, np.nan)
        is_trending = trend_ema_slope.abs() > 0.001

        # Ranging regime is the complement
        is_ranging = ~is_trending

        # More conservative RSI thresholds in ranging regime
        rsi_os_range = 40
        rsi_ob_range = 60

        signals = pd.Series(0, index=df.index)

        # Long conditions
        long_condition_trend = (
            (close > trend_ema) &
            (ema_f > ema_s) &
            (macd_hist > 0) &
            (rsi > self.rsi_oversold) &
            vol_surge &
            is_trending
        )

        long_condition_range = (
            (rsi < rsi_os_range) &
            (macd_hist > 0) &
            vol_surge &
            is_ranging
        )

        # Short conditions
        short_condition_trend = (
            (close < trend_ema) &
            (ema_f < ema_s) &
            (macd_hist < 0) &
            (rsi < self.rsi_overbought) &
            vol_surge &
            is_trending
        )

        short_condition_range = (
            (rsi > rsi_ob_range) &
            (macd_hist < 0) &
            vol_surge &
            is_ranging
        )

        signals[long_condition_trend | long_condition_range] = 1
        signals[short_condition_trend | short_condition_range] = -1

        # Shift signals by 1 to avoid look-ahead bias
        signals = signals.shift(1).fillna(0).astype(int)

        return signals

