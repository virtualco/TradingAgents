"""
CryptoDayTradingStrategy — v1 Baseline
Multi-indicator confluence: RSI + EMA crossover + MACD + Volume filter
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

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for each bar.
        Returns pd.Series: +1 (long), -1 (short), 0 (flat).
        """
        if len(df) < max(self.macd_slow, self.trend_ema) + 10:
            return pd.Series(0, index=df.index)

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        rsi = self._rsi(close)
        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        trend = self._ema(close, self.trend_ema)
        _, _, macd_hist = self._macd(close)

        vol_median = volume.rolling(self.vol_period, min_periods=self.vol_period // 2).median()
        vol_surge = volume > (self.vol_mult * vol_median)

        # LONG conditions
        bullish_trend = close > trend
        ema_cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
        rsi_recovering = (rsi > self.rsi_oversold) & (rsi.shift(1) <= self.rsi_oversold)
        macd_bullish = macd_hist > 0
        long_entry = bullish_trend & ema_cross_up & rsi_recovering & macd_bullish & vol_surge
        long_hold = bullish_trend & (ema_f > ema_s) & (rsi < self.rsi_overbought) & (rsi > 40)

        # SHORT conditions
        bearish_trend = close < trend
        ema_cross_down = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
        rsi_reversing = (rsi < self.rsi_overbought) & (rsi.shift(1) >= self.rsi_overbought)
        macd_bearish = macd_hist < 0
        short_entry = bearish_trend & ema_cross_down & rsi_reversing & macd_bearish & vol_surge
        short_hold = bearish_trend & (ema_f < ema_s) & (rsi > self.rsi_oversold) & (rsi < 60)

        raw = pd.Series(0, index=df.index, dtype=int)
        raw[long_entry | long_hold] = 1
        raw[short_entry | short_hold] = -1

        # Shift by 1 bar to avoid look-ahead bias
        signals = raw.shift(1).fillna(0).astype(int)
        return signals
