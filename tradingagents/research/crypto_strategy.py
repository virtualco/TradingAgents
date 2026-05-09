"""
CryptoDayTradingStrategy — v2 with regime filter, ATR-based sizing and improved entry+exit
Multi-indicator confluence: RSI + EMA crossover + MACD + Volume filter + Regime detection + ATR position sizing + trailing ATR stop loss
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
        rsi_oversold: float = 30,  # tightened for stronger signals
        rsi_overbought: float = 70,  # tightened
        ema_fast: int = 9,
        ema_slow: int = 21,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        vol_period: int = 20,
        vol_mult: float = 1.2,
        trend_ema: int = 50,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,  # trailing stop multiplier
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
        self.atr_stop_mult = atr_stop_mult

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

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Indicators
        rsi = self._rsi(close)
        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        trend_ema = self._ema(close, self.trend_ema)
        macd_line, signal_line, macd_hist = self._macd(close)
        atr = self._atr(high, low, close)

        # Volume filter: volume > vol_mult * rolling avg volume
        vol_avg = volume.rolling(self.vol_period, min_periods=self.vol_period).mean()
        vol_filter = volume > (vol_avg * self.vol_mult)

        # Regime filter: Trend if close above trend_ema else range
        is_uptrend = close > trend_ema
        is_downtrend = close < trend_ema

        # Entry conditions
        # LONG: 
        # - EMA fast above EMA slow (bullish momentum)
        # - MACD histogram > 0 and aligned with uptrend
        # - RSI oversold < RSI < 60 (rebound zone)
        # - volume filter passed
        # - in uptrend regime

        long_entry = (
            (ema_fast > ema_slow) &
            (macd_hist > 0) &
            (is_uptrend) &
            (rsi > self.rsi_oversold) & (rsi < 60) &
            (vol_filter)
        )

        # SHORT entry:
        # - EMA fast below EMA slow
        # - MACD histogram < 0 and aligned with downtrend
        # - RSI < rsi_overbought but >40 (rejection zone)
        # - volume filter passed
        # - in downtrend regime

        short_entry = (
            (ema_fast < ema_slow) &
            (macd_hist < 0) &
            (is_downtrend) &
            (rsi < self.rsi_overbought) & (rsi > 40) &
            (vol_filter)
        )

        # Initialize signal series
        signals = pd.Series(0, index=df.index)

        position = 0  # 1=long, -1=short, 0=flat
        entry_price = 0.0
        trail_stop = np.nan

        for i in range(len(df)):
            if i == 0:
                signals.iloc[i] = 0
                continue

            close_i = close.iloc[i]
            atr_i = atr.iloc[i]

            if position == 0:
                # Check entries
                if long_entry.iloc[i]:
                    position = 1
                    entry_price = close_i
                    trail_stop = close_i - self.atr_stop_mult * atr_i
                    signals.iloc[i] = 1
                elif short_entry.iloc[i]:
                    position = -1
                    entry_price = close_i
                    trail_stop = close_i + self.atr_stop_mult * atr_i
                    signals.iloc[i] = -1
                else:
                    signals.iloc[i] = 0

            elif position == 1:
                # Update trailing stop for LONG
                trail_stop = max(trail_stop, close_i - self.atr_stop_mult * atr_i)

                # Exit conditions LONG
                exit_long = (
                    # Price hits trailing stop
                    close_i < trail_stop or
                    # EMA fast crosses below slow
                    (ema_fast.iloc[i] < ema_slow.iloc[i]) or
                    # MACD histogram turns negative
                    (macd_hist.iloc[i] < 0)
                )

                if exit_long:
                    position = 0
                    signals.iloc[i] = 0
                    trail_stop = np.nan
                else:
                    signals.iloc[i] = 1

            elif position == -1:
                # Update trailing stop for SHORT
                trail_stop = min(trail_stop, close_i + self.atr_stop_mult * atr_i)

                # Exit conditions SHORT
                exit_short = (
                    # Price hits trailing stop
                    close_i > trail_stop or
                    # EMA fast crosses above slow
                    (ema_fast.iloc[i] > ema_slow.iloc[i]) or
                    # MACD histogram turns positive
                    (macd_hist.iloc[i] > 0)
                )

                if exit_short:
                    position = 0
                    signals.iloc[i] = 0
                    trail_stop = np.nan
                else:
                    signals.iloc[i] = -1

        # Shift signals by 1 to avoid look-ahead bias
        return signals.shift(1).fillna(0).astype(int)
