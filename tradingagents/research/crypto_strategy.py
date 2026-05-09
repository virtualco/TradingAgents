import pandas as pd
import numpy as np

class CryptoDayTradingStrategy:
    def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30, ema_fast_period=9, ema_slow_period=36, volume_period=20, volume_surge_mult=1.5):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.ema_fast_period = ema_fast_period  # for 1H data
        self.ema_slow_period = ema_slow_period  # for 4H data trend filter
        self.volume_period = volume_period
        self.volume_surge_mult = volume_surge_mult

    def _rsi(self, close):
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _ema(self, series, span):
        return series.ewm(span=span, adjust=False).mean()

    def generate_signals(self, df):
        """
        df must contain columns: ['open', 'high', 'low', 'close', 'volume'] with 1H timeframe
        Returns pd.Series of signals: +1 (long), -1 (short), 0 (flat)
        """
        close = df['close']
        volume = df['volume']
        index = df.index

        # Calculate fast RSI on 1H data
        rsi = self._rsi(close)

        # Calculate slow EMA trend filter on 4H timeframe
        # Resample to 4H OHLCV
        df_4h = df.resample('4H').agg({'close':'last', 'volume':'sum'})
        ema_fast_4h = self._ema(df_4h['close'], self.ema_fast_period)
        ema_slow_4h = self._ema(df_4h['close'], self.ema_slow_period)

        # Determine 4H trend: +1 if ema_fast > ema_slow else -1
        trend_4h = pd.Series(0, index=df_4h.index)
        trend_4h[ema_fast_4h > ema_slow_4h] = 1
        trend_4h[ema_fast_4h < ema_slow_4h] = -1

        # Forward-fill 4H trend to 1H index
        trend_4h_ffill = trend_4h.reindex(index, method='ffill').fillna(0).astype(int)

        # Volume surge filter
        vol_median = volume.rolling(self.volume_period, min_periods=self.volume_period//2).median()
        vol_filter = volume > self.volume_surge_mult * vol_median

        signals = pd.Series(0, index=index)

        # Long entry: RSI crosses above oversold (30) AND 4H trend is bullish AND volume surge
        rsi_cross_up = (rsi > self.rsi_oversold) & (rsi.shift(1) <= self.rsi_oversold)
        long_entry = rsi_cross_up & (trend_4h_ffill == 1) & vol_filter

        # Short entry: RSI crosses below overbought (70) AND 4H trend is bearish AND volume surge
        rsi_cross_down = (rsi < self.rsi_overbought) & (rsi.shift(1) >= self.rsi_overbought)
        short_entry = rsi_cross_down & (trend_4h_ffill == -1) & vol_filter

        signals[long_entry] = 1
        signals[short_entry] = -1

        # Hold position until opposite signal
        signals = signals.replace(to_replace=0, method='ffill')

        # Exit long when RSI crosses back below 50 or 4H trend flips bearish
        exit_long = ((rsi < 50) | (trend_4h_ffill == -1)) & (signals == 1)
        signals[exit_long] = 0

        # Exit short when RSI crosses back above 50 or 4H trend flips bullish
        exit_short = ((rsi > 50) | (trend_4h_ffill == 1)) & (signals == -1)
        signals[exit_short] = 0

        # Shift signals by 1 to avoid lookahead bias
        signals = signals.shift(1).fillna(0).astype(int)

        return signals
