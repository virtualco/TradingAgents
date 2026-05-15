from __future__ import annotations
import numpy as np
import pandas as pd


class CryptoDayTradingStrategy:
    def __init__(
        self,
        rsi_period: int = 28,  # Longer RSI period for stability
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        rsi_exit_lower: float = 45,
        rsi_exit_upper: float = 55,
        ema_fast: int = 15,
        ema_slow: int = 40,
        trend_ema: int = 150,  # Longer trend EMA for regime filter
        macd_fast: int = 15,
        macd_slow: int = 40,
        macd_signal: int = 10,
        adx_period: int = 21,
        adx_threshold: float = 28.0,
        vol_period: int = 60,
        vol_mult: float = 1.4,
        atr_period: int = 21,
        atr_stop_mult: float = 2.3,
        bb_period: int = 60,
        bb_std_mult: float = 2.2,
        mean_rev_cooldown: int = 12,
        momentum_cooldown: int = 10,
        volatility_filter_period: int = 28,
        volatility_filter_threshold: float = 1.1,
        momentum_lookback: int = 28,
        trailing_stop_time: int = 9,
        volume_confirmation_period: int = 5,
        min_volume_multiplier: float = 1.1,
    ):
        # Momentum params
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_exit_lower = rsi_exit_lower
        self.rsi_exit_upper = rsi_exit_upper
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.trend_ema = trend_ema
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

        # Mean reversion params
        self.bb_period = bb_period
        self.bb_std_mult = bb_std_mult

        # Cooldowns to reduce overtrading
        self.mean_rev_cooldown = mean_rev_cooldown
        self.momentum_cooldown = momentum_cooldown

        # Volatility filter params
        self.volatility_filter_period = volatility_filter_period
        self.volatility_filter_threshold = volatility_filter_threshold

        # Additional params
        self.momentum_lookback = momentum_lookback
        self.trailing_stop_time = trailing_stop_time

        # Volume confirmation to reduce false signals
        self.volume_confirmation_period = volume_confirmation_period
        self.min_volume_multiplier = min_volume_multiplier

    def _rsi(self, close):
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _ema(self, s, span):
        return s.ewm(span=span, adjust=False).mean()

    def _macd(self, close):
        ml = self._ema(close, self.macd_fast) - self._ema(close, self.macd_slow)
        sl = ml.ewm(span=self.macd_signal, adjust=False).mean()
        hist = ml - sl
        return ml, sl, hist

    def _atr(self, high, low, close):
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False, min_periods=self.atr_period).mean()

    def _adx(self, high, low, close):
        up = high.diff()
        dn = -low.diff()
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index)
        ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=self.adx_period, adjust=False).mean()
        pdi = 100 * pdm.ewm(span=self.adx_period, adjust=False).mean() / atr.replace(0, np.nan)
        ndi = 100 * ndm.ewm(span=self.adx_period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
        return dx.ewm(span=self.adx_period, adjust=False).mean().fillna(0)

    def _bollinger_bands(self, close):
        sma = close.rolling(self.bb_period, min_periods=self.bb_period).mean()
        std = close.rolling(self.bb_period, min_periods=self.bb_period).std()
        upper = sma + self.bb_std_mult * std
        lower = sma - self.bb_std_mult * std
        return upper, sma, lower

    def _volatility_filter(self, close):
        returns = close.pct_change().fillna(0)
        vol = returns.rolling(self.volatility_filter_period, min_periods=self.volatility_filter_period).std()
        vol_avg = vol.rolling(self.volatility_filter_period, min_periods=self.volatility_filter_period).mean()
        return vol > (vol_avg * self.volatility_filter_threshold)

    def _momentum_score(self, close):
        # Momentum score: normalized difference EMA(30) vs EMA(70), smoothed over momentum_lookback
        ema_short = self._ema(close, 30)
        ema_long = self._ema(close, 70)
        score_raw = (ema_short - ema_long) / ema_long.replace(0, np.nan)
        score = score_raw.rolling(window=self.momentum_lookback, min_periods=self.momentum_lookback).mean()
        return score.fillna(0)

    def _volume_confirmation(self, volume):
        # Confirm volume is elevated over recent average (rolling mean)
        vol_avg = volume.rolling(self.volume_confirmation_period, min_periods=self.volume_confirmation_period).mean()
        return volume > (vol_avg * self.min_volume_multiplier)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        min_rows = max(
            self.macd_slow,
            self.trend_ema,
            self.atr_period,
            self.bb_period,
            self.vol_period,
            self.volatility_filter_period,
            self.momentum_lookback,
            self.volume_confirmation_period,
        ) + 40  # Increased buffer for longer lookbacks

        if len(df) < min_rows:
            return pd.Series(0, index=df.index)

        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

        # Indicators
        rsi = self._rsi(close)
        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        trend_ema = self._ema(close, self.trend_ema)
        macd_line, macd_signal, _ = self._macd(close)
        atr = self._atr(high, low, close)
        adx = self._adx(high, low, close)
        vol_spike = self._volume_confirmation(volume)
        upper_bb, mid_bb, lower_bb = self._bollinger_bands(close)
        volatility_elevated = self._volatility_filter(close)
        momentum_score = self._momentum_score(close)

        signals = pd.Series(0, index=df.index)
        pos = 0  # 1=long, -1=short, 0=flat
        trail = np.nan
        trail_price = np.nan
        trail_start_idx = -99999
        last_mean_rev_exit = -99999
        last_momentum_exit = -99999

        for i in range(min_rows, len(df)):
            c = float(close.iloc[i])
            a = float(atr.iloc[i])
            current_adx = float(adx.iloc[i])
            current_rsi = float(rsi.iloc[i])
            current_vol_spike = vol_spike.iloc[i]
            vol_high = volatility_elevated.iloc[i]
            ema_fast_now = float(ema_f.iloc[i])
            ema_slow_now = float(ema_s.iloc[i])
            trend_ema_now = float(trend_ema.iloc[i])
            macd_line_now = float(macd_line.iloc[i])
            macd_signal_now = float(macd_signal.iloc[i])
            mom_score_now = float(momentum_score.iloc[i])
            upper_band = float(upper_bb.iloc[i])
            lower_band = float(lower_bb.iloc[i])
            mid_band = float(mid_bb.iloc[i])

            # Regime classification: Using ADX, momentum score, and volatility-adjusted momentum magnitude
            # Higher ADX (> threshold) + strong momentum score magnitude => Trending regime
            # Low ADX or low momentum magnitude => Ranging
            mom_strength = abs(mom_score_now)
            trending = (current_adx > self.adx_threshold) and (mom_strength > 0.008)
            ranging = not trending

            # Adaptive RSI thresholds for mean reversion - tighten in elevated volatility
            if vol_high:
                mean_rev_rsi_oversold = self.rsi_oversold + 7  # More conservative
                mean_rev_rsi_overbought = self.rsi_overbought - 7
            else:
                mean_rev_rsi_oversold = self.rsi_oversold
                mean_rev_rsi_overbought = self.rsi_overbought

            # Momentum RSI thresholds tightened to reduce whipsaws
            momentum_rsi_oversold_entry = 44
            momentum_rsi_overbought_entry = 56
            momentum_rsi_exit_lower = self.rsi_exit_lower
            momentum_rsi_exit_upper = self.rsi_exit_upper

            # Initialize signals for this bar
            long_entry = False
            short_entry = False
            exit_position = False

            # MOMENTUM regime (trending)
            if trending:
                if (pos == 0) and (i - last_momentum_exit > self.momentum_cooldown):
                    # Require volume confirmation and no volatility spike on entry
                    cond_long = (
                        (c > trend_ema_now)
                        and (ema_fast_now > ema_slow_now)
                        and (macd_line_now > macd_signal_now)
                        and (momentum_rsi_oversold_entry < current_rsi < momentum_rsi_overbought_entry)
                        and current_vol_spike
                        and (mom_score_now > 0)
                        and (not vol_high)  # avoid entries on volatility spikes
                    )
                    cond_short = (
                        (c < trend_ema_now)
                        and (ema_fast_now < ema_slow_now)
                        and (macd_line_now < macd_signal_now)
                        and (momentum_rsi_oversold_entry < (100 - current_rsi) < momentum_rsi_overbought_entry)
                        and current_vol_spike
                        and (mom_score_now < 0)
                        and (not vol_high)
                    )
                    long_entry = cond_long
                    short_entry = cond_short

                # Exits for momentum trades: ATR trailing stop + time-based trailing stop + signal invalidation + volatility spike exit
                if pos == 1:
                    if np.isnan(trail):
                        trail = c - self.atr_stop_mult * a if a > 0 else c * 0.985
                        trail_price = c
                        trail_start_idx = i
                    else:
                        trail = max(trail, c - self.atr_stop_mult * a)
                        bars_held = i - trail_start_idx
                        if bars_held >= self.trailing_stop_time:
                            locked_stop = trail_price + 0.75 * (c - trail_price)  # Lock more profit at time exit
                            trail = max(trail, locked_stop)
                    exit_cond = (
                        (c < trail)
                        or (ema_fast_now <= ema_slow_now)
                        or (macd_line_now <= macd_signal_now)
                        or (c < trend_ema_now)
                        or (current_rsi > momentum_rsi_exit_upper)
                        or vol_high
                    )
                    if exit_cond:
                        exit_position = True
                        last_momentum_exit = i
                        trail = np.nan
                        trail_price = np.nan
                        trail_start_idx = -99999

                elif pos == -1:
                    if np.isnan(trail):
                        trail = c + self.atr_stop_mult * a if a > 0 else c * 1.015
                        trail_price = c
                        trail_start_idx = i
                    else:
                        trail = min(trail, c + self.atr_stop_mult * a)
                        bars_held = i - trail_start_idx
                        if bars_held >= self.trailing_stop_time:
                            locked_stop = trail_price - 0.75 * (trail_price - c)
                            trail = min(trail, locked_stop)
                    exit_cond = (
                        (c > trail)
                        or (ema_fast_now >= ema_slow_now)
                        or (macd_line_now >= macd_signal_now)
                        or (c > trend_ema_now)
                        or (current_rsi < momentum_rsi_exit_lower)
                        or vol_high
                    )
                    if exit_cond:
                        exit_position = True
                        last_momentum_exit = i
                        trail = np.nan
                        trail_price = np.nan
                        trail_start_idx = -99999

            # MEAN-REVERSION regime (ranging)
            else:
                if (pos == 0) and (i - last_mean_rev_exit > self.mean_rev_cooldown):
                    # Mean reversion entries do NOT require volume spike (to catch low volume reversals)
                    # But avoid entering on volatility spikes or immediately after a volume spike (to filter fake breakouts)
                    recent_vol_spike = vol_spike.iloc[max(i - self.volume_confirmation_period, 0):i].any()
                    if (not vol_high) and (not recent_vol_spike):
                        cond_long = (c < lower_band) and (current_rsi < mean_rev_rsi_oversold)
                        cond_short = (c > upper_band) and (current_rsi > mean_rev_rsi_overbought)
                        long_entry = cond_long
                        short_entry = cond_short
                    else:
                        long_entry = False
                        short_entry = False

                # Exit mean reversion when price reverts to mid BB or RSI normalizes (rsi_exit_lower < rsi < rsi_exit_upper) or vol spikes
                if pos == 1:
                    if (c >= mid_band) or (self.rsi_exit_lower < current_rsi < self.rsi_exit_upper) or vol_high:
                        exit_position = True
                        last_mean_rev_exit = i
                elif pos == -1:
                    if (c <= mid_band) or (self.rsi_exit_lower < current_rsi < self.rsi_exit_upper) or vol_high:
                        exit_position = True
                        last_mean_rev_exit = i

            # Position management and signal assignment
            if pos == 0:
                if long_entry:
                    pos = 1
                    trail = c - self.atr_stop_mult * a if a > 0 else c * 0.985
                    trail_price = c
                    trail_start_idx = i
                    signals.iloc[i] = 1
                elif short_entry:
                    pos = -1
                    trail = c + self.atr_stop_mult * a if a > 0 else c * 1.015
                    trail_price = c
                    trail_start_idx = i
                    signals.iloc[i] = -1
                else:
                    signals.iloc[i] = 0
            elif exit_position:
                pos = 0
                trail = np.nan
                trail_price = np.nan
                trail_start_idx = -99999
                signals.iloc[i] = 0
            else:
                signals.iloc[i] = pos

        return signals.shift(1).fillna(0).astype(int)