"""Role-specific strategy rules for the TradingAgents Research Factory.

Each analyst role uses a distinct rule-based signal generation approach:
- TechnicalAnalyst: SMA crossovers, RSI, MACD, Bollinger Bands
- FundamentalAnalyst: P/E, P/B, revenue growth, FCF yield, earnings surprise
- SentimentAnalyst: news sentiment score, volume anomalies, social momentum
- MacroAnalyst: sector rotation, yield curve, VIX regime

These rules serve two purposes:
1. Quick Mode backtesting — generate signals without LLM calls (fast, cheap)
2. LLM context enrichment — pre-compute quantitative signals before the
   debate stage so agents reason from data, not just text

All rules are time-safe: they only use data with event_time <= trade_date.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------

class SignalStrength(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class RuleSignal:
    """Output from a single strategy rule."""
    rule_name: str
    role: str
    signal: SignalStrength
    score: float          # -1.0 (strong sell) to +1.0 (strong buy)
    confidence: float     # 0.0 to 1.0
    rationale: str
    data_points: Dict = field(default_factory=dict)


@dataclass
class RoleSignalSummary:
    """Aggregated signals from all rules for a given analyst role."""
    role: str
    ticker: str
    trade_date: str
    composite_score: float   # -1.0 to +1.0
    composite_signal: SignalStrength
    rule_signals: List[RuleSignal] = field(default_factory=list)
    context_summary: str = ""  # Human-readable summary for LLM context

    def to_context_string(self) -> str:
        """Format for injection into LLM analyst prompts."""
        lines = [
            f"## Quantitative Pre-Analysis: {self.role} ({self.ticker}, {self.trade_date})",
            f"**Composite Signal**: {self.composite_signal.value} "
            f"(score: {self.composite_score:+.2f})",
            "",
            "**Rule Breakdown**:",
        ]
        for sig in self.rule_signals:
            lines.append(
                f"- {sig.rule_name}: {sig.signal.value} "
                f"(score: {sig.score:+.2f}, confidence: {sig.confidence:.0%}) "
                f"— {sig.rationale}"
            )
        return "\n".join(lines)


def _score_to_signal(score: float) -> SignalStrength:
    """Convert a numeric score to a SignalStrength enum."""
    if score >= 0.6:
        return SignalStrength.STRONG_BUY
    elif score >= 0.2:
        return SignalStrength.BUY
    elif score >= -0.2:
        return SignalStrength.NEUTRAL
    elif score >= -0.6:
        return SignalStrength.SELL
    else:
        return SignalStrength.STRONG_SELL


# ---------------------------------------------------------------------------
# Technical Analyst Rules
# ---------------------------------------------------------------------------

class TechnicalStrategyRules:
    """Rule-based signals for the Technical Analyst role.

    Uses: SMA crossovers, RSI, MACD, Bollinger Band position.
    Requires: OHLCV DataFrame with at least 50 rows.
    """

    ROLE = "TechnicalAnalyst"
    MIN_ROWS = 20  # Minimum rows needed for any signal

    def compute(
        self,
        ohlcv: pd.DataFrame,
        ticker: str,
        trade_date: str,
    ) -> RoleSignalSummary:
        """Compute all technical signals and return a summary."""
        signals = []

        if ohlcv.empty or len(ohlcv) < self.MIN_ROWS:
            return RoleSignalSummary(
                role=self.ROLE,
                ticker=ticker,
                trade_date=trade_date,
                composite_score=0.0,
                composite_signal=SignalStrength.NEUTRAL,
                context_summary="Insufficient price history for technical analysis.",
            )

        close = ohlcv["close"].dropna().reset_index(drop=True)

        signals.append(self._sma_crossover(close))
        signals.append(self._rsi_signal(close))
        signals.append(self._macd_signal(close))
        signals.append(self._bollinger_signal(close))

        # Filter out None signals
        signals = [s for s in signals if s is not None]

        if not signals:
            composite_score = 0.0
        else:
            # Weighted average (confidence-weighted)
            total_weight = sum(s.confidence for s in signals)
            composite_score = (
                sum(s.score * s.confidence for s in signals) / total_weight
                if total_weight > 0 else 0.0
            )

        summary = RoleSignalSummary(
            role=self.ROLE,
            ticker=ticker,
            trade_date=trade_date,
            composite_score=round(composite_score, 3),
            composite_signal=_score_to_signal(composite_score),
            rule_signals=signals,
        )
        summary.context_summary = summary.to_context_string()
        return summary

    def _sma_crossover(self, close: pd.Series) -> Optional[RuleSignal]:
        """Golden/death cross: 20-day SMA vs 50-day SMA."""
        if len(close) < 50:
            return None
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        prev_sma20 = close.rolling(20).mean().iloc[-2]
        prev_sma50 = close.rolling(50).mean().iloc[-2]

        if pd.isna(sma20) or pd.isna(sma50):
            return None

        gap_pct = (sma20 - sma50) / sma50
        # tanh scaling: 1% gap -> 0.20, 2% gap -> 0.38, 3% gap -> 0.54, 5%+ -> ~0.60
        scaled = round(min(0.6, math.tanh(abs(gap_pct) * 20) * 0.6), 3)
        # Golden cross: SMA20 just crossed above SMA50
        if prev_sma20 <= prev_sma50 and sma20 > sma50:
            score, rationale = 0.8, f"Golden cross: SMA20 ({sma20:.2f}) crossed above SMA50 ({sma50:.2f})"
        elif sma20 > sma50:
            score = scaled
            rationale = f"Bullish: SMA20 ({sma20:.2f}) > SMA50 ({sma50:.2f}), gap {gap_pct:.1%}"
        elif prev_sma20 >= prev_sma50 and sma20 < sma50:
            score, rationale = -0.8, f"Death cross: SMA20 ({sma20:.2f}) crossed below SMA50 ({sma50:.2f})"
        else:
            score = -scaled
            rationale = f"Bearish: SMA20 ({sma20:.2f}) < SMA50 ({sma50:.2f}), gap {gap_pct:.1%}"

        return RuleSignal(
            rule_name="SMA_Crossover",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.70,
            rationale=rationale,
            data_points={"sma20": round(sma20, 2), "sma50": round(sma50, 2)},
        )

    def _rsi_signal(self, close: pd.Series) -> Optional[RuleSignal]:
        """RSI(14) overbought/oversold signal."""
        if len(close) < 15:
            return None
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]

        if pd.isna(rsi_val):
            return None

        if rsi_val < 30:
            score = 0.7
            rationale = f"RSI oversold at {rsi_val:.1f} — potential reversal"
        elif rsi_val < 40:
            score = 0.3
            rationale = f"RSI approaching oversold at {rsi_val:.1f}"
        elif rsi_val > 70:
            score = -0.7
            rationale = f"RSI overbought at {rsi_val:.1f} — potential pullback"
        elif rsi_val > 60:
            score = -0.3
            rationale = f"RSI approaching overbought at {rsi_val:.1f}"
        elif rsi_val > 50:
            # Weak bullish momentum: linear 0.0 at 50 → 0.15 at 60
            score = round((rsi_val - 50) / 10 * 0.15, 3)
            rationale = f"RSI mild bullish momentum at {rsi_val:.1f}"
        elif rsi_val < 50:
            # Weak bearish momentum: linear 0.0 at 50 → -0.15 at 40
            score = round((rsi_val - 50) / 10 * 0.15, 3)
            rationale = f"RSI mild bearish momentum at {rsi_val:.1f}"
        else:
            score = 0.0
            rationale = f"RSI neutral at {rsi_val:.1f}"

        return RuleSignal(
            rule_name="RSI_14",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.65,
            rationale=rationale,
            data_points={"rsi": round(rsi_val, 1)},
        )

    def _macd_signal(self, close: pd.Series) -> Optional[RuleSignal]:
        """MACD(12,26,9) signal line crossover."""
        if len(close) < 35:
            return None
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal_line

        hist_now = histogram.iloc[-1]
        hist_prev = histogram.iloc[-2]

        if pd.isna(hist_now):
            return None

        # Histogram momentum: growing positive = bullish, growing negative = bearish
        if hist_now > 0 and hist_now > hist_prev:
            score = 0.6
            rationale = f"MACD histogram growing positive ({hist_now:.3f})"
        elif hist_now > 0:
            score = 0.2
            rationale = f"MACD histogram positive but declining ({hist_now:.3f})"
        elif hist_now < 0 and hist_now < hist_prev:
            score = -0.6
            rationale = f"MACD histogram growing negative ({hist_now:.3f})"
        else:
            score = -0.2
            rationale = f"MACD histogram negative but recovering ({hist_now:.3f})"

        return RuleSignal(
            rule_name="MACD",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.75,  # MACD is a reliable momentum indicator
            rationale=rationale,
            data_points={"macd_histogram": round(hist_now, 4)},
        )

    def _bollinger_signal(self, close: pd.Series) -> Optional[RuleSignal]:
        """Bollinger Band position signal."""
        if len(close) < 20:
            return None
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std

        price = close.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        sma_val = sma.iloc[-1]

        if pd.isna(upper_val) or pd.isna(lower_val):
            return None

        band_width = upper_val - lower_val
        position = (price - lower_val) / band_width if band_width > 0 else 0.5
        # Trend-aware Bollinger: detect if price is in an uptrend (5-day slope > 0)
        trend_slope = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0.0
        in_uptrend = trend_slope > 0.005   # >0.5% gain over 5 days
        in_downtrend = trend_slope < -0.005
        if position < 0.1:
            score = 0.7
            rationale = f"Price near lower Bollinger Band ({price:.2f} vs lower {lower_val:.2f})"
        elif position < 0.3:
            score = 0.3
            rationale = f"Price in lower Bollinger zone (position: {position:.0%})"
        elif position > 0.9:
            if in_uptrend:
                score = 0.5   # Upper band in uptrend = momentum confirmation
                rationale = f"Price riding upper Bollinger Band in uptrend ({price:.2f})"
            else:
                score = -0.7
                rationale = f"Price near upper Bollinger Band ({price:.2f} vs upper {upper_val:.2f})"
        elif position > 0.7:
            if in_uptrend:
                score = 0.2   # Upper zone in uptrend = mild bullish
                rationale = f"Price in upper Bollinger zone with uptrend (position: {position:.0%})"
            elif in_downtrend:
                score = -0.5  # Upper zone in downtrend = bearish divergence
                rationale = f"Price in upper Bollinger zone despite downtrend (position: {position:.0%})"
            else:
                score = -0.3
                rationale = f"Price in upper Bollinger zone (position: {position:.0%})"
        else:
            score = 0.0
            rationale = f"Price mid-Bollinger Band (position: {position:.0%})"

        return RuleSignal(
            rule_name="BollingerBand",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.65,  # Bollinger is a reliable volatility/trend indicator
            rationale=rationale,
            data_points={"bb_position": round(position, 3), "price": round(price, 2)},
        )


# ---------------------------------------------------------------------------
# Fundamental Analyst Rules
# ---------------------------------------------------------------------------

class FundamentalStrategyRules:
    """Rule-based signals for the Fundamental Analyst role.

    Uses: Revenue growth, EPS trend, FCF yield, P/E relative to sector.
    Requires: Fundamentals DataFrame.
    """

    ROLE = "FundamentalAnalyst"

    def compute(
        self,
        fundamentals: pd.DataFrame,
        ticker: str,
        trade_date: str,
        market_cap: Optional[float] = None,
    ) -> RoleSignalSummary:
        """Compute all fundamental signals."""
        signals = []

        if fundamentals.empty:
            return RoleSignalSummary(
                role=self.ROLE,
                ticker=ticker,
                trade_date=trade_date,
                composite_score=0.0,
                composite_signal=SignalStrength.NEUTRAL,
                context_summary="No fundamentals data available.",
            )

        latest = fundamentals.sort_values("available_at", ascending=False).iloc[0]

        signals.append(self._revenue_signal(latest))
        signals.append(self._earnings_signal(latest))
        signals.append(self._fcf_signal(latest, market_cap))
        signals.append(self._balance_sheet_signal(latest))

        signals = [s for s in signals if s is not None]

        if not signals:
            composite_score = 0.0
        else:
            total_weight = sum(s.confidence for s in signals)
            composite_score = (
                sum(s.score * s.confidence for s in signals) / total_weight
                if total_weight > 0 else 0.0
            )

        summary = RoleSignalSummary(
            role=self.ROLE,
            ticker=ticker,
            trade_date=trade_date,
            composite_score=round(composite_score, 3),
            composite_signal=_score_to_signal(composite_score),
            rule_signals=signals,
        )
        summary.context_summary = summary.to_context_string()
        return summary

    def _revenue_signal(self, row: pd.Series) -> Optional[RuleSignal]:
        """Revenue presence and magnitude signal."""
        revenue = row.get("revenue")
        if revenue is None or pd.isna(revenue):
            return None

        # Simple sanity: positive revenue is baseline
        if revenue > 0:
            score = 0.3
            rationale = f"Revenue ${revenue/1e9:.1f}B — positive"
        else:
            score = -0.5
            rationale = f"Revenue negative or zero: ${revenue/1e9:.1f}B"

        return RuleSignal(
            rule_name="Revenue",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.50,
            rationale=rationale,
            data_points={"revenue_bn": round(revenue / 1e9, 2) if revenue else None},
        )

    def _earnings_signal(self, row: pd.Series) -> Optional[RuleSignal]:
        """EPS signal."""
        eps = row.get("eps")
        if eps is None or pd.isna(eps):
            return None

        if eps > 5.0:
            score, rationale = 0.7, f"Strong EPS: ${eps:.2f}"
        elif eps > 2.0:
            score, rationale = 0.4, f"Solid EPS: ${eps:.2f}"
        elif eps > 0:
            score, rationale = 0.1, f"Positive EPS: ${eps:.2f}"
        elif eps > -1.0:
            score, rationale = -0.3, f"Slightly negative EPS: ${eps:.2f}"
        else:
            score, rationale = -0.7, f"Significantly negative EPS: ${eps:.2f}"

        return RuleSignal(
            rule_name="EPS",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.65,
            rationale=rationale,
            data_points={"eps": round(eps, 2)},
        )

    def _fcf_signal(self, row: pd.Series, market_cap: Optional[float]) -> Optional[RuleSignal]:
        """Free cash flow yield signal."""
        fcf = row.get("free_cash_flow")
        if fcf is None or pd.isna(fcf) or market_cap is None or market_cap <= 0:
            return None

        fcf_yield = fcf / market_cap

        if fcf_yield > 0.06:
            score, rationale = 0.8, f"High FCF yield: {fcf_yield:.1%}"
        elif fcf_yield > 0.03:
            score, rationale = 0.4, f"Decent FCF yield: {fcf_yield:.1%}"
        elif fcf_yield > 0:
            score, rationale = 0.1, f"Positive FCF yield: {fcf_yield:.1%}"
        else:
            score, rationale = -0.5, f"Negative FCF yield: {fcf_yield:.1%}"

        return RuleSignal(
            rule_name="FCF_Yield",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.70,
            rationale=rationale,
            data_points={"fcf_yield": round(fcf_yield, 4)},
        )

    def _balance_sheet_signal(self, row: pd.Series) -> Optional[RuleSignal]:
        """Debt-to-equity signal."""
        assets = row.get("total_assets")
        liabilities = row.get("total_liabilities")
        equity = row.get("total_equity")

        if any(v is None or pd.isna(v) for v in [assets, liabilities, equity]):
            return None
        if equity <= 0:
            return None

        debt_to_equity = liabilities / equity

        if debt_to_equity < 0.5:
            score, rationale = 0.5, f"Low leverage: D/E = {debt_to_equity:.2f}"
        elif debt_to_equity < 1.5:
            score, rationale = 0.1, f"Moderate leverage: D/E = {debt_to_equity:.2f}"
        elif debt_to_equity < 3.0:
            score, rationale = -0.3, f"High leverage: D/E = {debt_to_equity:.2f}"
        else:
            score, rationale = -0.7, f"Very high leverage: D/E = {debt_to_equity:.2f}"

        return RuleSignal(
            rule_name="DebtToEquity",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.60,
            rationale=rationale,
            data_points={"debt_to_equity": round(debt_to_equity, 2)},
        )


# ---------------------------------------------------------------------------
# Sentiment Analyst Rules
# ---------------------------------------------------------------------------

class SentimentStrategyRules:
    """Rule-based signals for the Sentiment Analyst role.

    Uses: News sentiment scores, volume anomalies, price momentum.
    """

    ROLE = "SentimentAnalyst"

    def compute(
        self,
        news: pd.DataFrame,
        ohlcv: pd.DataFrame,
        ticker: str,
        trade_date: str,
    ) -> RoleSignalSummary:
        """Compute all sentiment signals."""
        signals = []

        signals.append(self._news_sentiment_signal(news))
        signals.append(self._volume_anomaly_signal(ohlcv))
        signals.append(self._price_momentum_signal(ohlcv))

        signals = [s for s in signals if s is not None]

        if not signals:
            composite_score = 0.0
        else:
            total_weight = sum(s.confidence for s in signals)
            composite_score = (
                sum(s.score * s.confidence for s in signals) / total_weight
                if total_weight > 0 else 0.0
            )

        summary = RoleSignalSummary(
            role=self.ROLE,
            ticker=ticker,
            trade_date=trade_date,
            composite_score=round(composite_score, 3),
            composite_signal=_score_to_signal(composite_score),
            rule_signals=signals,
        )
        summary.context_summary = summary.to_context_string()
        return summary

    def _news_sentiment_signal(self, news: pd.DataFrame) -> Optional[RuleSignal]:
        """Aggregate news sentiment scores."""
        if news.empty or "sentiment_score" not in news.columns:
            # If no scored news, neutral
            if not news.empty:
                return RuleSignal(
                    rule_name="NewsSentiment",
                    role=self.ROLE,
                    signal=SignalStrength.NEUTRAL,
                    score=0.0,
                    confidence=0.30,
                    rationale=f"{len(news)} news articles found but no sentiment scores",
                )
            return None

        scores = news["sentiment_score"].dropna()
        if scores.empty:
            return None

        avg_sentiment = scores.mean()
        article_count = len(news)

        # Scale confidence by article count
        confidence = min(0.80, 0.40 + article_count * 0.02)

        if avg_sentiment > 0.3:
            score, rationale = avg_sentiment * 0.8, f"Positive news sentiment ({avg_sentiment:.2f}, {article_count} articles)"
        elif avg_sentiment > 0:
            score, rationale = avg_sentiment * 0.5, f"Mildly positive sentiment ({avg_sentiment:.2f}, {article_count} articles)"
        elif avg_sentiment > -0.3:
            score, rationale = avg_sentiment * 0.5, f"Mildly negative sentiment ({avg_sentiment:.2f}, {article_count} articles)"
        else:
            score, rationale = avg_sentiment * 0.8, f"Negative news sentiment ({avg_sentiment:.2f}, {article_count} articles)"

        return RuleSignal(
            rule_name="NewsSentiment",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=round(confidence, 2),
            rationale=rationale,
            data_points={"avg_sentiment": round(avg_sentiment, 3), "articles": article_count},
        )

    def _volume_anomaly_signal(self, ohlcv: pd.DataFrame) -> Optional[RuleSignal]:
        """Detect unusual volume as a sentiment proxy."""
        if ohlcv.empty or "volume" not in ohlcv.columns or len(ohlcv) < 10:
            return None

        vol = ohlcv["volume"].dropna()
        avg_vol = vol.iloc[:-1].mean()
        recent_vol = vol.iloc[-1]

        if avg_vol <= 0:
            return None

        vol_ratio = recent_vol / avg_vol

        if vol_ratio > 2.5:
            score = 0.5
            rationale = f"Volume spike: {vol_ratio:.1f}x average — high interest"
        elif vol_ratio > 1.5:
            score = 0.2
            rationale = f"Above-average volume: {vol_ratio:.1f}x"
        elif vol_ratio < 0.4:
            score = -0.2
            rationale = f"Very low volume: {vol_ratio:.1f}x average — low conviction"
        else:
            score = 0.0
            rationale = f"Normal volume: {vol_ratio:.1f}x average"

        return RuleSignal(
            rule_name="VolumeAnomaly",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.45,
            rationale=rationale,
            data_points={"volume_ratio": round(vol_ratio, 2)},
        )

    def _price_momentum_signal(self, ohlcv: pd.DataFrame) -> Optional[RuleSignal]:
        """5-day and 20-day price momentum."""
        if ohlcv.empty or "close" not in ohlcv.columns or len(ohlcv) < 20:
            return None

        close = ohlcv["close"].dropna()
        mom5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) >= 6 else None
        mom20 = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if len(close) >= 21 else None

        if mom5 is None:
            return None

        # Combine 5-day and 20-day momentum
        if mom20 is not None:
            score = (mom5 * 0.6 + mom20 * 0.4) * 2  # Scale to -1..+1 range
        else:
            score = mom5 * 2

        score = max(-1.0, min(1.0, score))
        rationale = f"5d momentum: {mom5:.1%}"
        if mom20 is not None:
            rationale += f", 20d momentum: {mom20:.1%}"

        return RuleSignal(
            rule_name="PriceMomentum",
            role=self.ROLE,
            signal=_score_to_signal(score),
            score=round(score, 3),
            confidence=0.55,
            rationale=rationale,
            data_points={"mom5d": round(mom5, 4), "mom20d": round(mom20, 4) if mom20 else None},
        )


# ---------------------------------------------------------------------------
# Convenience: compute all roles at once
# ---------------------------------------------------------------------------

@dataclass
class MultiRoleSignals:
    """Combined signals from all analyst roles."""
    ticker: str
    trade_date: str
    technical: Optional[RoleSignalSummary] = None
    fundamental: Optional[RoleSignalSummary] = None
    sentiment: Optional[RoleSignalSummary] = None

    @property
    def ensemble_score(self) -> float:
        """Equal-weight ensemble of available role scores."""
        scores = []
        if self.technical:
            scores.append(self.technical.composite_score)
        if self.fundamental:
            scores.append(self.fundamental.composite_score)
        if self.sentiment:
            scores.append(self.sentiment.composite_score)
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    @property
    def ensemble_signal(self) -> SignalStrength:
        return _score_to_signal(self.ensemble_score)

    def to_context_string(self) -> str:
        """Full context string for injection into LLM debate prompts."""
        parts = [
            f"# Pre-Analysis Quantitative Signals: {self.ticker} ({self.trade_date})",
            f"**Ensemble Score**: {self.ensemble_score:+.2f} → {self.ensemble_signal.value.upper()}",
            "",
        ]
        for role_summary in [self.technical, self.fundamental, self.sentiment]:
            if role_summary:
                parts.append(role_summary.to_context_string())
                parts.append("")
        return "\n".join(parts)


def compute_multi_role_signals(
    ticker: str,
    trade_date: str,
    ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame,
    news: pd.DataFrame,
    market_cap: Optional[float] = None,
    active_roles: Optional[List[str]] = None,
) -> MultiRoleSignals:
    """Compute signals for all active analyst roles.

    Args:
        ticker: Instrument ticker symbol.
        trade_date: The date we're trading on.
        ohlcv: OHLCV DataFrame (time-safe, as of trade_date).
        fundamentals: Fundamentals DataFrame (time-safe).
        news: News DataFrame (time-safe).
        market_cap: Market cap for FCF yield calculation.
        active_roles: List of role names to compute. None = all roles.

    Returns:
        MultiRoleSignals with signals from each active role.
    """
    if active_roles is None:
        active_roles = ["TechnicalAnalyst", "FundamentalAnalyst", "SentimentAnalyst"]

    result = MultiRoleSignals(ticker=ticker, trade_date=trade_date)

    if "TechnicalAnalyst" in active_roles:
        result.technical = TechnicalStrategyRules().compute(ohlcv, ticker, trade_date)

    if "FundamentalAnalyst" in active_roles:
        result.fundamental = FundamentalStrategyRules().compute(
            fundamentals, ticker, trade_date, market_cap
        )

    if "SentimentAnalyst" in active_roles:
        result.sentiment = SentimentStrategyRules().compute(news, ohlcv, ticker, trade_date)

    return result
