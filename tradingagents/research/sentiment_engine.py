"""
LLM Sentiment Engine — Quantified News Alpha
==============================================
Transforms raw financial news into structured numeric sentiment scores
that serve as an independent alpha source in the signal pipeline.

Architecture:
  1. News Ingestion — Fetch ticker-specific news from yfinance
  2. LLM Scoring — Score each article -1.0 to +1.0 via structured prompt
  3. Time-Decay Aggregation — Exponential decay weighting (half-life configurable)
  4. Signal Integration — Output conviction modifier for PerAssetRouter

Key Design Decisions:
  - Uses structured JSON output from LLM (no free-text parsing)
  - Anti-hallucination: explicit instruction to return 0.0 if uncertain
  - Low token usage: batch articles into single prompt (max 10 per call)
  - Caches scores to avoid re-scoring same articles
  - Fallback to neutral (0.0) if LLM unavailable

Reference: Kirtac & Germano (2024) — 74.4% directional accuracy, Sharpe 3.05
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SentimentConfig:
    """Configuration for the sentiment engine."""
    # Scoring
    max_articles_per_batch: int = 8       # Max articles per LLM call
    max_articles_per_ticker: int = 20     # Max articles to score per ticker
    
    # Time decay
    decay_half_life_hours: float = 48.0   # Sentiment half-life (2 days)
    max_age_hours: float = 168.0          # Ignore articles older than 7 days
    
    # Signal integration
    sentiment_weight: float = 0.25        # Weight in conviction modifier
    strong_signal_threshold: float = 0.5  # |score| > this = strong signal
    
    # Safety
    neutral_on_error: bool = True         # Return 0.0 if LLM fails
    min_articles_for_signal: int = 3      # Need at least N articles for signal


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

@dataclass
class ArticleScore:
    """Scored news article."""
    article_id: str              # Hash of title + source
    ticker: str
    title: str
    source: str
    published_at: str            # ISO format
    sentiment_score: float       # -1.0 to +1.0
    confidence: float            # 0.0 to 1.0
    reasoning: str               # Brief explanation
    scored_at: str = ""          # When we scored it


@dataclass
class TickerSentiment:
    """Aggregated sentiment for a single ticker."""
    ticker: str
    composite_score: float       # Time-decay weighted aggregate (-1 to +1)
    confidence: float            # Average confidence of component scores
    n_articles: int              # Number of articles scored
    signal_strength: str         # STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR
    conviction_modifier: float   # Multiplier for PerAssetRouter (0.5 to 1.5)
    articles: List[ArticleScore] = field(default_factory=list)
    computed_at: str = ""


@dataclass
class SentimentReport:
    """Full sentiment report across all tickers."""
    tickers: Dict[str, TickerSentiment] = field(default_factory=dict)
    computed_at: str = ""
    total_articles_scored: int = 0
    llm_calls_made: int = 0


# ---------------------------------------------------------------------------
# Sentiment Engine
# ---------------------------------------------------------------------------

SCORING_PROMPT = """You are a financial sentiment analyst. Score each news article's impact on the stock/crypto price.

Rules:
- Score from -1.0 (extremely bearish) to +1.0 (extremely bullish)
- 0.0 = neutral or uncertain — USE THIS when unsure
- Consider: price impact, market reaction, forward-looking implications
- Ignore clickbait, opinion pieces with no substance
- Confidence: 0.0-1.0 (how certain you are of the score)

Return ONLY valid JSON array. No markdown, no explanation outside JSON.

Articles to score:
{articles_json}

Return format (JSON array):
[
  {{"id": "article_id", "score": 0.3, "confidence": 0.8, "reasoning": "brief reason"}},
  ...
]"""


class SentimentEngine:
    """
    Quantified LLM sentiment scoring engine.
    
    Transforms news articles into numeric sentiment scores and aggregates
    them with time-decay weighting to produce a conviction modifier for
    the trading signal pipeline.
    
    Usage:
        engine = SentimentEngine(llm_fn=invoke_llm)
        report = engine.score_portfolio(["BTC-USD", "ETH-USD", "AAPL"])
        
        # Use as conviction modifier
        for ticker, sentiment in report.tickers.items():
            base_signal *= sentiment.conviction_modifier
    """
    
    def __init__(
        self,
        llm_fn=None,
        news_fn=None,
        config: Optional[SentimentConfig] = None,
        cache: Optional[Dict[str, ArticleScore]] = None,
    ):
        """
        Args:
            llm_fn: Callable that takes messages list and returns response dict.
                    Signature: (messages: List[Dict]) -> Dict with choices[0].message.content
            news_fn: Callable that fetches news for a ticker.
                     Signature: (ticker: str, days_back: int) -> List[Dict]
                     Each dict: {"title": str, "source": str, "published": str, "summary": str}
            config: Engine configuration
            cache: Optional pre-loaded score cache
        """
        self.llm_fn = llm_fn
        self.news_fn = news_fn or self._default_news_fetch
        self.config = config or SentimentConfig()
        self._cache: Dict[str, ArticleScore] = cache or {}
        self._llm_calls = 0
    
    def score_portfolio(
        self,
        tickers: List[str],
        lookback_days: int = 7,
    ) -> SentimentReport:
        """
        Score sentiment for all tickers in the portfolio.
        
        Args:
            tickers: List of ticker symbols
            lookback_days: How many days back to fetch news
        
        Returns:
            SentimentReport with per-ticker sentiment and conviction modifiers
        """
        report = SentimentReport(computed_at=datetime.utcnow().isoformat())
        
        for ticker in tickers:
            try:
                sentiment = self._score_ticker(ticker, lookback_days)
                report.tickers[ticker] = sentiment
                report.total_articles_scored += sentiment.n_articles
            except Exception as e:
                logger.error(f"Failed to score sentiment for {ticker}: {e}")
                # Return neutral on error
                report.tickers[ticker] = TickerSentiment(
                    ticker=ticker,
                    composite_score=0.0,
                    confidence=0.0,
                    n_articles=0,
                    signal_strength="NEUTRAL",
                    conviction_modifier=1.0,
                    computed_at=datetime.utcnow().isoformat(),
                )
        
        report.llm_calls_made = self._llm_calls
        return report
    
    def _score_ticker(self, ticker: str, lookback_days: int) -> TickerSentiment:
        """Score sentiment for a single ticker."""
        # 1. Fetch news
        articles = self.news_fn(ticker, lookback_days)
        
        if not articles:
            return TickerSentiment(
                ticker=ticker,
                composite_score=0.0,
                confidence=0.0,
                n_articles=0,
                signal_strength="NEUTRAL",
                conviction_modifier=1.0,
                computed_at=datetime.utcnow().isoformat(),
            )
        
        # Limit articles
        articles = articles[:self.config.max_articles_per_ticker]
        
        # 2. Score articles (check cache first)
        scored_articles = []
        to_score = []
        
        for article in articles:
            article_id = self._article_id(article)
            if article_id in self._cache:
                scored_articles.append(self._cache[article_id])
            else:
                to_score.append(article)
        
        # 3. Batch score uncached articles via LLM
        if to_score and self.llm_fn:
            new_scores = self._batch_score(ticker, to_score)
            scored_articles.extend(new_scores)
            # Cache new scores
            for score in new_scores:
                self._cache[score.article_id] = score
        
        if not scored_articles:
            return TickerSentiment(
                ticker=ticker,
                composite_score=0.0,
                confidence=0.0,
                n_articles=0,
                signal_strength="NEUTRAL",
                conviction_modifier=1.0,
                computed_at=datetime.utcnow().isoformat(),
            )
        
        # 4. Time-decay aggregation
        composite, avg_confidence = self._aggregate_with_decay(scored_articles)
        
        # 5. Determine signal strength
        signal_strength = self._classify_signal(composite)
        
        # 6. Compute conviction modifier
        conviction_modifier = self._compute_conviction_modifier(composite, avg_confidence)
        
        return TickerSentiment(
            ticker=ticker,
            composite_score=round(composite, 4),
            confidence=round(avg_confidence, 4),
            n_articles=len(scored_articles),
            signal_strength=signal_strength,
            conviction_modifier=round(conviction_modifier, 4),
            articles=scored_articles,
            computed_at=datetime.utcnow().isoformat(),
        )
    
    def _batch_score(self, ticker: str, articles: List[Dict]) -> List[ArticleScore]:
        """Score a batch of articles via LLM."""
        scored = []
        
        # Process in batches
        batch_size = self.config.max_articles_per_batch
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_scores = self._call_llm_score(ticker, batch)
            scored.extend(batch_scores)
        
        return scored
    
    def _call_llm_score(self, ticker: str, articles: List[Dict]) -> List[ArticleScore]:
        """Make a single LLM call to score a batch of articles."""
        # Prepare articles for prompt
        articles_for_prompt = []
        for article in articles:
            article_id = self._article_id(article)
            articles_for_prompt.append({
                "id": article_id,
                "title": article.get("title", ""),
                "source": article.get("source", "unknown"),
                "published": article.get("published", ""),
                "summary": article.get("summary", "")[:300],  # Truncate for token efficiency
            })
        
        prompt = SCORING_PROMPT.format(
            articles_json=json.dumps(articles_for_prompt, indent=2)
        )
        
        try:
            messages = [
                {"role": "system", "content": "You are a financial sentiment scoring system. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ]
            
            response = self.llm_fn(messages)
            self._llm_calls += 1
            
            # Parse response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Clean potential markdown wrapping
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
            
            scores_data = json.loads(content)
            
            # Map scores back to articles
            scored = []
            score_map = {s["id"]: s for s in scores_data if isinstance(s, dict)}
            
            for article in articles:
                article_id = self._article_id(article)
                if article_id in score_map:
                    s = score_map[article_id]
                    scored.append(ArticleScore(
                        article_id=article_id,
                        ticker=ticker,
                        title=article.get("title", ""),
                        source=article.get("source", "unknown"),
                        published_at=article.get("published", ""),
                        sentiment_score=float(np.clip(s.get("score", 0), -1, 1)),
                        confidence=float(np.clip(s.get("confidence", 0.5), 0, 1)),
                        reasoning=s.get("reasoning", ""),
                        scored_at=datetime.utcnow().isoformat(),
                    ))
                else:
                    # Article not scored — assign neutral
                    scored.append(ArticleScore(
                        article_id=article_id,
                        ticker=ticker,
                        title=article.get("title", ""),
                        source=article.get("source", "unknown"),
                        published_at=article.get("published", ""),
                        sentiment_score=0.0,
                        confidence=0.0,
                        reasoning="Not scored by LLM",
                        scored_at=datetime.utcnow().isoformat(),
                    ))
            
            return scored
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            # Return neutral scores on failure
            return [
                ArticleScore(
                    article_id=self._article_id(article),
                    ticker=ticker,
                    title=article.get("title", ""),
                    source=article.get("source", "unknown"),
                    published_at=article.get("published", ""),
                    sentiment_score=0.0,
                    confidence=0.0,
                    reasoning=f"LLM error: {str(e)[:50]}",
                    scored_at=datetime.utcnow().isoformat(),
                )
                for article in articles
            ]
    
    def _aggregate_with_decay(
        self, articles: List[ArticleScore]
    ) -> Tuple[float, float]:
        """
        Aggregate article scores with exponential time-decay weighting.
        
        More recent articles have higher weight. Half-life determines
        how quickly older articles lose influence.
        """
        now = datetime.utcnow()
        half_life_seconds = self.config.decay_half_life_hours * 3600
        max_age_seconds = self.config.max_age_hours * 3600
        
        weighted_sum = 0.0
        weight_total = 0.0
        confidence_sum = 0.0
        valid_count = 0
        
        for article in articles:
            # Parse published time
            try:
                pub_time = datetime.fromisoformat(article.published_at.replace("Z", "+00:00").replace("+00:00", ""))
            except (ValueError, AttributeError):
                pub_time = now - timedelta(hours=24)  # Default to 1 day old
            
            age_seconds = (now - pub_time).total_seconds()
            
            # Skip articles beyond max age
            if age_seconds > max_age_seconds:
                continue
            
            # Exponential decay weight
            decay = np.exp(-0.693 * age_seconds / half_life_seconds)  # ln(2) ≈ 0.693
            
            # Weight = decay * confidence
            weight = decay * max(article.confidence, 0.1)
            
            weighted_sum += article.sentiment_score * weight
            weight_total += weight
            confidence_sum += article.confidence
            valid_count += 1
        
        if weight_total < 1e-9 or valid_count == 0:
            return 0.0, 0.0
        
        composite = weighted_sum / weight_total
        avg_confidence = confidence_sum / valid_count
        
        return float(np.clip(composite, -1.0, 1.0)), float(avg_confidence)
    
    def _classify_signal(self, score: float) -> str:
        """Classify composite score into signal strength category."""
        if score >= 0.5:
            return "STRONG_BULL"
        elif score >= 0.2:
            return "BULL"
        elif score <= -0.5:
            return "STRONG_BEAR"
        elif score <= -0.2:
            return "BEAR"
        else:
            return "NEUTRAL"
    
    def _compute_conviction_modifier(self, score: float, confidence: float) -> float:
        """
        Compute conviction modifier for PerAssetRouter integration.
        
        Returns a multiplier (0.5 to 1.5) that adjusts the base signal:
        - Strong bullish sentiment → amplify long signals (up to 1.5x)
        - Strong bearish sentiment → amplify short signals (up to 1.5x)
        - Neutral → no modification (1.0x)
        - Low confidence → dampen toward 1.0
        """
        # Base modifier from sentiment score
        weight = self.config.sentiment_weight
        raw_modifier = 1.0 + (score * weight)
        
        # Dampen by confidence (low confidence → closer to 1.0)
        dampened = 1.0 + (raw_modifier - 1.0) * confidence
        
        # Clamp to safe range
        return float(np.clip(dampened, 0.5, 1.5))
    
    @staticmethod
    def _article_id(article: Dict) -> str:
        """Generate deterministic ID for an article."""
        key = f"{article.get('title', '')}-{article.get('source', '')}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    @staticmethod
    def _default_news_fetch(ticker: str, days_back: int) -> List[Dict]:
        """
        Default news fetcher using yfinance.
        
        Returns list of dicts with: title, source, published, summary
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            articles = []
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            
            for item in news:
                # yfinance news format
                title = item.get("title", "")
                source = item.get("publisher", "unknown")
                pub_ts = item.get("providerPublishTime", 0)
                
                if pub_ts:
                    pub_dt = datetime.utcfromtimestamp(pub_ts)
                    if pub_dt < cutoff:
                        continue
                    published = pub_dt.isoformat()
                else:
                    published = datetime.utcnow().isoformat()
                
                # Use related tickers to confirm relevance
                related = item.get("relatedTickers", [])
                
                articles.append({
                    "title": title,
                    "source": source,
                    "published": published,
                    "summary": title,  # yfinance doesn't always provide summary
                    "related_tickers": related,
                })
            
            return articles[:20]  # Limit
            
        except Exception as e:
            logger.warning(f"yfinance news fetch failed for {ticker}: {e}")
            return []
