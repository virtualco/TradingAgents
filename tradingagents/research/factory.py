"""Research Factory — integration layer for the TradingAgents pipeline.

The ResearchFactory sits between the TradingGraph and the signal registry.
After each pipeline run, it:
1. Extracts the structured decision from the final state
2. Enriches it with quantitative pre-analysis signals (strategy rules)
3. Persists the signal to the registry
4. Returns the enriched signal for downstream use

It also provides the entry point for walk-forward validation runs.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .signal_registry import SignalDirection, SignalRecord, SignalRegistry, SignalStatus
from .strategy_rules import compute_multi_role_signals
from .walk_forward import WalkForwardConfig, WalkForwardEngine, WalkForwardResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Direction mapping from TradingAgents prose signals
# ---------------------------------------------------------------------------

_DIRECTION_MAP = {
    "buy": SignalDirection.LONG,
    "long": SignalDirection.LONG,
    "strong buy": SignalDirection.LONG,
    "overweight": SignalDirection.LONG,
    "sell": SignalDirection.SHORT,
    "short": SignalDirection.SHORT,
    "strong sell": SignalDirection.SHORT,
    "underweight": SignalDirection.SHORT,
    "hold": SignalDirection.FLAT,
    "neutral": SignalDirection.FLAT,
    "flat": SignalDirection.FLAT,
}


def _parse_direction(signal_text: str) -> SignalDirection:
    """Parse a prose signal string into a SignalDirection."""
    if not signal_text:
        return SignalDirection.FLAT
    lower = signal_text.lower().strip()
    for key, direction in _DIRECTION_MAP.items():
        if key in lower:
            return direction
    return SignalDirection.FLAT


def _extract_conviction(final_state: Dict) -> float:
    """Extract conviction score from the final state."""
    # Try structured output first (v0.2.4+)
    decision = final_state.get("final_trade_decision", "")
    if hasattr(decision, "conviction"):
        return float(decision.conviction)

    # Fallback: look for confidence keywords in the text
    text = str(decision).lower()
    if "high confidence" in text or "strong" in text:
        return 0.80
    elif "moderate" in text or "medium" in text:
        return 0.55
    elif "low confidence" in text or "weak" in text:
        return 0.30
    return 0.50


def _extract_debate_winner(final_state: Dict) -> Optional[str]:
    """Extract which side won the investment debate."""
    debate = final_state.get("investment_debate_state", {})
    judge = debate.get("judge_decision", "")
    if not judge:
        return None
    lower = str(judge).lower()
    if "bull" in lower:
        return "bull"
    elif "bear" in lower:
        return "bear"
    return None


# ---------------------------------------------------------------------------
# Research Factory
# ---------------------------------------------------------------------------

class ResearchFactory:
    """Integrates the TradingAgents pipeline with the signal registry.

    Usage:
        factory = ResearchFactory(
            registry=SignalRegistry("data/signals.db"),
            pipeline_version="0.2.4",
        )

        # After a TradingGraph run:
        signal = factory.record_pipeline_result(
            ticker="AAPL",
            trade_date="2026-04-28",
            final_state=final_state,
            ohlcv=ohlcv_df,
        )

        # Run walk-forward validation:
        result = factory.run_walk_forward(
            ticker="AAPL",
            start_date="2024-01-01",
            end_date="2026-04-28",
        )
    """

    def __init__(
        self,
        registry: Optional[SignalRegistry] = None,
        pipeline_version: str = "0.2.4",
        data_lake=None,
    ):
        self.registry = registry or SignalRegistry()
        self.pipeline_version = pipeline_version
        self.data_lake = data_lake

    def record_pipeline_result(
        self,
        ticker: str,
        trade_date: str,
        final_state: Dict[str, Any],
        ohlcv: Optional[pd.DataFrame] = None,
        fundamentals: Optional[pd.DataFrame] = None,
        news: Optional[pd.DataFrame] = None,
        agent_config_id: Optional[str] = None,
        active_roles: Optional[List[str]] = None,
        market_cap: Optional[float] = None,
    ) -> SignalRecord:
        """Extract, enrich, and persist a signal from a pipeline run.

        Args:
            ticker: The stock ticker.
            trade_date: The date of the run (YYYY-MM-DD).
            final_state: The final LangGraph state dict from TradingGraph.
            ohlcv: OHLCV DataFrame for quantitative enrichment (optional).
            fundamentals: Fundamentals DataFrame (optional).
            news: News DataFrame (optional).
            agent_config_id: ID of the agent configuration used.
            active_roles: List of analyst roles that were active.
            market_cap: Market cap for FCF yield calculation.

        Returns:
            The persisted SignalRecord.
        """
        # 1. Extract core decision
        decision_text = str(final_state.get("final_trade_decision", ""))
        direction = _parse_direction(decision_text)
        conviction = _extract_conviction(final_state)
        debate_winner = _extract_debate_winner(final_state)

        # 2. Compute quantitative signals if price data is available
        role_scores = {}
        ensemble_score = conviction if direction == SignalDirection.LONG else -conviction

        if ohlcv is not None and not ohlcv.empty:
            multi_signals = compute_multi_role_signals(
                ticker=ticker,
                trade_date=trade_date,
                ohlcv=ohlcv,
                fundamentals=fundamentals or pd.DataFrame(),
                news=news or pd.DataFrame(),
                market_cap=market_cap,
                active_roles=active_roles,
            )
            if multi_signals.technical:
                role_scores["TechnicalAnalyst"] = multi_signals.technical.composite_score
            if multi_signals.fundamental:
                role_scores["FundamentalAnalyst"] = multi_signals.fundamental.composite_score
            if multi_signals.sentiment:
                role_scores["SentimentAnalyst"] = multi_signals.sentiment.composite_score
            ensemble_score = multi_signals.ensemble_score

        # 3. Extract entry price from OHLCV
        entry_price = None
        if ohlcv is not None and not ohlcv.empty and "close" in ohlcv.columns:
            entry_price = float(ohlcv["close"].iloc[-1])

        # 4. Build the signal record
        signal = SignalRecord(
            signal_id=str(uuid.uuid4()),
            ticker=ticker.upper(),
            trade_date=trade_date,
            pipeline_version=self.pipeline_version,
            direction=direction,
            conviction=round(conviction, 3),
            target_horizon_days=45,
            entry_price=round(entry_price, 2) if entry_price else None,
            agent_config_id=agent_config_id,
            active_roles=active_roles or [],
            role_scores=role_scores,
            ensemble_score=round(ensemble_score, 3),
            debate_winner=debate_winner,
            executive_summary=str(final_state.get("investment_plan", ""))[:500],
            investment_thesis=decision_text[:1000],
        )

        # 5. Persist to registry
        self.registry.save(signal)
        logger.info(
            f"Recorded signal {signal.signal_id}: {ticker} {direction.value} "
            f"(conviction={conviction:.2f}) on {trade_date}"
        )

        return signal

    def run_walk_forward(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        mode: str = "quick",
        active_roles: Optional[List[str]] = None,
        test_window_days: int = 21,
        min_train_days: int = 252,
        signal_horizon_days: int = 45,
    ) -> WalkForwardResult:
        """Run walk-forward validation for a ticker.

        Args:
            ticker: Stock ticker to validate.
            start_date: Start of the full evaluation period.
            end_date: End of the full evaluation period.
            mode: "quick" (rule-based) or "deep" (LLM-powered).
            active_roles: Analyst roles to include.
            test_window_days: Size of each test window.
            min_train_days: Minimum training period.
            signal_horizon_days: How long to hold each signal.

        Returns:
            WalkForwardResult with fold metrics and aggregate performance.
        """
        config = WalkForwardConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            test_window_days=test_window_days,
            min_train_days=min_train_days,
            signal_horizon_days=signal_horizon_days,
            active_roles=active_roles,
            mode=mode,
            pipeline_version=self.pipeline_version,
        )

        engine = WalkForwardEngine(
            data_lake=self.data_lake,
            signal_registry=self.registry,
        )

        return engine.run(config)

    def get_performance_report(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict:
        """Get performance metrics from the signal registry."""
        return self.registry.compute_metrics(ticker, start_date, end_date)

    def get_open_signals(self, ticker: Optional[str] = None) -> List[SignalRecord]:
        """Get all open signals."""
        return self.registry.get_open_signals(ticker)

    def get_signals_dataframe(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Get all signals as a DataFrame for analysis."""
        return self.registry.to_dataframe(ticker)
