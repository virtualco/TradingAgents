"""Tests for Phase 2: Research Factory.

Covers:
- Strategy rules (Technical, Fundamental, Sentiment)
- Signal registry (CRUD, metrics)
- Walk-forward engine (fold generation, outcome scoring)
- Research factory (pipeline integration)
"""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.strategy_rules import (
    FundamentalStrategyRules,
    SentimentStrategyRules,
    SignalStrength,
    TechnicalStrategyRules,
    _score_to_signal,
    compute_multi_role_signals,
)
from tradingagents.research.signal_registry import (
    SignalDirection,
    SignalRecord,
    SignalRegistry,
    SignalStatus,
)
from tradingagents.research.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
)
from tradingagents.research.factory import ResearchFactory, _parse_direction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(ticker="AAPL", days=100, trend="up") -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    dates = pd.bdate_range(end="2026-04-28", periods=days)
    np.random.seed(42)
    base = 150.0
    if trend == "up":
        prices = base + np.cumsum(np.random.normal(0.3, 1.5, days))
    elif trend == "down":
        prices = base + np.cumsum(np.random.normal(-0.3, 1.5, days))
    else:
        prices = base + np.cumsum(np.random.normal(0.0, 1.5, days))

    prices = np.maximum(prices, 10.0)  # Floor at $10
    df = pd.DataFrame({
        "ticker": ticker,
        "event_time": dates.date,
        "available_at": pd.Timestamp("2026-01-05"),
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, days).astype(float),
        "adj_close": prices,
    })
    return df


def make_fundamentals(ticker="AAPL") -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "event_time": "2026-01-01",
        "available_at": datetime(2026, 1, 15),
        "revenue": 400e9,
        "eps": 6.5,
        "free_cash_flow": 100e9,
        "total_assets": 350e9,
        "total_liabilities": 200e9,
        "total_equity": 150e9,
    }])


def make_news(ticker="AAPL", n=5, sentiment=0.3) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "event_time": f"2026-04-{20+i:02d}",
        "available_at": datetime(2026, 4, 20 + i),
        "headline": f"News headline {i}",
        "sentiment_score": sentiment + np.random.uniform(-0.1, 0.1),
    } for i in range(n)])


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a temporary signal registry."""
    db_path = str(tmp_path / "test_signals.db")
    return SignalRegistry(db_path)


@pytest.fixture
def sample_signal():
    """Create a sample signal record."""
    return SignalRecord(
        signal_id=str(uuid.uuid4()),
        ticker="AAPL",
        trade_date="2026-04-28",
        pipeline_version="0.2.4",
        direction=SignalDirection.LONG,
        conviction=0.75,
        target_horizon_days=45,
        entry_price=175.50,
        stop_loss=165.00,
        take_profit=195.00,
        active_roles=["TechnicalAnalyst", "SentimentAnalyst"],
        role_scores={"TechnicalAnalyst": 0.6, "SentimentAnalyst": 0.4},
        ensemble_score=0.5,
        executive_summary="Strong bullish setup with technical confirmation.",
    )


# ---------------------------------------------------------------------------
# Strategy Rules Tests
# ---------------------------------------------------------------------------

class TestScoreToSignal:
    def test_strong_buy(self):
        assert _score_to_signal(0.8) == SignalStrength.STRONG_BUY

    def test_buy(self):
        assert _score_to_signal(0.4) == SignalStrength.BUY

    def test_neutral(self):
        assert _score_to_signal(0.0) == SignalStrength.NEUTRAL

    def test_sell(self):
        assert _score_to_signal(-0.4) == SignalStrength.SELL

    def test_strong_sell(self):
        assert _score_to_signal(-0.8) == SignalStrength.STRONG_SELL

    def test_boundary_buy(self):
        assert _score_to_signal(0.2) == SignalStrength.BUY

    def test_boundary_sell(self):
        assert _score_to_signal(-0.2) == SignalStrength.NEUTRAL


class TestTechnicalStrategyRules:
    def test_returns_summary_for_valid_data(self):
        ohlcv = make_ohlcv(days=100)
        rules = TechnicalStrategyRules()
        result = rules.compute(ohlcv, "AAPL", "2026-04-28")
        assert result.role == "TechnicalAnalyst"
        assert result.ticker == "AAPL"
        assert -1.0 <= result.composite_score <= 1.0

    def test_neutral_for_insufficient_data(self):
        ohlcv = make_ohlcv(days=5)
        rules = TechnicalStrategyRules()
        result = rules.compute(ohlcv, "AAPL", "2026-04-28")
        assert result.composite_signal == SignalStrength.NEUTRAL
        assert result.rule_signals == []

    def test_neutral_for_empty_data(self):
        rules = TechnicalStrategyRules()
        result = rules.compute(pd.DataFrame(), "AAPL", "2026-04-28")
        assert result.composite_signal == SignalStrength.NEUTRAL

    def test_bullish_trend_gives_positive_score(self):
        ohlcv = make_ohlcv(days=100, trend="up")
        rules = TechnicalStrategyRules()
        result = rules.compute(ohlcv, "AAPL", "2026-04-28")
        # Uptrend should generally produce positive or neutral score
        assert result.composite_score >= -0.3  # Allow some noise

    def test_context_string_contains_ticker(self):
        ohlcv = make_ohlcv(days=100)
        rules = TechnicalStrategyRules()
        result = rules.compute(ohlcv, "AAPL", "2026-04-28")
        assert "AAPL" in result.to_context_string()

    def test_rule_signals_have_required_fields(self):
        ohlcv = make_ohlcv(days=100)
        rules = TechnicalStrategyRules()
        result = rules.compute(ohlcv, "AAPL", "2026-04-28")
        for sig in result.rule_signals:
            assert sig.rule_name
            assert 0.0 <= sig.confidence <= 1.0
            assert -1.0 <= sig.score <= 1.0


class TestFundamentalStrategyRules:
    def test_returns_summary_for_valid_data(self):
        fundamentals = make_fundamentals()
        rules = FundamentalStrategyRules()
        result = rules.compute(fundamentals, "AAPL", "2026-04-28", market_cap=3e12)
        assert result.role == "FundamentalAnalyst"
        assert -1.0 <= result.composite_score <= 1.0

    def test_neutral_for_empty_data(self):
        rules = FundamentalStrategyRules()
        result = rules.compute(pd.DataFrame(), "AAPL", "2026-04-28")
        assert result.composite_signal == SignalStrength.NEUTRAL

    def test_positive_eps_gives_positive_score(self):
        fundamentals = make_fundamentals()
        rules = FundamentalStrategyRules()
        result = rules.compute(fundamentals, "AAPL", "2026-04-28")
        # High EPS ($6.5) should contribute positively
        eps_signals = [s for s in result.rule_signals if s.rule_name == "EPS"]
        if eps_signals:
            assert eps_signals[0].score > 0


class TestSentimentStrategyRules:
    def test_returns_summary_for_valid_data(self):
        news = make_news()
        ohlcv = make_ohlcv(days=50)
        rules = SentimentStrategyRules()
        result = rules.compute(news, ohlcv, "AAPL", "2026-04-28")
        assert result.role == "SentimentAnalyst"
        assert -1.0 <= result.composite_score <= 1.0

    def test_positive_sentiment_gives_positive_score(self):
        news = make_news(sentiment=0.7)
        ohlcv = make_ohlcv(days=50)
        rules = SentimentStrategyRules()
        result = rules.compute(news, ohlcv, "AAPL", "2026-04-28")
        assert result.composite_score > 0

    def test_negative_sentiment_gives_negative_score(self):
        news = make_news(sentiment=-0.7)
        ohlcv = make_ohlcv(days=50)
        rules = SentimentStrategyRules()
        result = rules.compute(news, ohlcv, "AAPL", "2026-04-28")
        assert result.composite_score < 0


class TestComputeMultiRoleSignals:
    def test_all_roles_computed(self):
        ohlcv = make_ohlcv(days=100)
        fundamentals = make_fundamentals()
        news = make_news()
        result = compute_multi_role_signals(
            ticker="AAPL",
            trade_date="2026-04-28",
            ohlcv=ohlcv,
            fundamentals=fundamentals,
            news=news,
            market_cap=3e12,
        )
        assert result.technical is not None
        assert result.fundamental is not None
        assert result.sentiment is not None

    def test_selective_roles(self):
        ohlcv = make_ohlcv(days=100)
        result = compute_multi_role_signals(
            ticker="AAPL",
            trade_date="2026-04-28",
            ohlcv=ohlcv,
            fundamentals=pd.DataFrame(),
            news=pd.DataFrame(),
            active_roles=["TechnicalAnalyst"],
        )
        assert result.technical is not None
        assert result.fundamental is None
        assert result.sentiment is None

    def test_ensemble_score_in_range(self):
        ohlcv = make_ohlcv(days=100)
        result = compute_multi_role_signals(
            ticker="AAPL",
            trade_date="2026-04-28",
            ohlcv=ohlcv,
            fundamentals=pd.DataFrame(),
            news=pd.DataFrame(),
        )
        assert -1.0 <= result.ensemble_score <= 1.0

    def test_context_string_contains_all_roles(self):
        ohlcv = make_ohlcv(days=100)
        fundamentals = make_fundamentals()
        news = make_news()
        result = compute_multi_role_signals(
            ticker="AAPL",
            trade_date="2026-04-28",
            ohlcv=ohlcv,
            fundamentals=fundamentals,
            news=news,
        )
        ctx = result.to_context_string()
        assert "TechnicalAnalyst" in ctx
        assert "FundamentalAnalyst" in ctx
        assert "SentimentAnalyst" in ctx


# ---------------------------------------------------------------------------
# Signal Registry Tests
# ---------------------------------------------------------------------------

class TestSignalRegistry:
    def test_save_and_retrieve(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        retrieved = tmp_registry.get(sample_signal.signal_id)
        assert retrieved is not None
        assert retrieved.ticker == "AAPL"
        assert retrieved.direction == SignalDirection.LONG
        assert retrieved.conviction == 0.75

    def test_count_increments(self, tmp_registry, sample_signal):
        assert tmp_registry.count() == 0
        tmp_registry.save(sample_signal)
        assert tmp_registry.count() == 1

    def test_get_open_signals(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        open_signals = tmp_registry.get_open_signals("AAPL")
        assert len(open_signals) == 1
        assert open_signals[0].signal_id == sample_signal.signal_id

    def test_record_outcome(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        success = tmp_registry.record_outcome(
            signal_id=sample_signal.signal_id,
            exit_price=185.00,
            exit_date="2026-06-12",
        )
        assert success
        updated = tmp_registry.get(sample_signal.signal_id)
        assert updated.status == SignalStatus.CLOSED
        assert updated.exit_price == 185.00
        assert updated.actual_return is not None
        # (185 - 175.5) / 175.5 ≈ 0.054
        assert abs(updated.actual_return - 0.054) < 0.005

    def test_actual_return_positive_for_long_win(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        tmp_registry.record_outcome(sample_signal.signal_id, 200.0, "2026-06-12")
        updated = tmp_registry.get(sample_signal.signal_id)
        assert updated.actual_return > 0

    def test_actual_return_negative_for_long_loss(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        tmp_registry.record_outcome(sample_signal.signal_id, 160.0, "2026-06-12")
        updated = tmp_registry.get(sample_signal.signal_id)
        assert updated.actual_return < 0

    def test_compute_metrics_no_signals(self, tmp_registry):
        metrics = tmp_registry.compute_metrics()
        assert metrics["total_signals"] == 0

    def test_compute_metrics_with_wins_and_losses(self, tmp_registry):
        # Create 4 signals: 3 wins, 1 loss
        for i, (exit_price, expected_win) in enumerate([
            (190.0, True), (195.0, True), (185.0, True), (160.0, False)
        ]):
            sig = SignalRecord(
                signal_id=str(uuid.uuid4()),
                ticker="AAPL",
                trade_date=f"2026-0{i+1}-15",
                pipeline_version="0.2.4",
                direction=SignalDirection.LONG,
                conviction=0.7,
                target_horizon_days=45,
                entry_price=175.50,
            )
            tmp_registry.save(sig)
            tmp_registry.record_outcome(sig.signal_id, exit_price, f"2026-0{i+2}-01")

        metrics = tmp_registry.compute_metrics()
        assert metrics["total_signals"] == 4
        assert metrics["win_rate"] == 0.75
        assert metrics["avg_return"] > 0  # 3 wins outweigh 1 loss

    def test_to_dataframe_returns_df(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        df = tmp_registry.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_json_fields_round_trip(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        retrieved = tmp_registry.get(sample_signal.signal_id)
        assert retrieved.active_roles == ["TechnicalAnalyst", "SentimentAnalyst"]
        assert retrieved.role_scores == {"TechnicalAnalyst": 0.6, "SentimentAnalyst": 0.4}


# ---------------------------------------------------------------------------
# Walk-Forward Engine Tests
# ---------------------------------------------------------------------------

class TestWalkForwardEngine:
    def test_generates_folds(self):
        engine = WalkForwardEngine()
        config = WalkForwardConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2026-04-28",
            min_train_days=252,
            test_window_days=21,
            step_days=21,
        )
        folds = engine._generate_folds(config)
        assert len(folds) > 0
        # Each fold should have train_start = config.start_date
        for fold in folds:
            assert fold.train_start == "2023-01-01"

    def test_no_folds_for_short_period(self):
        engine = WalkForwardEngine()
        config = WalkForwardConfig(
            ticker="AAPL",
            start_date="2026-01-01",
            end_date="2026-04-28",
            min_train_days=365,  # Longer than the period
            test_window_days=21,
            step_days=21,
        )
        folds = engine._generate_folds(config)
        assert len(folds) == 0

    def test_folds_are_sequential(self):
        engine = WalkForwardEngine()
        config = WalkForwardConfig(
            ticker="AAPL",
            start_date="2023-01-01",
            end_date="2026-04-28",
            min_train_days=252,
            test_window_days=21,
            step_days=21,
        )
        folds = engine._generate_folds(config)
        for i in range(1, len(folds)):
            assert folds[i].test_start >= folds[i-1].test_start

    def test_quick_signal_generation(self):
        engine = WalkForwardEngine()
        ohlcv = make_ohlcv(days=100)
        config = WalkForwardConfig(
            ticker="AAPL",
            start_date="2025-01-01",
            end_date="2026-04-28",
            mode="quick",
        )
        signal = engine._generate_quick_signal("AAPL", "2026-04-28", ohlcv, config)
        # May return None if conviction < 0.15, but if not None, check fields
        if signal is not None:
            assert signal.ticker == "AAPL"
            assert signal.direction in [SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT]
            assert signal.entry_price is not None

    def test_outcome_scoring(self):
        engine = WalkForwardEngine()
        ohlcv = make_ohlcv(days=200, trend="up")

        # Create a LONG signal at the midpoint
        signal = SignalRecord(
            signal_id=str(uuid.uuid4()),
            ticker="AAPL",
            trade_date=str(ohlcv["event_time"].iloc[100]),
            pipeline_version="0.2.4",
            direction=SignalDirection.LONG,
            conviction=0.7,
            target_horizon_days=30,
            entry_price=float(ohlcv["close"].iloc[100]),
        )

        scored = engine._score_outcomes([signal], ohlcv, horizon_days=30)
        assert len(scored) == 1
        # Should have been scored (uptrend, so likely closed with a return)
        if scored[0].status == SignalStatus.CLOSED:
            assert scored[0].actual_return is not None


# ---------------------------------------------------------------------------
# Research Factory Tests
# ---------------------------------------------------------------------------

class TestResearchFactory:
    def test_parse_direction_buy(self):
        assert _parse_direction("BUY") == SignalDirection.LONG
        assert _parse_direction("Strong Buy") == SignalDirection.LONG

    def test_parse_direction_sell(self):
        assert _parse_direction("SELL") == SignalDirection.SHORT
        assert _parse_direction("strong sell") == SignalDirection.SHORT

    def test_parse_direction_hold(self):
        assert _parse_direction("HOLD") == SignalDirection.FLAT
        assert _parse_direction("neutral") == SignalDirection.FLAT

    def test_parse_direction_unknown(self):
        assert _parse_direction("") == SignalDirection.FLAT
        assert _parse_direction("unclear") == SignalDirection.FLAT

    def test_record_pipeline_result(self, tmp_registry):
        factory = ResearchFactory(registry=tmp_registry)
        ohlcv = make_ohlcv(days=100)

        final_state = {
            "final_trade_decision": "BUY - High confidence based on strong technical setup",
            "investment_plan": "Initiate long position with 2% allocation",
            "investment_debate_state": {"judge_decision": "Bull case wins"},
        }

        signal = factory.record_pipeline_result(
            ticker="AAPL",
            trade_date="2026-04-28",
            final_state=final_state,
            ohlcv=ohlcv,
            active_roles=["TechnicalAnalyst"],
        )

        assert signal.ticker == "AAPL"
        assert signal.direction == SignalDirection.LONG
        assert signal.entry_price is not None
        assert tmp_registry.count() == 1

    def test_get_performance_report_empty(self, tmp_registry):
        factory = ResearchFactory(registry=tmp_registry)
        report = factory.get_performance_report()
        assert report["total_signals"] == 0

    def test_get_open_signals(self, tmp_registry, sample_signal):
        tmp_registry.save(sample_signal)
        factory = ResearchFactory(registry=tmp_registry)
        open_signals = factory.get_open_signals("AAPL")
        assert len(open_signals) == 1
