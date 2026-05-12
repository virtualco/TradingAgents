"""Tests for tradingagents.agents.thesis_validator — enforcement at agent output boundaries."""

import pytest

from tradingagents.agents.thesis_validator import (
    ThesisValidationError,
    enforce_agent_thesis,
    enforce_portfolio_decision,
    enforce_research_plan,
    enforce_trader_proposal,
    validate_agent_thesis,
    validate_portfolio_decision,
    validate_research_plan,
    validate_trader_proposal,
)
from tradingagents.agents.thesis_schema import (
    AgentThesis,
    EvidenceItem,
    EvidenceType,
    SignalDirection,
    SignalObject,
)


# ---------------------------------------------------------------------------
# Research Plan validation
# ---------------------------------------------------------------------------


class TestValidateResearchPlan:
    def test_valid_plan(self):
        output = (
            "**Recommendation**: Buy\n\n"
            "**Rationale**: The bull case is stronger because of strong earnings growth.\n\n"
            "**Strategic Actions**: Enter a 5% position at current levels."
        )
        violations = validate_research_plan(output)
        assert violations == []

    def test_empty_output(self):
        violations = validate_research_plan("")
        assert len(violations) == 1
        assert "empty" in violations[0].lower()

    def test_missing_rating(self):
        output = "This is a plan without any clear recommendation keyword."
        violations = validate_research_plan(output)
        assert len(violations) >= 1
        assert "recommendation" in violations[0].lower() or "rating" in violations[0].lower()

    def test_too_short(self):
        output = "Buy now."
        violations = validate_research_plan(output)
        assert any("short" in v.lower() for v in violations)

    def test_strict_mode_missing_rationale(self):
        output = "**Recommendation**: Hold\n\nSome text without the required sections."
        violations = validate_research_plan(output, strict=True)
        # Should flag missing rationale or strategic actions
        assert len(violations) >= 1

    def test_enforce_raises(self):
        with pytest.raises(ThesisValidationError) as exc_info:
            enforce_research_plan("")
        assert exc_info.value.agent_name == "Research Manager"
        assert len(exc_info.value.violations) >= 1

    def test_enforce_passes(self):
        output = (
            "**Recommendation**: Overweight\n\n"
            "**Rationale**: Strong fundamentals and improving sentiment.\n\n"
            "**Strategic Actions**: Gradually increase position over 2 weeks."
        )
        result = enforce_research_plan(output)
        assert result == output


# ---------------------------------------------------------------------------
# Trader Proposal validation
# ---------------------------------------------------------------------------


class TestValidateTraderProposal:
    def test_valid_proposal(self):
        output = (
            "**Action**: Buy\n\n"
            "**Reasoning**: Strong momentum and positive earnings.\n\n"
            "FINAL TRANSACTION PROPOSAL: **BUY**"
        )
        violations = validate_trader_proposal(output)
        assert violations == []

    def test_empty_output(self):
        violations = validate_trader_proposal("")
        assert len(violations) == 1
        assert "empty" in violations[0].lower()

    def test_missing_action(self):
        output = "This proposal does not contain any action keyword at all."
        violations = validate_trader_proposal(output)
        assert any("action" in v.lower() for v in violations)

    def test_missing_final_line(self):
        output = "**Action**: Buy\n\n**Reasoning**: Good setup."
        violations = validate_trader_proposal(output)
        assert any("FINAL TRANSACTION PROPOSAL" in v for v in violations)

    def test_enforce_raises(self):
        with pytest.raises(ThesisValidationError) as exc_info:
            enforce_trader_proposal("")
        assert exc_info.value.agent_name == "Trader"


# ---------------------------------------------------------------------------
# Portfolio Decision validation
# ---------------------------------------------------------------------------


class TestValidatePortfolioDecision:
    def test_valid_decision(self):
        output = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Enter a full position at current levels.\n\n"
            "**Investment Thesis**: Strong earnings growth supports the bull case."
        )
        violations = validate_portfolio_decision(output)
        assert violations == []

    def test_empty_output(self):
        violations = validate_portfolio_decision("")
        assert len(violations) == 1

    def test_missing_rating(self):
        output = "This decision has no clear rating keyword anywhere in it."
        violations = validate_portfolio_decision(output)
        assert any("rating" in v.lower() for v in violations)

    def test_enforce_raises(self):
        with pytest.raises(ThesisValidationError) as exc_info:
            enforce_portfolio_decision("")
        assert exc_info.value.agent_name == "Portfolio Manager"


# ---------------------------------------------------------------------------
# AgentThesis validation
# ---------------------------------------------------------------------------


class TestValidateAgentThesis:
    @pytest.fixture
    def valid_thesis(self):
        return AgentThesis(
            ticker="AAPL",
            trade_date="2026-04-30",
            rating="Buy",
            confidence=0.85,
            executive_summary="Enter a full position in AAPL at current levels with a 5% allocation.",
            investment_thesis="Strong earnings growth, expanding margins, and positive analyst revisions support the bull case for AAPL.",
            evidence=[
                EvidenceItem(
                    source_type=EvidenceType.PRICE_DATA,
                    source_name="Yahoo Finance OHLCV",
                    available_at="2026-04-29",
                    claim="AAPL up 3.2% on above-average volume.",
                    supporting=True,
                    confidence=0.9,
                )
            ],
            signal=SignalObject(
                ticker="AAPL",
                direction=SignalDirection.LONG,
                conviction=0.85,
                entry_price=220.0,
                stop_loss=210.0,
                take_profit=240.0,
            ),
        )

    def test_valid_thesis(self, valid_thesis):
        violations = validate_agent_thesis(valid_thesis)
        assert violations == []

    def test_empty_ticker(self, valid_thesis):
        valid_thesis.ticker = ""
        violations = validate_agent_thesis(valid_thesis)
        assert any("ticker" in v.lower() for v in violations)

    def test_invalid_rating(self, valid_thesis):
        valid_thesis.rating = "StrongBuy"
        violations = validate_agent_thesis(valid_thesis)
        assert any("rating" in v.lower() for v in violations)

    def test_time_safety_violation(self, valid_thesis):
        # Evidence available AFTER trade_date — lookahead bias
        valid_thesis.evidence[0].available_at = "2026-05-01"
        violations = validate_agent_thesis(valid_thesis, strict=True)
        assert any("lookahead" in v.lower() for v in violations)

    def test_no_evidence_strict(self, valid_thesis):
        valid_thesis.evidence = []
        violations = validate_agent_thesis(valid_thesis, strict=True)
        assert any("evidence" in v.lower() for v in violations)

    def test_signal_stop_loss_invalid_long(self, valid_thesis):
        # LONG signal with stop_loss above entry
        valid_thesis.signal.stop_loss = 225.0
        violations = validate_agent_thesis(valid_thesis)
        assert any("stop_loss" in v.lower() for v in violations)

    def test_signal_stop_loss_invalid_short(self, valid_thesis):
        valid_thesis.signal.direction = SignalDirection.SHORT
        valid_thesis.signal.stop_loss = 200.0  # Below entry for SHORT = invalid
        violations = validate_agent_thesis(valid_thesis)
        assert any("stop_loss" in v.lower() for v in violations)

    def test_enforce_raises(self, valid_thesis):
        valid_thesis.ticker = ""
        with pytest.raises(ThesisValidationError) as exc_info:
            enforce_agent_thesis(valid_thesis)
        assert exc_info.value.agent_name == "AgentThesis"

    def test_enforce_passes(self, valid_thesis):
        result = enforce_agent_thesis(valid_thesis)
        assert result == valid_thesis
