"""Tests for the extended thesis schema, time-safety validation, and failure handling.

Covers:
- Schema validation (required fields, enums, ranges)
- Time-safety checks (lookahead bias detection)
- Evidence coverage scoring
- JSON round-trip serialization
- Markdown rendering
- Malformed input rejection
- Edge cases (empty evidence, missing optional fields)
"""
import json
import pytest
from pydantic import ValidationError

from tradingagents.agents.thesis_schema import (
    AgentThesis,
    EvidenceItem,
    EvidenceType,
    RiskItem,
    SignalDirection,
    SignalObject,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_thesis() -> AgentThesis:
    """A thesis with only required fields."""
    return AgentThesis(
        ticker="AAPL",
        trade_date="2026-04-28",
        rating="Buy",
        confidence=0.82,
        executive_summary="Strong earnings beat with raised guidance. Enter near $195.",
        investment_thesis="Apple reported Q2 revenue of $95.4B, beating estimates by 4%.",
    )


@pytest.fixture
def full_thesis() -> AgentThesis:
    """A thesis with all fields populated."""
    return AgentThesis(
        ticker="NVDA",
        trade_date="2026-04-28",
        pipeline_version="0.2.4",
        rating="Overweight",
        confidence=0.75,
        executive_summary="Data center demand remains strong. Overweight with $950 target.",
        investment_thesis="NVDA's data center revenue grew 73% YoY driven by AI infrastructure.",
        evidence=[
            EvidenceItem(
                source_type=EvidenceType.FUNDAMENTAL,
                source_name="SEC 10-Q Filing",
                source_url="https://sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=NVDA",
                available_at="2026-04-25",
                claim="Data center revenue grew 73% YoY to $26.3B.",
                supporting=True,
                confidence=0.95,
            ),
            EvidenceItem(
                source_type=EvidenceType.PRICE_DATA,
                source_name="Yahoo Finance OHLCV",
                available_at="2026-04-28",
                claim="Stock trading at $890, 5% below 52-week high.",
                supporting=True,
                confidence=0.90,
            ),
            EvidenceItem(
                source_type=EvidenceType.NEWS,
                source_name="Reuters",
                available_at="2026-04-27",
                claim="China export restrictions may limit H200 shipments.",
                supporting=False,
                confidence=0.70,
            ),
            EvidenceItem(
                source_type=EvidenceType.TECHNICAL_INDICATOR,
                source_name="StockStats RSI/SMA",
                available_at="2026-04-28",
                claim="RSI at 58, above 50-day SMA, bullish momentum intact.",
                supporting=True,
                confidence=0.80,
            ),
        ],
        risks=[
            RiskItem(
                category="regulatory",
                description="China export restrictions could reduce H200 revenue by 10-15%.",
                severity="high",
                probability=0.4,
                mitigation="Diversified customer base; domestic demand offsets.",
            ),
            RiskItem(
                category="valuation",
                description="Forward P/E of 35x leaves limited margin of safety.",
                severity="medium",
            ),
        ],
        expected_catalyst="Q3 earnings report in late July with data center guidance.",
        invalidation_condition="Data center revenue growth decelerates below 40% YoY.",
        signal=SignalObject(
            ticker="NVDA",
            direction=SignalDirection.LONG,
            conviction=0.75,
            target_horizon_days=60,
            entry_price=890.0,
            stop_loss=820.0,
            take_profit=950.0,
            max_position_pct=0.06,
        ),
        bull_case_summary="AI infrastructure spending is secular, not cyclical. NVDA dominates.",
        bear_case_summary="Valuation stretched; China risk underpriced; competition from AMD/custom silicon.",
        debate_winner="bull",
        risk_assessment_summary="Moderate risk. Position sizing should reflect regulatory uncertainty.",
    )


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchemaValidation:
    def test_minimal_thesis_valid(self, minimal_thesis):
        assert minimal_thesis.ticker == "AAPL"
        assert minimal_thesis.rating == "Buy"
        assert minimal_thesis.confidence == 0.82

    def test_full_thesis_valid(self, full_thesis):
        assert full_thesis.ticker == "NVDA"
        assert len(full_thesis.evidence) == 4
        assert len(full_thesis.risks) == 2
        assert full_thesis.signal is not None
        assert full_thesis.signal.direction == SignalDirection.LONG

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValidationError):
            AgentThesis(
                ticker="AAPL",
                trade_date="2026-04-28",
                rating="Buy",
                confidence=1.5,  # > 1.0
                executive_summary="Test",
                investment_thesis="Test",
            )

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValidationError):
            AgentThesis(
                ticker="AAPL",
                trade_date="2026-04-28",
                rating="Buy",
                confidence=-0.1,  # < 0.0
                executive_summary="Test",
                investment_thesis="Test",
            )

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AgentThesis(
                ticker="AAPL",
                trade_date="2026-04-28",
                # missing rating
                confidence=0.5,
                executive_summary="Test",
                investment_thesis="Test",
            )

    def test_invalid_trade_date_format(self):
        with pytest.raises(ValidationError):
            AgentThesis(
                ticker="AAPL",
                trade_date="not-a-date",
                rating="Hold",
                confidence=0.5,
                executive_summary="Test",
                investment_thesis="Test",
            )

    def test_evidence_confidence_range(self):
        with pytest.raises(ValidationError):
            EvidenceItem(
                source_type=EvidenceType.NEWS,
                source_name="Test",
                claim="Test claim",
                confidence=2.0,  # > 1.0
            )

    def test_signal_conviction_range(self):
        with pytest.raises(ValidationError):
            SignalObject(
                ticker="AAPL",
                direction=SignalDirection.LONG,
                conviction=1.5,  # > 1.0
            )

    def test_risk_severity_accepts_any_string(self):
        """Risk severity is a free string field (low/medium/high/critical)."""
        r = RiskItem(category="test", description="test risk", severity="critical")
        assert r.severity == "critical"


# ---------------------------------------------------------------------------
# Time-Safety Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTimeSafety:
    def test_no_violations_when_evidence_before_trade_date(self, full_thesis):
        violations = full_thesis.check_time_safety()
        assert violations == [], f"Unexpected violations: {violations}"

    def test_detects_lookahead_bias(self):
        thesis = AgentThesis(
            ticker="AAPL",
            trade_date="2026-04-28",
            rating="Buy",
            confidence=0.8,
            executive_summary="Test",
            investment_thesis="Test",
            evidence=[
                EvidenceItem(
                    source_type=EvidenceType.NEWS,
                    source_name="Future News",
                    available_at="2026-04-30",  # AFTER trade_date
                    claim="Earnings beat expectations.",
                    confidence=0.9,
                ),
            ],
        )
        violations = thesis.check_time_safety()
        assert len(violations) == 1
        assert "lookahead bias" in violations[0]

    def test_no_violation_when_available_at_equals_trade_date(self):
        thesis = AgentThesis(
            ticker="AAPL",
            trade_date="2026-04-28",
            rating="Hold",
            confidence=0.5,
            executive_summary="Test",
            investment_thesis="Test",
            evidence=[
                EvidenceItem(
                    source_type=EvidenceType.PRICE_DATA,
                    source_name="Yahoo Finance",
                    available_at="2026-04-28",  # Same as trade_date
                    claim="Price closed at $195.",
                    confidence=0.95,
                ),
            ],
        )
        violations = thesis.check_time_safety()
        assert violations == []

    def test_no_violation_when_available_at_is_none(self, minimal_thesis):
        """Evidence without available_at should not trigger violations."""
        minimal_thesis.evidence = [
            EvidenceItem(
                source_type=EvidenceType.OTHER,
                source_name="Manual input",
                claim="Analyst opinion.",
                confidence=0.5,
            ),
        ]
        violations = minimal_thesis.check_time_safety()
        assert violations == []

    def test_multiple_violations_detected(self):
        thesis = AgentThesis(
            ticker="TSLA",
            trade_date="2026-04-01",
            rating="Sell",
            confidence=0.6,
            executive_summary="Test",
            investment_thesis="Test",
            evidence=[
                EvidenceItem(
                    source_type=EvidenceType.NEWS,
                    source_name="Future News 1",
                    available_at="2026-04-05",
                    claim="Claim 1",
                    confidence=0.5,
                ),
                EvidenceItem(
                    source_type=EvidenceType.FUNDAMENTAL,
                    source_name="Future Filing",
                    available_at="2026-04-10",
                    claim="Claim 2",
                    confidence=0.5,
                ),
            ],
        )
        violations = thesis.check_time_safety()
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# Evidence Coverage
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvidenceCoverage:
    def test_full_coverage(self, full_thesis):
        score = full_thesis.evidence_coverage_score()
        assert score == 1.0  # All 4 core types present

    def test_no_evidence(self, minimal_thesis):
        score = minimal_thesis.evidence_coverage_score()
        assert score == 0.0

    def test_partial_coverage(self):
        thesis = AgentThesis(
            ticker="MSFT",
            trade_date="2026-04-28",
            rating="Hold",
            confidence=0.5,
            executive_summary="Test",
            investment_thesis="Test",
            evidence=[
                EvidenceItem(
                    source_type=EvidenceType.PRICE_DATA,
                    source_name="Yahoo",
                    claim="Price at $420.",
                    confidence=0.9,
                ),
                EvidenceItem(
                    source_type=EvidenceType.NEWS,
                    source_name="Reuters",
                    claim="Azure growth strong.",
                    confidence=0.8,
                ),
            ],
        )
        score = thesis.evidence_coverage_score()
        assert score == 0.5  # 2 out of 4 core types


# ---------------------------------------------------------------------------
# JSON Serialization Round-Trip
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSerialization:
    def test_json_round_trip_minimal(self, minimal_thesis):
        json_str = minimal_thesis.to_json()
        restored = AgentThesis.from_json(json_str)
        assert restored.ticker == minimal_thesis.ticker
        assert restored.rating == minimal_thesis.rating
        assert restored.confidence == minimal_thesis.confidence

    def test_json_round_trip_full(self, full_thesis):
        json_str = full_thesis.to_json()
        restored = AgentThesis.from_json(json_str)
        assert restored.ticker == full_thesis.ticker
        assert len(restored.evidence) == len(full_thesis.evidence)
        assert len(restored.risks) == len(full_thesis.risks)
        assert restored.signal.direction == full_thesis.signal.direction

    def test_json_is_valid_json(self, full_thesis):
        json_str = full_thesis.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["ticker"] == "NVDA"

    def test_from_invalid_json_raises(self):
        with pytest.raises(Exception):
            AgentThesis.from_json("not valid json")

    def test_from_incomplete_json_raises(self):
        with pytest.raises(Exception):
            AgentThesis.from_json('{"ticker": "AAPL"}')


# ---------------------------------------------------------------------------
# Markdown Rendering
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMarkdownRendering:
    def test_minimal_renders(self, minimal_thesis):
        md = minimal_thesis.render_markdown()
        assert "# Thesis: AAPL" in md
        assert "**Rating**: Buy" in md
        assert "82%" in md

    def test_full_renders_all_sections(self, full_thesis):
        md = full_thesis.render_markdown()
        assert "# Thesis: NVDA" in md
        assert "## Evidence" in md
        assert "## Risks" in md
        assert "## Debate Record" in md
        assert "## Signal Object" in md
        assert "## Expected Catalyst" in md
        assert "## Invalidation Condition" in md

    def test_contradicting_evidence_labeled(self, full_thesis):
        md = full_thesis.render_markdown()
        assert "[Contradicting]" in md
        assert "[Supporting]" in md


# ---------------------------------------------------------------------------
# Content Hash
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestContentHash:
    def test_hash_deterministic(self):
        content = "Apple Q2 revenue: $95.4B"
        h1 = EvidenceItem.compute_hash(content)
        h2 = EvidenceItem.compute_hash(content)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_hash_changes_with_content(self):
        h1 = EvidenceItem.compute_hash("version 1")
        h2 = EvidenceItem.compute_hash("version 2")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Malformed LLM Output Handling
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMalformedOutput:
    """Tests that the schema correctly rejects malformed data that might
    come from an LLM producing invalid structured output."""

    def test_rejects_empty_ticker(self):
        with pytest.raises(ValidationError):
            AgentThesis(
                ticker="",  # Empty string — Pydantic won't catch this by default
                trade_date="2026-04-28",
                rating="Buy",
                confidence=0.5,
                executive_summary="",
                investment_thesis="",
            )
        # Note: empty strings are valid for Pydantic str fields.
        # If we want to reject them, we need min_length validators.
        # This test documents current behavior.

    def test_rejects_none_for_required_fields(self):
        with pytest.raises(ValidationError):
            AgentThesis(
                ticker=None,
                trade_date="2026-04-28",
                rating="Buy",
                confidence=0.5,
                executive_summary="Test",
                investment_thesis="Test",
            )

    def test_rejects_string_confidence(self):
        """LLMs sometimes return numbers as strings."""
        with pytest.raises(ValidationError):
            AgentThesis(
                ticker="AAPL",
                trade_date="2026-04-28",
                rating="Buy",
                confidence="high",  # Not a number
                executive_summary="Test",
                investment_thesis="Test",
            )

    def test_accepts_numeric_string_confidence(self):
        """Pydantic coerces '0.8' to 0.8 — this is acceptable."""
        thesis = AgentThesis(
            ticker="AAPL",
            trade_date="2026-04-28",
            rating="Buy",
            confidence="0.8",  # String that can be coerced
            executive_summary="Test",
            investment_thesis="Test",
        )
        assert thesis.confidence == 0.8

    def test_rejects_invalid_evidence_type(self):
        with pytest.raises(ValidationError):
            EvidenceItem(
                source_type="made_up_type",
                source_name="Test",
                claim="Test",
                confidence=0.5,
            )

    def test_rejects_invalid_signal_direction(self):
        with pytest.raises(ValidationError):
            SignalObject(
                ticker="AAPL",
                direction="sideways",  # Not a valid enum
                conviction=0.5,
            )
