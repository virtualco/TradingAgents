#!/usr/bin/env python3
"""Smoke test: validate that a mock pipeline run produces a valid AgentThesis.

This script does NOT call any LLM — it constructs a thesis from mock data
to verify the schema, time-safety checks, serialization, and rendering
all work end-to-end in the current environment.

Usage:
    python scripts/smoke_thesis_output.py
"""
import json
import sys

from tradingagents.agents.thesis_schema import (
    AgentThesis,
    EvidenceItem,
    EvidenceType,
    RiskItem,
    SignalDirection,
    SignalObject,
)


def build_mock_thesis() -> AgentThesis:
    """Construct a realistic thesis from mock data."""
    return AgentThesis(
        ticker="AAPL",
        trade_date="2026-04-28",
        pipeline_version="0.2.4",
        rating="Overweight",
        confidence=0.78,
        executive_summary=(
            "Apple beat Q2 estimates with $95.4B revenue (+8% YoY). "
            "Services hit record $24.2B. Enter near $195 with stop at $180."
        ),
        investment_thesis=(
            "iPhone revenue stabilized at $46.8B despite macro headwinds. "
            "Services margin expansion to 75% drives FCF growth. "
            "Buyback program reduces share count by 3% annually."
        ),
        evidence=[
            EvidenceItem(
                source_type=EvidenceType.FUNDAMENTAL,
                source_name="SEC 10-Q Filing (Q2 2026)",
                available_at="2026-04-25",
                claim="Revenue $95.4B, beating consensus of $91.8B by 3.9%.",
                supporting=True,
                confidence=0.95,
                content_hash=EvidenceItem.compute_hash("Revenue $95.4B"),
            ),
            EvidenceItem(
                source_type=EvidenceType.PRICE_DATA,
                source_name="Yahoo Finance OHLCV",
                available_at="2026-04-28",
                claim="Stock at $195.20, RSI 55, above 50-day SMA.",
                supporting=True,
                confidence=0.90,
            ),
            EvidenceItem(
                source_type=EvidenceType.NEWS,
                source_name="Reuters",
                available_at="2026-04-27",
                claim="China tariff risk could impact iPhone shipments by 5-10%.",
                supporting=False,
                confidence=0.65,
            ),
            EvidenceItem(
                source_type=EvidenceType.TECHNICAL_INDICATOR,
                source_name="StockStats SMA/RSI",
                available_at="2026-04-28",
                claim="Golden cross on 50/200 SMA. Bullish momentum.",
                supporting=True,
                confidence=0.80,
            ),
        ],
        risks=[
            RiskItem(
                category="regulatory",
                description="China tariffs could reduce iPhone revenue by 5-10%.",
                severity="medium",
                probability=0.3,
                mitigation="Diversified supply chain; India manufacturing ramp.",
            ),
            RiskItem(
                category="macro",
                description="Consumer spending slowdown in US/EU.",
                severity="medium",
            ),
        ],
        expected_catalyst="WWDC 2026 AI announcements in June.",
        invalidation_condition="Services revenue growth decelerates below 10% YoY.",
        signal=SignalObject(
            ticker="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.78,
            target_horizon_days=45,
            entry_price=195.20,
            stop_loss=180.00,
            take_profit=215.00,
            max_position_pct=0.05,
        ),
        bull_case_summary="Services flywheel + buybacks + AI catalyst = sustained growth.",
        bear_case_summary="China risk + premium valuation + macro headwinds.",
        debate_winner="bull",
        risk_assessment_summary="Moderate risk. Position sizing reflects tariff uncertainty.",
    )


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST: AgentThesis Structured Output")
    print("=" * 60)

    # 1. Build thesis
    thesis = build_mock_thesis()
    print(f"\n✓ Thesis constructed for {thesis.ticker} ({thesis.trade_date})")

    # 2. Time-safety check
    violations = thesis.check_time_safety()
    if violations:
        print(f"\n✗ TIME-SAFETY VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("✓ Time-safety check passed (no lookahead bias)")

    # 3. Evidence coverage
    coverage = thesis.evidence_coverage_score()
    print(f"✓ Evidence coverage: {coverage:.0%} (4 core types)")

    # 4. JSON serialization
    json_str = thesis.to_json()
    parsed = json.loads(json_str)
    assert parsed["ticker"] == "AAPL"
    assert len(parsed["evidence"]) == 4
    print(f"✓ JSON serialization: {len(json_str)} bytes")

    # 5. Round-trip
    restored = AgentThesis.from_json(json_str)
    assert restored.ticker == thesis.ticker
    assert restored.confidence == thesis.confidence
    assert len(restored.evidence) == len(thesis.evidence)
    print("✓ JSON round-trip: identical")

    # 6. Markdown rendering
    md = thesis.render_markdown()
    assert "# Thesis: AAPL" in md
    assert "## Evidence" in md
    assert "## Risks" in md
    assert "## Signal Object" in md
    print(f"✓ Markdown rendering: {len(md)} chars")

    # 7. Print the rendered markdown
    print("\n" + "=" * 60)
    print("RENDERED THESIS:")
    print("=" * 60)
    print(md)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
