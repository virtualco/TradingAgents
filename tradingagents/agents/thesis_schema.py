"""Extended thesis schema for institutional-grade structured output.

This module defines the full "decision object" that the roadmap requires:
rating, horizon, confidence, evidence list, risk list, expected catalyst,
invalidation condition, proposed signal, and portfolio action.

It wraps and extends the upstream schemas (ResearchPlan, TraderProposal,
PortfolioDecision) with evidence provenance, time-safety metadata, and
a machine-readable signal object suitable for downstream quant systems.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Evidence & Provenance
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    """Classification of evidence source."""
    PRICE_DATA = "price_data"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    FILING = "filing"
    TRANSCRIPT = "transcript"
    ANALYST_ESTIMATE = "analyst_estimate"
    TECHNICAL_INDICATOR = "technical_indicator"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    OTHER = "other"


class EvidenceItem(BaseModel):
    """A single piece of evidence supporting or contradicting a claim."""
    source_type: EvidenceType = Field(
        description="Classification of the evidence source."
    )
    source_name: str = Field(
        description="Human-readable source name, e.g. 'Yahoo Finance OHLCV', 'SEC 10-K Filing'."
    )
    source_url: Optional[str] = Field(
        default=None,
        description="URL or URI of the source, if available."
    )
    available_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp when this data was available (point-in-time). "
                    "Must not be after the trade_date to prevent lookahead bias."
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the raw source content for reproducibility."
    )
    claim: str = Field(
        description="The specific factual claim extracted from this source."
    )
    supporting: bool = Field(
        default=True,
        description="True if this evidence supports the thesis, False if it contradicts."
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this evidence item (0.0 to 1.0)."
    )

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content for reproducibility tracking."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class RiskItem(BaseModel):
    """A specific risk identified in the analysis."""
    category: str = Field(
        description="Risk category, e.g. 'earnings', 'liquidity', 'macro', 'crowding', 'event'."
    )
    description: str = Field(
        description="Description of the risk."
    )
    severity: str = Field(
        description="One of: low, medium, high, critical."
    )
    probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Estimated probability of the risk materializing (0.0 to 1.0)."
    )
    mitigation: Optional[str] = Field(
        default=None,
        description="Suggested mitigation or hedge."
    )


# ---------------------------------------------------------------------------
# Signal Object (machine-readable for downstream quant systems)
# ---------------------------------------------------------------------------

class SignalDirection(str, Enum):
    """Machine-readable signal direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalObject(BaseModel):
    """Machine-readable signal for downstream portfolio/execution systems."""
    ticker: str = Field(description="Instrument ticker symbol.")
    direction: SignalDirection = Field(description="Signal direction: long, short, or flat.")
    conviction: float = Field(
        ge=0.0, le=1.0,
        description="Signal conviction strength (0.0 to 1.0)."
    )
    target_horizon_days: Optional[int] = Field(
        default=None,
        description="Expected holding period in trading days."
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Suggested entry price."
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Suggested stop-loss price."
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Suggested take-profit price."
    )
    max_position_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum position size as fraction of portfolio (0.0 to 1.0)."
    )


# ---------------------------------------------------------------------------
# Full Thesis Object
# ---------------------------------------------------------------------------

class AgentThesis(BaseModel):
    """The complete structured thesis produced by the TradingAgents pipeline.

    This is the "decision object" that the roadmap requires: a single JSON
    document capturing the full output of a research cycle, suitable for
    audit, replay, and downstream consumption by quant systems.
    """

    # --- Metadata ---
    ticker: str = Field(min_length=1, description="Instrument ticker symbol.")
    trade_date: str = Field(
        description="ISO-8601 date for which this thesis was generated. "
                    "All evidence must have available_at <= trade_date."
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO-8601 timestamp when this thesis was generated."
    )
    pipeline_version: str = Field(
        default="0.2.4",
        description="Version of the TradingAgents pipeline that produced this thesis."
    )
    model_config_hash: Optional[str] = Field(
        default=None,
        description="Hash of the LLM configuration (model, temperature, etc.) for reproducibility."
    )

    # --- Core Decision ---
    rating: str = Field(
        description="Final rating: Buy, Overweight, Hold, Underweight, or Sell."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the thesis (0.0 to 1.0)."
    )
    executive_summary: str = Field(
        description="2-4 sentence action plan covering entry, sizing, risk, and horizon."
    )
    investment_thesis: str = Field(
        description="Detailed reasoning anchored in specific evidence."
    )

    # --- Evidence & Risk ---
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="List of evidence items supporting or contradicting the thesis."
    )
    risks: List[RiskItem] = Field(
        default_factory=list,
        description="List of identified risks."
    )

    # --- Catalyst & Invalidation ---
    expected_catalyst: Optional[str] = Field(
        default=None,
        description="The expected event or condition that would trigger the thesis."
    )
    invalidation_condition: Optional[str] = Field(
        default=None,
        description="Condition under which the thesis should be abandoned."
    )

    # --- Signal ---
    signal: Optional[SignalObject] = Field(
        default=None,
        description="Machine-readable signal for downstream systems."
    )

    # --- Debate Record ---
    bull_case_summary: Optional[str] = Field(
        default=None,
        description="Summary of the bull researcher's argument."
    )
    bear_case_summary: Optional[str] = Field(
        default=None,
        description="Summary of the bear researcher's argument."
    )
    debate_winner: Optional[str] = Field(
        default=None,
        description="Which side won the debate: 'bull', 'bear', or 'balanced'."
    )
    risk_assessment_summary: Optional[str] = Field(
        default=None,
        description="Summary from the risk debate (aggressive/conservative/neutral)."
    )

    # --- Agent Scoring (for future tournament tracking) ---
    analyst_contributions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-analyst contribution metadata for scoring/tournament tracking."
    )

    @field_validator("trade_date")
    @classmethod
    def validate_trade_date_format(cls, v: str) -> str:
        """Ensure trade_date is a valid ISO date."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"trade_date must be ISO-8601 format, got: {v}")
        return v

    def check_time_safety(self) -> List[str]:
        """Validate that no evidence has available_at after trade_date.

        Returns a list of violation descriptions. Empty list means safe.
        """
        violations = []
        for i, ev in enumerate(self.evidence):
            if ev.available_at and ev.available_at > self.trade_date:
                violations.append(
                    f"Evidence[{i}] ({ev.source_name}): available_at={ev.available_at} "
                    f"is after trade_date={self.trade_date} — lookahead bias!"
                )
        return violations

    def evidence_coverage_score(self) -> float:
        """Calculate what fraction of evidence types are represented."""
        if not self.evidence:
            return 0.0
        types_present = {ev.source_type for ev in self.evidence}
        # Core types we want coverage of
        core_types = {
            EvidenceType.PRICE_DATA,
            EvidenceType.FUNDAMENTAL,
            EvidenceType.NEWS,
            EvidenceType.TECHNICAL_INDICATOR,
        }
        return len(types_present & core_types) / len(core_types)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentThesis":
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)

    def render_markdown(self) -> str:
        """Render the thesis as a human-readable markdown report."""
        parts = [
            f"# Thesis: {self.ticker} ({self.trade_date})",
            "",
            f"**Rating**: {self.rating}  ",
            f"**Confidence**: {self.confidence:.0%}  ",
            f"**Generated**: {self.generated_at}  ",
            f"**Pipeline**: v{self.pipeline_version}",
            "",
            "## Executive Summary",
            "",
            self.executive_summary,
            "",
            "## Investment Thesis",
            "",
            self.investment_thesis,
        ]

        if self.expected_catalyst:
            parts.extend(["", "## Expected Catalyst", "", self.expected_catalyst])

        if self.invalidation_condition:
            parts.extend(["", "## Invalidation Condition", "", self.invalidation_condition])

        if self.evidence:
            parts.extend(["", "## Evidence", ""])
            for i, ev in enumerate(self.evidence, 1):
                direction = "Supporting" if ev.supporting else "Contradicting"
                parts.append(
                    f"{i}. **[{direction}]** ({ev.source_type.value}) {ev.claim} "
                    f"— *{ev.source_name}* (confidence: {ev.confidence:.0%})"
                )

        if self.risks:
            parts.extend(["", "## Risks", ""])
            for r in self.risks:
                parts.append(f"- **{r.category}** [{r.severity}]: {r.description}")
                if r.mitigation:
                    parts.append(f"  - Mitigation: {r.mitigation}")

        if self.bull_case_summary or self.bear_case_summary:
            parts.extend(["", "## Debate Record", ""])
            if self.bull_case_summary:
                parts.extend([f"**Bull Case**: {self.bull_case_summary}", ""])
            if self.bear_case_summary:
                parts.extend([f"**Bear Case**: {self.bear_case_summary}", ""])
            if self.debate_winner:
                parts.append(f"**Winner**: {self.debate_winner}")

        if self.signal:
            parts.extend([
                "", "## Signal Object", "",
                f"- Direction: {self.signal.direction.value}",
                f"- Conviction: {self.signal.conviction:.0%}",
            ])
            if self.signal.target_horizon_days:
                parts.append(f"- Horizon: {self.signal.target_horizon_days} days")
            if self.signal.entry_price:
                parts.append(f"- Entry: ${self.signal.entry_price:.2f}")
            if self.signal.stop_loss:
                parts.append(f"- Stop Loss: ${self.signal.stop_loss:.2f}")
            if self.signal.take_profit:
                parts.append(f"- Take Profit: ${self.signal.take_profit:.2f}")

        return "\n".join(parts)
