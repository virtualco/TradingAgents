"""Thesis output validation — enforces schema at agent output boundaries.

This module provides validation functions that MUST be called at the output
boundary of each decision-making agent (Research Manager, Trader, Portfolio
Manager) to ensure that:

1. Required fields are present and well-formed.
2. Time-safety constraints are not violated (no lookahead bias).
3. Malformed LLM outputs are rejected before reaching the order manager.

Usage:
    from tradingagents.agents.thesis_validator import (
        validate_portfolio_decision,
        validate_trader_proposal,
        validate_research_plan,
        validate_agent_thesis,
        ThesisValidationError,
    )
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.thesis_schema import AgentThesis, SignalDirection

logger = logging.getLogger(__name__)


class ThesisValidationError(Exception):
    """Raised when an agent output fails schema validation.

    This error is intentionally NOT caught by the pipeline — it must
    propagate to halt the cycle and prevent malformed decisions from
    reaching the order manager.
    """

    def __init__(self, agent_name: str, violations: List[str]):
        self.agent_name = agent_name
        self.violations = violations
        msg = (
            f"[{agent_name}] Output validation failed with "
            f"{len(violations)} violation(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_research_plan(output: str, *, strict: bool = False) -> List[str]:
    """Validate Research Manager output.

    Args:
        output: The rendered markdown string from the Research Manager.
        strict: If True, require all fields to be non-empty.

    Returns:
        List of violation descriptions. Empty means valid.
    """
    violations = []

    if not output or not output.strip():
        violations.append("Research plan output is empty.")
        return violations

    # Must contain a recommendation keyword
    valid_ratings = {"buy", "overweight", "hold", "underweight", "sell"}
    lower = output.lower()
    has_rating = any(f"**{r}**" in lower or f"recommendation**: {r}" in lower for r in valid_ratings)
    if not has_rating:
        # Looser check: any rating word present
        has_rating = any(r in lower for r in valid_ratings)

    if not has_rating:
        violations.append(
            "Research plan does not contain a valid recommendation "
            "(Buy/Overweight/Hold/Underweight/Sell)."
        )

    if strict:
        if "rationale" not in lower and "reasoning" not in lower:
            violations.append("Research plan missing rationale/reasoning section.")
        if "strategic" not in lower and "action" not in lower:
            violations.append("Research plan missing strategic actions section.")

    if len(output.strip()) < 50:
        violations.append(
            f"Research plan is suspiciously short ({len(output.strip())} chars). "
            "Minimum expected: 50 characters."
        )

    return violations


def validate_trader_proposal(output: str, *, strict: bool = False) -> List[str]:
    """Validate Trader output.

    Args:
        output: The rendered markdown string from the Trader.
        strict: If True, require all fields to be non-empty.

    Returns:
        List of violation descriptions. Empty means valid.
    """
    violations = []

    if not output or not output.strip():
        violations.append("Trader proposal output is empty.")
        return violations

    lower = output.lower()
    valid_actions = {"buy", "hold", "sell"}
    has_action = any(a in lower for a in valid_actions)

    if not has_action:
        violations.append(
            "Trader proposal does not contain a valid action (Buy/Hold/Sell)."
        )

    # Must have FINAL TRANSACTION PROPOSAL line for backward compat
    if "final transaction proposal" not in lower:
        violations.append(
            "Trader proposal missing 'FINAL TRANSACTION PROPOSAL' line "
            "(required for downstream parsing)."
        )

    if strict:
        if "reasoning" not in lower and "rationale" not in lower:
            violations.append("Trader proposal missing reasoning section.")

    if len(output.strip()) < 30:
        violations.append(
            f"Trader proposal is suspiciously short ({len(output.strip())} chars). "
            "Minimum expected: 30 characters."
        )

    return violations


def validate_portfolio_decision(output: str, *, strict: bool = False) -> List[str]:
    """Validate Portfolio Manager output.

    Args:
        output: The rendered markdown string from the Portfolio Manager.
        strict: If True, require all fields to be non-empty.

    Returns:
        List of violation descriptions. Empty means valid.
    """
    violations = []

    if not output or not output.strip():
        violations.append("Portfolio decision output is empty.")
        return violations

    lower = output.lower()
    valid_ratings = {"buy", "overweight", "hold", "underweight", "sell"}
    has_rating = any(r in lower for r in valid_ratings)

    if not has_rating:
        violations.append(
            "Portfolio decision does not contain a valid rating "
            "(Buy/Overweight/Hold/Underweight/Sell)."
        )

    if strict:
        if "executive summary" not in lower and "summary" not in lower:
            violations.append("Portfolio decision missing executive summary.")
        if "investment thesis" not in lower and "thesis" not in lower:
            violations.append("Portfolio decision missing investment thesis.")

    if len(output.strip()) < 50:
        violations.append(
            f"Portfolio decision is suspiciously short ({len(output.strip())} chars). "
            "Minimum expected: 50 characters."
        )

    return violations


def validate_agent_thesis(thesis: AgentThesis, *, strict: bool = True) -> List[str]:
    """Validate a fully-constructed AgentThesis object.

    This is the strongest validation — called when the factory builds
    the full thesis object for downstream consumption.

    Args:
        thesis: The AgentThesis instance to validate.
        strict: If True, enforce time-safety and evidence requirements.

    Returns:
        List of violation descriptions. Empty means valid.
    """
    violations = []

    # Required fields
    if not thesis.ticker or not thesis.ticker.strip():
        violations.append("AgentThesis.ticker is empty.")

    if not thesis.trade_date:
        violations.append("AgentThesis.trade_date is empty.")

    if not thesis.rating:
        violations.append("AgentThesis.rating is empty.")

    valid_ratings = {"Buy", "Overweight", "Hold", "Underweight", "Sell"}
    if thesis.rating not in valid_ratings:
        violations.append(
            f"AgentThesis.rating '{thesis.rating}' is not in valid set: {valid_ratings}"
        )

    if not thesis.executive_summary or len(thesis.executive_summary.strip()) < 20:
        violations.append(
            "AgentThesis.executive_summary is missing or too short (< 20 chars)."
        )

    if not thesis.investment_thesis or len(thesis.investment_thesis.strip()) < 30:
        violations.append(
            "AgentThesis.investment_thesis is missing or too short (< 30 chars)."
        )

    # Confidence bounds
    if thesis.confidence < 0.0 or thesis.confidence > 1.0:
        violations.append(
            f"AgentThesis.confidence={thesis.confidence} is out of range [0.0, 1.0]."
        )

    # Time-safety (critical for preventing lookahead bias)
    if strict:
        time_violations = thesis.check_time_safety()
        violations.extend(time_violations)

    # Signal validation
    if thesis.signal:
        if thesis.signal.conviction < 0.0 or thesis.signal.conviction > 1.0:
            violations.append(
                f"Signal.conviction={thesis.signal.conviction} is out of range [0.0, 1.0]."
            )
        if thesis.signal.stop_loss and thesis.signal.entry_price:
            if thesis.signal.direction == SignalDirection.LONG:
                if thesis.signal.stop_loss >= thesis.signal.entry_price:
                    violations.append(
                        f"LONG signal has stop_loss ({thesis.signal.stop_loss}) >= "
                        f"entry_price ({thesis.signal.entry_price})."
                    )
            elif thesis.signal.direction == SignalDirection.SHORT:
                if thesis.signal.stop_loss <= thesis.signal.entry_price:
                    violations.append(
                        f"SHORT signal has stop_loss ({thesis.signal.stop_loss}) <= "
                        f"entry_price ({thesis.signal.entry_price})."
                    )

    # Evidence requirements (strict mode)
    if strict and not thesis.evidence:
        violations.append(
            "AgentThesis has no evidence items. At least one evidence item is required."
        )

    return violations


# ---------------------------------------------------------------------------
# Enforcement wrappers (for use in agent nodes)
# ---------------------------------------------------------------------------


def enforce_research_plan(output: str, *, strict: bool = False) -> str:
    """Validate and return research plan output, raising on failure."""
    violations = validate_research_plan(output, strict=strict)
    if violations:
        logger.error(f"Research Manager validation failed: {violations}")
        raise ThesisValidationError("Research Manager", violations)
    return output


def enforce_trader_proposal(output: str, *, strict: bool = False) -> str:
    """Validate and return trader proposal output, raising on failure."""
    violations = validate_trader_proposal(output, strict=strict)
    if violations:
        logger.error(f"Trader validation failed: {violations}")
        raise ThesisValidationError("Trader", violations)
    return output


def enforce_portfolio_decision(output: str, *, strict: bool = False) -> str:
    """Validate and return portfolio decision output, raising on failure."""
    violations = validate_portfolio_decision(output, strict=strict)
    if violations:
        logger.error(f"Portfolio Manager validation failed: {violations}")
        raise ThesisValidationError("Portfolio Manager", violations)
    return output


def enforce_agent_thesis(thesis: AgentThesis, *, strict: bool = True) -> AgentThesis:
    """Validate and return AgentThesis, raising on failure."""
    violations = validate_agent_thesis(thesis, strict=strict)
    if violations:
        logger.error(f"AgentThesis validation failed: {violations}")
        raise ThesisValidationError("AgentThesis", violations)
    return thesis
