"""Kill Switch & Circuit Breaker System.

Provides multi-level automated and manual trading halts:

Level 1 — Circuit Breakers (auto-triggered):
  - Portfolio drawdown exceeds threshold (e.g., -5% intraday)
  - Daily loss limit breached
  - Consecutive losing signals exceed threshold
  - Anomalous signal frequency (too many signals in short window)
  - LLM output fails schema validation repeatedly

Level 2 — Manual Kill Switch:
  - Operator-triggered full halt (persisted to disk)
  - Per-ticker halt (block specific symbols)
  - Time-based halt (halt until market open next day)

All state is persisted to a JSON file for crash recovery.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

KILL_SWITCH_FILE = "kill_switch_state.json"


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------

class HaltLevel(str, Enum):
    NONE = "none"
    TICKER = "ticker"          # Single ticker halted
    CIRCUIT_BREAKER = "circuit_breaker"  # Auto-triggered
    MANUAL = "manual"          # Operator-triggered
    EMERGENCY = "emergency"    # Full system halt


class CircuitBreakerType(str, Enum):
    DRAWDOWN = "drawdown"
    DAILY_LOSS = "daily_loss"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    SIGNAL_FREQUENCY = "signal_frequency"
    SCHEMA_FAILURES = "schema_failures"


@dataclass
class CircuitBreakerConfig:
    """Thresholds for automatic circuit breakers."""
    max_drawdown_pct: float = 0.05          # 5% portfolio drawdown triggers halt
    max_daily_loss_pct: float = 0.03        # 3% daily loss triggers halt
    max_consecutive_losses: int = 5         # 5 consecutive losing signals
    max_signals_per_hour: int = 20          # Anomalous signal frequency
    max_schema_failures: int = 3            # LLM schema validation failures
    auto_reset_hours: int = 24              # Auto-reset circuit breakers after N hours


@dataclass
class HaltEvent:
    """Record of a halt event."""
    halt_id: str
    level: HaltLevel
    reason: str
    triggered_at: str
    triggered_by: str          # "auto" | "operator"
    reset_at: Optional[str] = None
    tickers_affected: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class KillSwitchState:
    """Persisted kill switch state."""
    system_halted: bool = False
    halt_level: HaltLevel = HaltLevel.NONE
    halt_reason: str = ""
    halted_at: Optional[str] = None
    halted_tickers: List[str] = field(default_factory=list)
    consecutive_losses: int = 0
    schema_failures_today: int = 0
    signals_last_hour: List[str] = field(default_factory=list)  # timestamps
    halt_history: List[Dict] = field(default_factory=list)
    circuit_breaker_resets: Dict[str, str] = field(default_factory=dict)  # type -> reset_time


# ---------------------------------------------------------------------------
# Kill Switch Manager
# ---------------------------------------------------------------------------

class KillSwitchManager:
    """Multi-level kill switch and circuit breaker manager.

    Usage:
        ks = KillSwitchManager()
        # Check before any order
        if ks.is_halted():
            return
        # Auto-trigger on drawdown
        ks.check_drawdown(current_nav=95_000, peak_nav=100_000)
        # Manual halt
        ks.manual_halt("Risk review required")
        # Reset
        ks.reset("Risk review complete")
    """

    def __init__(
        self,
        state_file: str = KILL_SWITCH_FILE,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.state_file = state_file
        self.config = config or CircuitBreakerConfig()
        self.state = self._load_state()

    # ------------------------------------------------------------------
    # Core halt checks
    # ------------------------------------------------------------------

    def is_halted(self, ticker: Optional[str] = None) -> bool:
        """Check if trading is halted (system-wide or for a specific ticker)."""
        if self.state.system_halted:
            return True
        if ticker and ticker.upper() in self.state.halted_tickers:
            return True
        return False

    def get_halt_status(self) -> Dict:
        """Get current halt status summary."""
        return {
            "system_halted": self.state.system_halted,
            "halt_level": self.state.halt_level.value,
            "halt_reason": self.state.halt_reason,
            "halted_at": self.state.halted_at,
            "halted_tickers": self.state.halted_tickers,
            "consecutive_losses": self.state.consecutive_losses,
            "schema_failures_today": self.state.schema_failures_today,
        }

    # ------------------------------------------------------------------
    # Circuit breakers (auto-triggered)
    # ------------------------------------------------------------------

    def check_drawdown(self, current_nav: float, peak_nav: float) -> bool:
        """Trigger circuit breaker if drawdown exceeds threshold.

        Returns True if circuit breaker was triggered.
        """
        if peak_nav <= 0:
            return False
        drawdown = (peak_nav - current_nav) / peak_nav
        if drawdown >= self.config.max_drawdown_pct:
            self._trigger_circuit_breaker(
                cb_type=CircuitBreakerType.DRAWDOWN,
                reason=f"Portfolio drawdown {drawdown:.1%} exceeds limit {self.config.max_drawdown_pct:.1%}",
                metadata={"drawdown_pct": drawdown, "current_nav": current_nav, "peak_nav": peak_nav},
            )
            return True
        return False

    def check_daily_loss(self, daily_pnl: float, portfolio_nav: float) -> bool:
        """Trigger circuit breaker if daily loss exceeds threshold."""
        if portfolio_nav <= 0 or daily_pnl >= 0:
            return False
        loss_pct = abs(daily_pnl) / portfolio_nav
        if loss_pct >= self.config.max_daily_loss_pct:
            self._trigger_circuit_breaker(
                cb_type=CircuitBreakerType.DAILY_LOSS,
                reason=f"Daily loss {loss_pct:.1%} exceeds limit {self.config.max_daily_loss_pct:.1%}",
                metadata={"daily_pnl": daily_pnl, "loss_pct": loss_pct},
            )
            return True
        return False

    def record_signal_outcome(self, ticker: str, was_profitable: bool) -> bool:
        """Record a signal outcome. Triggers CB if too many consecutive losses.

        Returns True if circuit breaker was triggered.
        """
        if was_profitable:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        self._save_state()

        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger_circuit_breaker(
                cb_type=CircuitBreakerType.CONSECUTIVE_LOSSES,
                reason=f"{self.state.consecutive_losses} consecutive losing signals",
                metadata={"consecutive_losses": self.state.consecutive_losses, "last_ticker": ticker},
            )
            return True
        return False

    def record_signal_submitted(self, signal_id: str) -> bool:
        """Record a signal submission. Triggers CB if frequency is anomalous.

        Returns True if circuit breaker was triggered.
        """
        now = datetime.utcnow()
        cutoff = (now - timedelta(hours=1)).isoformat()

        # Prune old entries
        self.state.signals_last_hour = [
            ts for ts in self.state.signals_last_hour if ts >= cutoff
        ]
        self.state.signals_last_hour.append(now.isoformat())
        self._save_state()

        if len(self.state.signals_last_hour) > self.config.max_signals_per_hour:
            self._trigger_circuit_breaker(
                cb_type=CircuitBreakerType.SIGNAL_FREQUENCY,
                reason=f"Anomalous signal frequency: {len(self.state.signals_last_hour)} signals/hour",
                metadata={"signals_per_hour": len(self.state.signals_last_hour)},
            )
            return True
        return False

    def record_schema_failure(self) -> bool:
        """Record an LLM schema validation failure. Triggers CB if too many.

        Returns True if circuit breaker was triggered.
        """
        self.state.schema_failures_today += 1
        self._save_state()

        if self.state.schema_failures_today >= self.config.max_schema_failures:
            self._trigger_circuit_breaker(
                cb_type=CircuitBreakerType.SCHEMA_FAILURES,
                reason=f"{self.state.schema_failures_today} LLM schema failures today",
                metadata={"failures": self.state.schema_failures_today},
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def manual_halt(self, reason: str, tickers: Optional[List[str]] = None) -> HaltEvent:
        """Manually halt trading (operator-triggered).

        Args:
            reason: Human-readable reason for the halt.
            tickers: If provided, halt only these tickers. Otherwise halt system.

        Returns:
            HaltEvent record.
        """
        import uuid
        now = datetime.utcnow().isoformat()

        if tickers:
            for ticker in tickers:
                if ticker.upper() not in self.state.halted_tickers:
                    self.state.halted_tickers.append(ticker.upper())
            level = HaltLevel.TICKER
        else:
            self.state.system_halted = True
            self.state.halt_level = HaltLevel.MANUAL
            self.state.halt_reason = reason
            self.state.halted_at = now
            level = HaltLevel.MANUAL

        event = HaltEvent(
            halt_id=str(uuid.uuid4()),
            level=level,
            reason=reason,
            triggered_at=now,
            triggered_by="operator",
            tickers_affected=tickers or [],
        )
        self.state.halt_history.append(asdict(event))
        self._save_state()

        logger.warning(f"MANUAL HALT [{level.value}]: {reason}")
        return event

    def halt_ticker(self, ticker: str, reason: str = "") -> None:
        """Halt a specific ticker."""
        ticker = ticker.upper()
        if ticker not in self.state.halted_tickers:
            self.state.halted_tickers.append(ticker)
        self._save_state()
        logger.warning(f"Ticker halted: {ticker} — {reason}")

    def resume_ticker(self, ticker: str) -> None:
        """Resume trading for a specific ticker."""
        ticker = ticker.upper()
        if ticker in self.state.halted_tickers:
            self.state.halted_tickers.remove(ticker)
        self._save_state()
        logger.info(f"Ticker resumed: {ticker}")

    def reset(self, reason: str = "Manual reset") -> None:
        """Reset all circuit breakers and resume trading."""
        self.state.system_halted = False
        self.state.halt_level = HaltLevel.NONE
        self.state.halt_reason = ""
        self.state.halted_at = None
        self.state.consecutive_losses = 0
        self.state.schema_failures_today = 0
        self.state.circuit_breaker_resets = {}
        self._save_state()
        logger.info(f"Kill switch reset: {reason}")

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at market open each day)."""
        self.state.schema_failures_today = 0
        self.state.signals_last_hour = []
        # Auto-reset circuit breakers older than config.auto_reset_hours
        now = datetime.utcnow()
        if self.state.system_halted and self.state.halted_at:
            halted_at = datetime.fromisoformat(self.state.halted_at)
            if (now - halted_at).total_seconds() > self.config.auto_reset_hours * 3600:
                if self.state.halt_level == HaltLevel.CIRCUIT_BREAKER:
                    self.reset("Auto-reset after cooldown period")
                    logger.info("Circuit breaker auto-reset after cooldown")
        self._save_state()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trigger_circuit_breaker(
        self,
        cb_type: CircuitBreakerType,
        reason: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Trigger a circuit breaker halt."""
        import uuid
        if self.state.system_halted:
            return  # Already halted

        now = datetime.utcnow().isoformat()
        self.state.system_halted = True
        self.state.halt_level = HaltLevel.CIRCUIT_BREAKER
        self.state.halt_reason = f"[{cb_type.value}] {reason}"
        self.state.halted_at = now

        event = HaltEvent(
            halt_id=str(uuid.uuid4()),
            level=HaltLevel.CIRCUIT_BREAKER,
            reason=reason,
            triggered_at=now,
            triggered_by="auto",
            metadata=metadata or {},
        )
        self.state.halt_history.append(asdict(event))
        self._save_state()

        logger.critical(f"CIRCUIT BREAKER TRIGGERED [{cb_type.value}]: {reason}")

    def _load_state(self) -> KillSwitchState:
        """Load state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                state = KillSwitchState(**{
                    k: v for k, v in data.items()
                    if k in KillSwitchState.__dataclass_fields__
                })
                state.halt_level = HaltLevel(state.halt_level) if isinstance(state.halt_level, str) else state.halt_level
                return state
            except Exception as e:
                logger.warning(f"Failed to load kill switch state: {e} — using fresh state")
        return KillSwitchState()

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            data = asdict(self.state)
            data["halt_level"] = self.state.halt_level.value
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")
