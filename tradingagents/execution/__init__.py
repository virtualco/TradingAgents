"""TradingAgents Execution Package.

Paper trading execution engine with institutional-grade risk controls:
- PaperOrderManager: Pre-trade risk checks, order lifecycle management
- KillSwitchManager: Multi-level circuit breakers and manual halts
- PositionTracker: Mark-to-market, NAV, drawdown monitoring
- ReconciliationEngine: Internal vs broker position comparison
- DailyObserver: Full daily observation cycle orchestrator
- ObservationLogger: Persistent observation records and readiness assessment
"""
from tradingagents.execution.order_manager import (
    PaperOrderManager,
    PreTradeRiskConfig,
    Order,
    OrderSide,
    OrderStatus,
    RejectionReason,
    PreTradeCheckResult,
)
from tradingagents.execution.kill_switch import (
    KillSwitchManager,
    CircuitBreakerConfig,
    HaltLevel,
    CircuitBreakerType,
    HaltEvent,
)
from tradingagents.execution.reconciliation import (
    PositionTracker,
    ReconciliationEngine,
    BrokerPosition,
    ReconciliationReport,
    ReconciliationBreak,
    PortfolioSnapshot,
)
from tradingagents.execution.observer import (
    DailyObserver,
    ObservationLogger,
    ObservationConfig,
    DailyObservation,
    ObservationSummary,
)

__all__ = [
    # Order manager
    "PaperOrderManager", "PreTradeRiskConfig", "Order",
    "OrderSide", "OrderStatus", "RejectionReason", "PreTradeCheckResult",
    # Kill switch
    "KillSwitchManager", "CircuitBreakerConfig", "HaltLevel",
    "CircuitBreakerType", "HaltEvent",
    # Reconciliation & tracking
    "PositionTracker", "ReconciliationEngine", "BrokerPosition",
    "ReconciliationReport", "ReconciliationBreak", "PortfolioSnapshot",
    # Observer
    "DailyObserver", "ObservationLogger", "ObservationConfig",
    "DailyObservation", "ObservationSummary",
]
