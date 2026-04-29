"""TradingAgents Research Factory.

Public API:
    from tradingagents.research import ResearchFactory, SignalRegistry, WalkForwardEngine
    from tradingagents.research import compute_multi_role_signals, WalkForwardConfig
"""
from .factory import ResearchFactory
from .signal_registry import SignalRegistry, SignalRecord, SignalDirection, SignalStatus
from .strategy_rules import (
    compute_multi_role_signals,
    MultiRoleSignals,
    RoleSignalSummary,
    RuleSignal,
    SignalStrength,
    TechnicalStrategyRules,
    FundamentalStrategyRules,
    SentimentStrategyRules,
)
from .walk_forward import WalkForwardEngine, WalkForwardConfig, WalkForwardResult

__all__ = [
    "ResearchFactory",
    "SignalRegistry",
    "SignalRecord",
    "SignalDirection",
    "SignalStatus",
    "compute_multi_role_signals",
    "MultiRoleSignals",
    "RoleSignalSummary",
    "RuleSignal",
    "SignalStrength",
    "TechnicalStrategyRules",
    "FundamentalStrategyRules",
    "SentimentStrategyRules",
    "WalkForwardEngine",
    "WalkForwardConfig",
    "WalkForwardResult",
]
