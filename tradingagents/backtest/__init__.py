"""TradingAgents Backtesting Engine.

Public API:
    from tradingagents.backtest import BacktestEngine, BacktestConfig, BacktestResult
    from tradingagents.backtest import PortfolioOptimizer, FactorRiskModel, PerformanceAnalytics
    from tradingagents.backtest import StressTester, StressScenario
"""
from .engine import BacktestEngine, BacktestConfig, BacktestResult, Trade, Position
from .optimizer import PortfolioOptimizer, OptimizationConfig, OptimizationResult
from .risk_model import FactorRiskModel, FactorExposures, RiskDecomposition
from .stress import StressTester, StressScenario, StressResult
from .analytics import PerformanceAnalytics, PerformanceReport

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "Position",
    "PortfolioOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "FactorRiskModel",
    "FactorExposures",
    "RiskDecomposition",
    "StressTester",
    "StressScenario",
    "StressResult",
    "PerformanceAnalytics",
    "PerformanceReport",
]
