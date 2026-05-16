"""RL Position Sizing Agent for TradingAgents.

This module implements a Reinforcement Learning overlay that dynamically
sizes positions based on market state. The quant signal pipeline remains
authoritative (direction + conviction), while the RL agent controls
*how much* capital to allocate per trade.
"""
from .env import TradingSizingEnv
from .agent import RLSizingAgent
from .trainer import RLTrainer
from .evaluator import RLEvaluator

__all__ = ["TradingSizingEnv", "RLSizingAgent", "RLTrainer", "RLEvaluator"]
