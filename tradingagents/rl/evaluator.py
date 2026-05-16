"""RL Evaluator — Compare RL sizing agent vs static baseline.

Provides A/B testing framework to quantify the RL agent's edge
over fixed-size position management.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .agent import RLSizingAgent
from .env import TradingSizingEnv

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of A/B comparison between RL and baseline."""
    # RL metrics
    rl_sharpe: float
    rl_max_dd: float
    rl_total_return: float
    rl_trade_count: int
    rl_avg_multiplier: float
    rl_calmar: float
    
    # Baseline metrics (static sizing, multiplier = 1.0)
    baseline_sharpe: float
    baseline_max_dd: float
    baseline_total_return: float
    baseline_trade_count: int
    baseline_calmar: float
    
    # Comparison
    sharpe_improvement: float
    dd_improvement: float
    return_improvement: float
    
    @property
    def rl_has_edge(self) -> bool:
        """Whether RL demonstrates meaningful improvement."""
        return (
            self.sharpe_improvement > 0.1 and  # At least 0.1 Sharpe improvement
            self.dd_improvement >= 0.0          # No worse drawdown
        )
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        verdict = "RL WINS" if self.rl_has_edge else "BASELINE WINS"
        return (
            f"[{verdict}] "
            f"Sharpe: {self.rl_sharpe:.3f} vs {self.baseline_sharpe:.3f} "
            f"(Δ{self.sharpe_improvement:+.3f}) | "
            f"DD: {self.rl_max_dd:.2%} vs {self.baseline_max_dd:.2%} | "
            f"Return: {self.rl_total_return:.2%} vs {self.baseline_total_return:.2%} | "
            f"Avg Multiplier: {self.rl_avg_multiplier:.2f}"
        )


class RLEvaluator:
    """Evaluates RL sizing agent against static baseline."""
    
    def __init__(
        self,
        base_qty: float = 1.0,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        max_position_pct: float = 0.15,
    ):
        self.base_qty = base_qty
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
    
    def compare(
        self,
        agent: RLSizingAgent,
        signals: list[dict],
        prices: np.ndarray,
    ) -> ComparisonResult:
        """Run A/B comparison on the same data.
        
        Args:
            agent: Trained RL sizing agent.
            signals: Signal history for evaluation period.
            prices: Aligned price array.
            
        Returns:
            ComparisonResult with both RL and baseline metrics.
        """
        # Run RL agent
        rl_stats = self._run_episode(signals, prices, agent=agent)
        
        # Run baseline (static multiplier = 1.0)
        baseline_stats = self._run_episode(signals, prices, agent=None)
        
        # Calculate improvements
        sharpe_imp = rl_stats["sharpe"] - baseline_stats["sharpe"]
        dd_imp = baseline_stats["max_drawdown"] - rl_stats["max_drawdown"]  # positive = RL better
        ret_imp = rl_stats["total_return"] - baseline_stats["total_return"]
        
        # Calmar ratio
        rl_calmar = rl_stats["total_return"] / max(rl_stats["max_drawdown"], 0.001)
        bl_calmar = baseline_stats["total_return"] / max(baseline_stats["max_drawdown"], 0.001)
        
        return ComparisonResult(
            rl_sharpe=rl_stats["sharpe"],
            rl_max_dd=rl_stats["max_drawdown"],
            rl_total_return=rl_stats["total_return"],
            rl_trade_count=rl_stats["trade_count"],
            rl_avg_multiplier=rl_stats["avg_multiplier"],
            rl_calmar=rl_calmar,
            baseline_sharpe=baseline_stats["sharpe"],
            baseline_max_dd=baseline_stats["max_drawdown"],
            baseline_total_return=baseline_stats["total_return"],
            baseline_trade_count=baseline_stats["trade_count"],
            baseline_calmar=bl_calmar,
            sharpe_improvement=sharpe_imp,
            dd_improvement=dd_imp,
            return_improvement=ret_imp,
        )
    
    def _run_episode(
        self,
        signals: list[dict],
        prices: np.ndarray,
        agent: Optional[RLSizingAgent] = None,
    ) -> dict:
        """Run a single episode with either RL agent or static baseline."""
        env = TradingSizingEnv(
            signals=signals,
            prices=prices,
            base_qty=self.base_qty,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            max_position_pct=self.max_position_pct,
        )
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            if agent is not None and agent.is_trained:
                multiplier = agent.predict(obs, deterministic=True)
            else:
                multiplier = 1.0  # static baseline
            
            action = np.array([multiplier], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        return env.episode_stats
    
    def multi_seed_comparison(
        self,
        agent: RLSizingAgent,
        signals: list[dict],
        prices: np.ndarray,
        n_seeds: int = 5,
    ) -> dict:
        """Run comparison across multiple random seeds for robustness.
        
        Returns aggregated statistics with confidence intervals.
        """
        rl_sharpes = []
        bl_sharpes = []
        rl_dds = []
        bl_dds = []
        
        for seed in range(n_seeds):
            # Slight noise to prices to test robustness
            noise = 1.0 + np.random.default_rng(seed).normal(0, 0.0001, len(prices))
            noisy_prices = prices * noise
            
            result = self.compare(agent, signals, noisy_prices)
            rl_sharpes.append(result.rl_sharpe)
            bl_sharpes.append(result.baseline_sharpe)
            rl_dds.append(result.rl_max_dd)
            bl_dds.append(result.baseline_max_dd)
        
        return {
            "rl_sharpe_mean": float(np.mean(rl_sharpes)),
            "rl_sharpe_std": float(np.std(rl_sharpes)),
            "baseline_sharpe_mean": float(np.mean(bl_sharpes)),
            "baseline_sharpe_std": float(np.std(bl_sharpes)),
            "sharpe_improvement_mean": float(np.mean(rl_sharpes) - np.mean(bl_sharpes)),
            "rl_dd_mean": float(np.mean(rl_dds)),
            "baseline_dd_mean": float(np.mean(bl_dds)),
            "consistent_improvement": all(
                rs > bs for rs, bs in zip(rl_sharpes, bl_sharpes)
            ),
        }
