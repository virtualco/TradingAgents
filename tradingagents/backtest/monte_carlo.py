"""
Monte Carlo Robustness Validator
=================================
Implements multi-layered robustness testing for trading strategies:

1. Bootstrap Resampling — Scramble trade ordering (1,000+ simulations)
   to test if equity curve shape is path-dependent or genuinely robust.

2. Parameter Jitter — Perturb strategy parameters by ±10-20% (500+
   perturbations) to test if performance sits on a knife-edge optimum.

3. Execution Degradation — Systematically increase slippage/commission
   to find the break-even transaction cost threshold.

4. Composite Robustness Score — Weighted combination:
   - Probability of Profit: 40%
   - Consistency (Sharpe stability): 30%
   - Survival (probability of avoiding ruin): 30%

Usage:
    from tradingagents.backtest.monte_carlo import MonteCarloValidator, MCConfig

    validator = MonteCarloValidator()
    result = validator.validate(trades=trade_pnl_list, config=MCConfig())
    print(result.summary())
    
    # Gate deployment
    if result.robustness_score < 60:
        raise ValueError("Strategy fails robustness gate")

References:
    - StrategyQuant (2025): 5 Monte Carlo Methods
    - Strategy Arena (2026): Monte Carlo Simulation for Robustness
    - Slepaczuk (2026): Walk-forward with double OOS
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCConfig:
    """Configuration for Monte Carlo robustness validation."""
    # Bootstrap resampling
    n_bootstrap: int = 1000          # Number of trade-order scrambles
    initial_capital: float = 10000.0  # Starting equity
    ruin_threshold: float = 0.50      # 50% loss = ruin
    
    # Parameter jitter
    n_jitter: int = 500              # Number of parameter perturbations
    jitter_range: float = 0.15       # ±15% perturbation
    
    # Execution degradation
    slippage_steps: int = 20         # Number of slippage levels to test
    max_slippage_bps: float = 50.0   # Maximum slippage in basis points
    
    # Robustness score weights
    weight_profit_prob: float = 0.40
    weight_consistency: float = 0.30
    weight_survival: float = 0.30
    
    # Thresholds
    min_robustness_score: float = 60.0  # Minimum score for deployment
    min_profit_probability: float = 0.70
    min_wfe: float = 0.50


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Results from bootstrap resampling."""
    n_simulations: int
    profit_probability: float          # P(final_equity > initial)
    median_final_equity: float
    percentile_5: float                # 5th percentile outcome
    percentile_25: float
    percentile_75: float
    percentile_95: float
    mean_max_drawdown: float           # Average worst drawdown across sims
    worst_drawdown: float              # Worst drawdown across all sims
    ruin_probability: float            # P(equity drops below ruin threshold)
    equity_curves: Optional[np.ndarray] = None  # Shape: (n_sims, n_trades+1)


@dataclass
class JitterResult:
    """Results from parameter jitter testing."""
    n_perturbations: int
    original_sharpe: float
    mean_perturbed_sharpe: float
    std_perturbed_sharpe: float
    sharpe_degradation_pct: float      # How much Sharpe drops on average
    stability_score: float             # 0-100: higher = more stable
    pct_profitable: float              # % of perturbations still profitable
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)


@dataclass
class DegradationResult:
    """Results from execution degradation testing."""
    break_even_slippage_bps: float     # Slippage at which strategy breaks even
    slippage_levels: List[float] = field(default_factory=list)
    sharpe_at_levels: List[float] = field(default_factory=list)
    profit_at_levels: List[float] = field(default_factory=list)
    safety_margin_bps: float = 0.0     # How much slippage headroom exists


@dataclass
class MCResult:
    """Complete Monte Carlo validation result."""
    robustness_score: float            # 0-100 composite score
    passes_gate: bool                  # Whether strategy passes deployment gate
    
    # Component scores (0-100 each)
    profit_score: float
    consistency_score: float
    survival_score: float
    
    # Detailed results
    bootstrap: BootstrapResult
    jitter: JitterResult
    degradation: DegradationResult
    
    # Metadata
    n_trades_analysed: int = 0
    config: MCConfig = field(default_factory=MCConfig)
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passes_gate else "FAIL"
        return (
            f"Monte Carlo Robustness Validation [{status}]\n"
            f"{'=' * 50}\n"
            f"  Composite Score: {self.robustness_score:.1f}/100 "
            f"(threshold: {self.config.min_robustness_score:.0f})\n"
            f"  ├─ Profit Score:      {self.profit_score:.1f}/100 "
            f"(P(profit)={self.bootstrap.profit_probability:.1%})\n"
            f"  ├─ Consistency Score: {self.consistency_score:.1f}/100 "
            f"(Sharpe stability={self.jitter.stability_score:.1f}%)\n"
            f"  └─ Survival Score:    {self.survival_score:.1f}/100 "
            f"(P(ruin)={self.bootstrap.ruin_probability:.1%})\n"
            f"\n"
            f"  Bootstrap ({self.bootstrap.n_simulations} sims):\n"
            f"    Median final equity: ${self.bootstrap.median_final_equity:,.0f}\n"
            f"    5th-95th percentile: ${self.bootstrap.percentile_5:,.0f} — "
            f"${self.bootstrap.percentile_95:,.0f}\n"
            f"    Mean max drawdown: {self.bootstrap.mean_max_drawdown:.1%}\n"
            f"\n"
            f"  Parameter Jitter ({self.jitter.n_perturbations} perturbations):\n"
            f"    Original Sharpe: {self.jitter.original_sharpe:.2f}\n"
            f"    Mean perturbed Sharpe: {self.jitter.mean_perturbed_sharpe:.2f} "
            f"(±{self.jitter.std_perturbed_sharpe:.2f})\n"
            f"    Degradation: {self.jitter.sharpe_degradation_pct:.1f}%\n"
            f"\n"
            f"  Execution Degradation:\n"
            f"    Break-even slippage: {self.degradation.break_even_slippage_bps:.1f} bps\n"
            f"    Safety margin: {self.degradation.safety_margin_bps:.1f} bps\n"
            f"\n"
            f"  Trades analysed: {self.n_trades_analysed}"
        )


# ---------------------------------------------------------------------------
# Monte Carlo Validator
# ---------------------------------------------------------------------------

class MonteCarloValidator:
    """
    Multi-layered Monte Carlo robustness validator.
    
    Validates that a strategy's performance is not an artifact of:
    - Lucky trade ordering (bootstrap)
    - Fragile parameter optimisation (jitter)
    - Unrealistic execution assumptions (degradation)
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def validate(
        self,
        trades: List[float],
        config: Optional[MCConfig] = None,
        parameters: Optional[Dict[str, float]] = None,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        actual_slippage_bps: float = 5.0,
    ) -> MCResult:
        """
        Run full Monte Carlo validation on a list of trade P&L values.
        
        Args:
            trades: List of trade P&L values (absolute dollar amounts)
            config: Validation configuration
            parameters: Current strategy parameters (for jitter testing)
            parameter_ranges: Valid ranges for each parameter
            actual_slippage_bps: Current actual slippage in basis points
        
        Returns:
            MCResult with composite robustness score and detailed breakdowns
        """
        if config is None:
            config = MCConfig()
        
        if len(trades) < 10:
            logger.warning(f"Only {len(trades)} trades — results may be unreliable")
        
        trade_array = np.array(trades, dtype=float)
        
        # 1. Bootstrap resampling
        logger.info(f"Running bootstrap resampling ({config.n_bootstrap} simulations)...")
        bootstrap = self._bootstrap_resample(trade_array, config)
        
        # 2. Parameter jitter
        logger.info(f"Running parameter jitter ({config.n_jitter} perturbations)...")
        jitter = self._parameter_jitter(
            trade_array, config, parameters, parameter_ranges
        )
        
        # 3. Execution degradation
        logger.info(f"Running execution degradation ({config.slippage_steps} levels)...")
        degradation = self._execution_degradation(
            trade_array, config, actual_slippage_bps
        )
        
        # 4. Compute composite score
        profit_score = self._compute_profit_score(bootstrap, config)
        consistency_score = self._compute_consistency_score(jitter)
        survival_score = self._compute_survival_score(bootstrap)
        
        robustness_score = (
            config.weight_profit_prob * profit_score +
            config.weight_consistency * consistency_score +
            config.weight_survival * survival_score
        )
        
        passes_gate = (
            robustness_score >= config.min_robustness_score and
            bootstrap.profit_probability >= config.min_profit_probability
        )
        
        result = MCResult(
            robustness_score=round(robustness_score, 1),
            passes_gate=passes_gate,
            profit_score=round(profit_score, 1),
            consistency_score=round(consistency_score, 1),
            survival_score=round(survival_score, 1),
            bootstrap=bootstrap,
            jitter=jitter,
            degradation=degradation,
            n_trades_analysed=len(trades),
            config=config,
        )
        
        logger.info(f"Robustness validation complete: score={robustness_score:.1f}, "
                    f"gate={'PASS' if passes_gate else 'FAIL'}")
        
        return result
    
    # ─── Bootstrap Resampling ─────────────────────────────────────────────
    
    def _bootstrap_resample(
        self, trades: np.ndarray, config: MCConfig
    ) -> BootstrapResult:
        """
        Resample trade sequences with replacement to test ordering sensitivity.
        
        For each simulation:
        1. Randomly reorder trades (with replacement)
        2. Build equity curve from initial capital
        3. Track max drawdown and final equity
        """
        n_trades = len(trades)
        n_sims = config.n_bootstrap
        initial = config.initial_capital
        ruin_level = initial * (1 - config.ruin_threshold)
        
        final_equities = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        ruin_count = 0
        
        # Store sample equity curves for visualization (first 100)
        store_curves = min(100, n_sims)
        equity_curves = np.zeros((store_curves, n_trades + 1))
        
        for i in range(n_sims):
            # Resample trades with replacement
            resampled = self.rng.choice(trades, size=n_trades, replace=True)
            
            # Build equity curve
            equity = np.zeros(n_trades + 1)
            equity[0] = initial
            peak = initial
            worst_dd = 0.0
            hit_ruin = False
            
            for j, pnl in enumerate(resampled):
                equity[j + 1] = equity[j] + pnl
                if equity[j + 1] > peak:
                    peak = equity[j + 1]
                dd = (peak - equity[j + 1]) / peak if peak > 0 else 0
                if dd > worst_dd:
                    worst_dd = dd
                if equity[j + 1] <= ruin_level:
                    hit_ruin = True
            
            final_equities[i] = equity[-1]
            max_drawdowns[i] = worst_dd
            if hit_ruin:
                ruin_count += 1
            
            if i < store_curves:
                equity_curves[i] = equity
        
        profit_probability = float(np.mean(final_equities > initial))
        
        return BootstrapResult(
            n_simulations=n_sims,
            profit_probability=round(profit_probability, 4),
            median_final_equity=round(float(np.median(final_equities)), 2),
            percentile_5=round(float(np.percentile(final_equities, 5)), 2),
            percentile_25=round(float(np.percentile(final_equities, 25)), 2),
            percentile_75=round(float(np.percentile(final_equities, 75)), 2),
            percentile_95=round(float(np.percentile(final_equities, 95)), 2),
            mean_max_drawdown=round(float(np.mean(max_drawdowns)), 4),
            worst_drawdown=round(float(np.max(max_drawdowns)), 4),
            ruin_probability=round(ruin_count / n_sims, 4),
            equity_curves=equity_curves,
        )
    
    # ─── Parameter Jitter ─────────────────────────────────────────────────
    
    def _parameter_jitter(
        self,
        trades: np.ndarray,
        config: MCConfig,
        parameters: Optional[Dict[str, float]],
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]],
    ) -> JitterResult:
        """
        Test parameter sensitivity by perturbing trade P&L values.
        
        When actual parameters are not provided, we simulate parameter
        sensitivity by scaling trade magnitudes (a proxy for parameter
        changes affecting trade outcomes).
        """
        n_trades = len(trades)
        original_sharpe = self._compute_sharpe(trades)
        
        perturbed_sharpes = np.zeros(config.n_jitter)
        
        if parameters and parameter_ranges:
            # Full parameter jitter with actual parameters
            param_names = list(parameters.keys())
            n_params = len(param_names)
            sensitivities = {name: [] for name in param_names}
            
            for i in range(config.n_jitter):
                # Perturb each parameter independently
                scale_factors = np.ones(n_trades)
                for name in param_names:
                    original_val = parameters[name]
                    low, high = parameter_ranges[name]
                    jitter = self.rng.uniform(
                        -config.jitter_range, config.jitter_range
                    )
                    new_val = np.clip(
                        original_val * (1 + jitter), low, high
                    )
                    # Scale factor proportional to parameter change
                    param_scale = new_val / original_val if original_val != 0 else 1.0
                    scale_factors *= (0.5 + 0.5 * param_scale)
                
                perturbed_trades = trades * scale_factors
                perturbed_sharpes[i] = self._compute_sharpe(perturbed_trades)
            
            # Compute per-parameter sensitivity
            for name in param_names:
                param_sharpes = []
                for _ in range(50):
                    original_val = parameters[name]
                    low, high = parameter_ranges[name]
                    jitter = self.rng.uniform(-config.jitter_range, config.jitter_range)
                    new_val = np.clip(original_val * (1 + jitter), low, high)
                    scale = new_val / original_val if original_val != 0 else 1.0
                    param_sharpes.append(
                        self._compute_sharpe(trades * (0.5 + 0.5 * scale))
                    )
                sensitivities[name] = float(np.std(param_sharpes))
        else:
            # Proxy jitter: scale trade magnitudes randomly
            sensitivities = {}
            for i in range(config.n_jitter):
                # Random scaling of trade outcomes (simulates parameter changes)
                jitter_factors = 1.0 + self.rng.uniform(
                    -config.jitter_range, config.jitter_range, size=n_trades
                )
                perturbed_trades = trades * jitter_factors
                perturbed_sharpes[i] = self._compute_sharpe(perturbed_trades)
        
        mean_sharpe = float(np.mean(perturbed_sharpes))
        std_sharpe = float(np.std(perturbed_sharpes))
        
        # Degradation: how much worse is the average perturbed Sharpe
        degradation = 0.0
        if original_sharpe > 0:
            degradation = max(0, (original_sharpe - mean_sharpe) / original_sharpe * 100)
        
        # Stability score: 100 = no degradation, 0 = complete collapse
        stability = max(0, min(100, 100 - degradation))
        
        # % of perturbations still profitable
        pct_profitable = float(np.mean(perturbed_sharpes > 0))
        
        return JitterResult(
            n_perturbations=config.n_jitter,
            original_sharpe=round(original_sharpe, 4),
            mean_perturbed_sharpe=round(mean_sharpe, 4),
            std_perturbed_sharpe=round(std_sharpe, 4),
            sharpe_degradation_pct=round(degradation, 2),
            stability_score=round(stability, 2),
            pct_profitable=round(pct_profitable, 4),
            parameter_sensitivity=sensitivities,
        )
    
    # ─── Execution Degradation ────────────────────────────────────────────
    
    def _execution_degradation(
        self,
        trades: np.ndarray,
        config: MCConfig,
        actual_slippage_bps: float,
    ) -> DegradationResult:
        """
        Systematically increase transaction costs to find break-even point.
        
        Models slippage as a fixed cost per trade, increasing from 0 to
        max_slippage_bps in equal steps.
        """
        n_trades = len(trades)
        avg_trade_size = float(np.mean(np.abs(trades)))
        
        slippage_levels = np.linspace(0, config.max_slippage_bps, config.slippage_steps)
        sharpe_at_levels = []
        profit_at_levels = []
        break_even_bps = config.max_slippage_bps  # Default if never breaks even
        
        for bps in slippage_levels:
            # Cost per trade = avg_trade_size * bps / 10000
            cost_per_trade = avg_trade_size * bps / 10000.0
            degraded_trades = trades - cost_per_trade
            
            sharpe = self._compute_sharpe(degraded_trades)
            total_profit = float(np.sum(degraded_trades))
            
            sharpe_at_levels.append(round(sharpe, 4))
            profit_at_levels.append(round(total_profit, 2))
            
            # Find break-even (first level where total profit <= 0)
            if total_profit <= 0 and break_even_bps == config.max_slippage_bps:
                # Interpolate between this and previous level
                if len(profit_at_levels) >= 2:
                    prev_profit = profit_at_levels[-2]
                    prev_bps = slippage_levels[len(profit_at_levels) - 2]
                    if prev_profit > 0:
                        # Linear interpolation
                        ratio = prev_profit / (prev_profit - total_profit)
                        break_even_bps = prev_bps + ratio * (bps - prev_bps)
                    else:
                        break_even_bps = bps
                else:
                    break_even_bps = bps
        
        safety_margin = break_even_bps - actual_slippage_bps
        
        return DegradationResult(
            break_even_slippage_bps=round(break_even_bps, 2),
            slippage_levels=[round(s, 2) for s in slippage_levels.tolist()],
            sharpe_at_levels=sharpe_at_levels,
            profit_at_levels=profit_at_levels,
            safety_margin_bps=round(max(0, safety_margin), 2),
        )
    
    # ─── Scoring Functions ────────────────────────────────────────────────
    
    def _compute_profit_score(self, bootstrap: BootstrapResult, config: MCConfig) -> float:
        """
        Profit score (0-100) based on probability of profit and percentile spread.
        
        100 = P(profit) >= 95% with tight distribution
        0 = P(profit) < 50%
        """
        p_profit = bootstrap.profit_probability
        
        # Base score from profit probability (0-80)
        if p_profit >= 0.95:
            base = 80.0
        elif p_profit >= 0.70:
            base = 40.0 + (p_profit - 0.70) / 0.25 * 40.0
        elif p_profit >= 0.50:
            base = (p_profit - 0.50) / 0.20 * 40.0
        else:
            base = 0.0
        
        # Bonus for tight distribution (0-20)
        spread = bootstrap.percentile_95 - bootstrap.percentile_5
        median = bootstrap.median_final_equity
        if median > 0:
            cv = spread / median  # Coefficient of variation
            distribution_bonus = max(0, 20 * (1 - cv / 2))
        else:
            distribution_bonus = 0.0
        
        return min(100.0, base + distribution_bonus)
    
    def _compute_consistency_score(self, jitter: JitterResult) -> float:
        """
        Consistency score (0-100) based on parameter stability.
        
        100 = Sharpe barely changes with parameter perturbation
        0 = Sharpe collapses with small parameter changes
        """
        # Primary: stability score (already 0-100)
        primary = jitter.stability_score * 0.7
        
        # Secondary: % of perturbations still profitable
        secondary = jitter.pct_profitable * 100 * 0.3
        
        return min(100.0, primary + secondary)
    
    def _compute_survival_score(self, bootstrap: BootstrapResult) -> float:
        """
        Survival score (0-100) based on probability of avoiding ruin.
        
        100 = P(ruin) = 0% with low max drawdown
        0 = P(ruin) > 20%
        """
        p_ruin = bootstrap.ruin_probability
        mean_dd = bootstrap.mean_max_drawdown
        
        # Base score from ruin probability (0-70)
        if p_ruin == 0:
            base = 70.0
        elif p_ruin < 0.01:
            base = 60.0
        elif p_ruin < 0.05:
            base = 40.0
        elif p_ruin < 0.10:
            base = 20.0
        elif p_ruin < 0.20:
            base = 10.0
        else:
            base = 0.0
        
        # Bonus for low average drawdown (0-30)
        if mean_dd < 0.05:
            dd_bonus = 30.0
        elif mean_dd < 0.10:
            dd_bonus = 20.0
        elif mean_dd < 0.15:
            dd_bonus = 10.0
        elif mean_dd < 0.25:
            dd_bonus = 5.0
        else:
            dd_bonus = 0.0
        
        return min(100.0, base + dd_bonus)
    
    # ─── Utility Functions ────────────────────────────────────────────────
    
    @staticmethod
    def _compute_sharpe(trades: np.ndarray, annualisation: float = 252.0) -> float:
        """Compute annualised Sharpe ratio from trade P&L array."""
        if len(trades) < 2:
            return 0.0
        mean_return = float(np.mean(trades))
        std_return = float(np.std(trades, ddof=1))
        if std_return < 1e-9:
            return 0.0
        return mean_return / std_return * np.sqrt(annualisation)
