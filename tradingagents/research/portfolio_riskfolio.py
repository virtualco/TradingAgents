"""Portfolio Optimisation via Riskfolio-Lib.

Implements three advanced portfolio construction methods:
1. Hierarchical Risk Parity (HRP) — cluster-based diversification
2. CVaR Optimisation — tail-risk aware allocation (minimise Conditional VaR)
3. Mean-CVaR — maximise return per unit of tail risk

All methods support:
  - Turnover constraints (max rebalance per period)
  - Long-only constraint
  - Max/min position size bounds
  - Covariance shrinkage (Ledoit-Wolf)

Integration:
  - Drop-in replacement for existing optimizer.py methods
  - Compatible with dynamic_portfolio.py rebalancing loop
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import riskfolio
_RISKFOLIO_AVAILABLE = None


def _check_riskfolio():
    global _RISKFOLIO_AVAILABLE
    if _RISKFOLIO_AVAILABLE is None:
        try:
            import riskfolio as rp  # noqa: F401
            _RISKFOLIO_AVAILABLE = True
        except ImportError:
            _RISKFOLIO_AVAILABLE = False
            logger.warning("riskfolio-lib not available: pip install riskfolio-lib")
    return _RISKFOLIO_AVAILABLE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskfolioConfig:
    """Configuration for Riskfolio-Lib portfolio optimisation."""
    method: Literal["HRP", "CVaR", "MeanCVaR"] = "HRP"
    confidence_level: float = 0.95        # CVaR confidence (95% = 5% tail)
    max_weight: float = 0.25              # Max weight per asset
    min_weight: float = 0.01              # Min weight per asset
    turnover_limit: Optional[float] = 0.30  # Max turnover per rebalance (None = no limit)
    lookback_days: int = 252              # Days of returns for estimation
    covariance_method: str = "ledoit"     # "ledoit" | "sample" | "shrunk"
    risk_free_rate: float = 0.05          # Annual risk-free rate
    hrp_linkage: str = "ward"             # Linkage for HRP clustering
    hrp_codependence: str = "pearson"     # Codependence metric for HRP
    allow_short: bool = False


@dataclass
class RiskfolioResult:
    """Result of Riskfolio-Lib optimisation."""
    weights: Dict[str, float]
    method: str
    expected_return: float
    expected_volatility: float
    expected_cvar: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n: float  # Effective number of assets (1/sum(w^2))
    turnover: float     # Turnover from previous weights
    
    @property
    def summary(self) -> str:
        top3 = sorted(self.weights.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{k}:{v:.1%}" for k, v in top3)
        return (
            f"[{self.method}] E[R]={self.expected_return:.2%}, "
            f"Vol={self.expected_volatility:.2%}, CVaR={self.expected_cvar:.2%}, "
            f"Sharpe={self.sharpe_ratio:.2f}, EffN={self.effective_n:.1f}, "
            f"Turnover={self.turnover:.2%} | Top: {top3_str}"
        )


# ---------------------------------------------------------------------------
# Main Optimiser
# ---------------------------------------------------------------------------

class RiskfolioOptimiser:
    """Advanced portfolio optimiser using Riskfolio-Lib.
    
    Provides HRP, CVaR, and Mean-CVaR optimisation with turnover constraints.
    """
    
    def __init__(self, config: Optional[RiskfolioConfig] = None):
        if not _check_riskfolio():
            raise ImportError("riskfolio-lib required: pip install riskfolio-lib")
        self.config = config or RiskfolioConfig()
        self._prev_weights: Optional[Dict[str, float]] = None
    
    def optimise(
        self,
        returns: pd.DataFrame,
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> RiskfolioResult:
        """Run portfolio optimisation.
        
        Args:
            returns: DataFrame of asset returns (columns = tickers, rows = periods).
                     Should be daily or hourly returns.
            prev_weights: Previous period weights for turnover calculation.
            
        Returns:
            RiskfolioResult with optimal weights and metrics.
        """
        import riskfolio as rp
        
        cfg = self.config
        
        if prev_weights is not None:
            self._prev_weights = prev_weights
        
        # Trim to lookback
        if len(returns) > cfg.lookback_days:
            returns = returns.iloc[-cfg.lookback_days:]
        
        tickers = list(returns.columns)
        
        if cfg.method == "HRP":
            weights = self._optimise_hrp(returns, cfg)
        elif cfg.method == "CVaR":
            weights = self._optimise_cvar(returns, cfg)
        elif cfg.method == "MeanCVaR":
            weights = self._optimise_mean_cvar(returns, cfg)
        else:
            raise ValueError(f"Unknown method: {cfg.method}")
        
        # Apply bounds
        weights = self._apply_bounds(weights, tickers, cfg)
        
        # Apply turnover constraint
        if cfg.turnover_limit and self._prev_weights:
            weights = self._apply_turnover_constraint(weights, tickers, cfg)
        
        # Calculate metrics
        w_arr = np.array([weights.get(t, 0.0) for t in tickers])
        mu = returns.mean() * 252  # annualised
        cov = returns.cov() * 252
        
        exp_ret = float(w_arr @ mu.values)
        exp_vol = float(np.sqrt(w_arr @ cov.values @ w_arr))
        sharpe = (exp_ret - cfg.risk_free_rate) / max(exp_vol, 1e-8)
        
        # CVaR calculation
        portfolio_returns = (returns.values @ w_arr)
        var_threshold = np.percentile(portfolio_returns, (1 - cfg.confidence_level) * 100)
        cvar = -float(np.mean(portfolio_returns[portfolio_returns <= var_threshold])) * np.sqrt(252)
        
        # Diversification metrics
        effective_n = 1.0 / max(np.sum(w_arr ** 2), 1e-8)
        
        # Individual vols
        asset_vols = np.sqrt(np.diag(cov.values))
        div_ratio = float(w_arr @ asset_vols) / max(exp_vol, 1e-8)
        
        # Turnover
        turnover = 0.0
        if self._prev_weights:
            for t in tickers:
                turnover += abs(weights.get(t, 0.0) - self._prev_weights.get(t, 0.0))
            turnover /= 2  # one-way turnover
        
        # Update prev weights
        self._prev_weights = weights
        
        return RiskfolioResult(
            weights=weights,
            method=cfg.method,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_cvar=cvar,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            turnover=turnover,
        )
    
    def _optimise_hrp(self, returns: pd.DataFrame, cfg: RiskfolioConfig) -> Dict[str, float]:
        """Hierarchical Risk Parity optimisation."""
        import riskfolio as rp
        
        port = rp.HCPortfolio(returns=returns)
        
        w = port.optimization(
            model="HRP",
            codependence=cfg.hrp_codependence,
            rm="MV",  # risk measure for leaf allocation
            rf=cfg.risk_free_rate / 252,
            linkage=cfg.hrp_linkage,
            leaf_order=True,
        )
        
        if w is None or w.empty:
            # Fallback to equal weight
            n = len(returns.columns)
            return {t: 1.0 / n for t in returns.columns}
        
        return {t: float(w.loc[t, "weights"]) for t in returns.columns if t in w.index}
    
    def _optimise_cvar(self, returns: pd.DataFrame, cfg: RiskfolioConfig) -> Dict[str, float]:
        """Minimum CVaR optimisation."""
        import riskfolio as rp
        
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov=cfg.covariance_method)
        port.alpha = 1 - cfg.confidence_level  # Set confidence level as property
        
        w = port.optimization(
            model="Classic",
            rm="CVaR",
            obj="MinRisk",
            rf=cfg.risk_free_rate / 252,
            hist=True,
        )
        
        if w is None or w.empty:
            n = len(returns.columns)
            return {t: 1.0 / n for t in returns.columns}
        
        return {t: float(w.loc[t, "weights"]) for t in returns.columns if t in w.index}
    
    def _optimise_mean_cvar(self, returns: pd.DataFrame, cfg: RiskfolioConfig) -> Dict[str, float]:
        """Mean-CVaR optimisation (maximise return per unit CVaR)."""
        import riskfolio as rp
        
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov=cfg.covariance_method)
        port.alpha = 1 - cfg.confidence_level  # Set confidence level as property
        
        w = port.optimization(
            model="Classic",
            rm="CVaR",
            obj="Sharpe",
            rf=cfg.risk_free_rate / 252,
            hist=True,
        )
        
        if w is None or w.empty:
            n = len(returns.columns)
            return {t: 1.0 / n for t in returns.columns}
        
        return {t: float(w.loc[t, "weights"]) for t in returns.columns if t in w.index}
    
    def _apply_bounds(
        self, weights: Dict[str, float], tickers: List[str], cfg: RiskfolioConfig
    ) -> Dict[str, float]:
        """Apply min/max weight bounds and renormalise."""
        bounded = {}
        for t in tickers:
            w = weights.get(t, 0.0)
            if w < cfg.min_weight:
                w = 0.0  # Below minimum → zero out
            elif w > cfg.max_weight:
                w = cfg.max_weight
            bounded[t] = w
        
        # Renormalise to sum to 1
        total = sum(bounded.values())
        if total > 0:
            bounded = {t: w / total for t, w in bounded.items()}
        else:
            # Fallback to equal weight
            n = len(tickers)
            bounded = {t: 1.0 / n for t in tickers}
        
        return bounded
    
    def _apply_turnover_constraint(
        self, weights: Dict[str, float], tickers: List[str], cfg: RiskfolioConfig
    ) -> Dict[str, float]:
        """Constrain turnover by blending towards previous weights."""
        if not self._prev_weights or not cfg.turnover_limit:
            return weights
        
        # Calculate current turnover
        turnover = sum(
            abs(weights.get(t, 0.0) - self._prev_weights.get(t, 0.0))
            for t in tickers
        ) / 2
        
        if turnover <= cfg.turnover_limit:
            return weights
        
        # Blend towards previous weights to reduce turnover
        blend_factor = cfg.turnover_limit / max(turnover, 1e-8)
        blended = {}
        for t in tickers:
            new_w = weights.get(t, 0.0)
            old_w = self._prev_weights.get(t, 0.0)
            blended[t] = old_w + blend_factor * (new_w - old_w)
        
        # Renormalise
        total = sum(blended.values())
        if total > 0:
            blended = {t: w / total for t, w in blended.items()}
        
        return blended
    
    def compare_methods(
        self,
        returns: pd.DataFrame,
        methods: Optional[List[str]] = None,
    ) -> List[RiskfolioResult]:
        """Compare multiple optimisation methods on the same data.
        
        Returns list of RiskfolioResult, one per method.
        """
        if methods is None:
            methods = ["HRP", "CVaR", "MeanCVaR"]
        
        results = []
        for method in methods:
            cfg = RiskfolioConfig(
                method=method,
                max_weight=self.config.max_weight,
                min_weight=self.config.min_weight,
                turnover_limit=self.config.turnover_limit,
                lookback_days=self.config.lookback_days,
                confidence_level=self.config.confidence_level,
                risk_free_rate=self.config.risk_free_rate,
            )
            opt = RiskfolioOptimiser(config=cfg)
            result = opt.optimise(returns, prev_weights=self._prev_weights)
            results.append(result)
        
        return results
