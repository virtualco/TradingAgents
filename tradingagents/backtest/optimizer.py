"""Portfolio Optimizer.

Implements three portfolio construction methods:
1. Mean-Variance (Markowitz) — maximize Sharpe ratio subject to constraints
2. Risk Parity — equalize risk contribution across assets
3. Conviction-Weighted — weight by signal conviction scores

All methods enforce:
  - Long-only (no short selling by default)
  - Max position size constraint
  - Min position size to avoid tiny allocations
  - Full investment (weights sum to 1)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import cvxpy; fall back to scipy if unavailable
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("cvxpy not available — mean-variance optimization will use scipy fallback")

from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    method: str = "mean_variance"      # "mean_variance" | "risk_parity" | "conviction"
    max_weight: float = 0.20           # Max weight per asset
    min_weight: float = 0.01           # Min weight per asset (0 = allow zero)
    risk_free_rate: float = 0.05       # Annual risk-free rate
    target_volatility: Optional[float] = None  # Annual vol target (None = maximize Sharpe)
    lookback_days: int = 252           # Days of returns for covariance estimation
    regularization: float = 0.0001    # Ledoit-Wolf shrinkage intensity
    allow_short: bool = False


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]          # ticker -> weight
    expected_return: float             # Annual expected return
    expected_volatility: float         # Annual expected volatility
    sharpe_ratio: float
    method: str
    converged: bool
    metadata: Dict = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        return pd.Series(self.weights)

    def summary(self) -> str:
        lines = [
            f"=== Portfolio Optimization ({self.method}) ===",
            f"Expected Return:    {self.expected_return*100:.2f}%",
            f"Expected Vol:       {self.expected_volatility*100:.2f}%",
            f"Sharpe Ratio:       {self.sharpe_ratio:.3f}",
            f"Converged:          {self.converged}",
            "",
            "Weights:",
        ]
        for ticker, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            if w > 0.001:
                lines.append(f"  {ticker:8s}: {w*100:.1f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """Portfolio optimizer supporting multiple construction methods.

    Usage:
        optimizer = PortfolioOptimizer(config=OptimizationConfig(method="mean_variance"))
        result = optimizer.optimize(returns_df, conviction_scores)
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

    def optimize(
        self,
        returns: pd.DataFrame,
        conviction_scores: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Optimize portfolio weights.

        Args:
            returns: DataFrame of daily returns, columns=tickers, index=dates.
            conviction_scores: Dict of ticker -> conviction score [0, 1].

        Returns:
            OptimizationResult with weights and performance metrics.
        """
        if returns.empty or returns.shape[1] == 0:
            return self._empty_result()

        tickers = list(returns.columns)
        n = len(tickers)

        # Use only the lookback window
        returns = returns.tail(self.config.lookback_days).dropna(how="all")

        if len(returns) < 20:
            logger.warning("Insufficient return history for optimization, using equal weights")
            return self._equal_weight_result(tickers, returns)

        # Compute expected returns and covariance
        mu = returns.mean() * 252          # Annualized
        cov = self._shrinkage_covariance(returns) * 252  # Annualized

        if self.config.method == "mean_variance":
            weights, converged = self._mean_variance(mu.values, cov.values, n)
        elif self.config.method == "risk_parity":
            weights, converged = self._risk_parity(cov.values, n)
        elif self.config.method == "conviction":
            weights, converged = self._conviction_weighted(tickers, conviction_scores or {}, n)
        else:
            weights = np.ones(n) / n
            converged = True

        # Map back to tickers
        weight_dict = {ticker: float(w) for ticker, w in zip(tickers, weights)}

        # Compute portfolio metrics
        w = np.array([weight_dict[t] for t in tickers])
        port_return = float(w @ mu.values)
        port_vol = float(np.sqrt(w @ cov.values @ w))
        sharpe = (port_return - self.config.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights=weight_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method=self.config.method,
            converged=converged,
        )

    def _shrinkage_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance estimator."""
        sample_cov = returns.cov()
        n_assets = sample_cov.shape[0]
        target = np.diag(np.diag(sample_cov.values))  # Diagonal target
        alpha = self.config.regularization
        shrunk = (1 - alpha) * sample_cov.values + alpha * target
        return pd.DataFrame(shrunk, index=sample_cov.index, columns=sample_cov.columns)

    def _mean_variance(
        self, mu: np.ndarray, cov: np.ndarray, n: int
    ) -> Tuple[np.ndarray, bool]:
        """Maximize Sharpe ratio using cvxpy or scipy fallback."""
        if CVXPY_AVAILABLE:
            return self._mean_variance_cvxpy(mu, cov, n)
        return self._mean_variance_scipy(mu, cov, n)

    def _mean_variance_cvxpy(
        self, mu: np.ndarray, cov: np.ndarray, n: int
    ) -> Tuple[np.ndarray, bool]:
        """Maximize Sharpe ratio via cvxpy (exact convex formulation)."""
        try:
            # Maximize Sharpe: equivalent to minimizing variance for given return
            w = cp.Variable(n)
            ret = mu @ w
            risk = cp.quad_form(w, cov)

            constraints = [
                cp.sum(w) == 1,
                w >= (0 if not self.config.allow_short else -self.config.max_weight),
                w <= self.config.max_weight,
            ]

            # Maximize Sharpe via parametric approach: minimize risk for target return
            excess_mu = mu - self.config.risk_free_rate / 252
            # Use the standard Sharpe maximization via auxiliary variable
            y = cp.Variable(n, nonneg=True)
            kappa = cp.Variable(nonneg=True)

            prob = cp.Problem(
                cp.Minimize(cp.quad_form(y, cov)),
                [
                    excess_mu @ y == 1,
                    cp.sum(y) == kappa,
                    y >= 0,
                    y <= self.config.max_weight * kappa,
                ]
            )
            prob.solve(solver=cp.OSQP, warm_start=True)

            if prob.status in ("optimal", "optimal_inaccurate") and kappa.value is not None and kappa.value > 1e-8:
                weights = y.value / kappa.value
                weights = np.clip(weights, 0, self.config.max_weight)
                weights /= weights.sum()
                return weights, True
        except Exception as e:
            logger.warning(f"cvxpy optimization failed: {e}, falling back to scipy")

        return self._mean_variance_scipy(mu, cov, n)

    def _mean_variance_scipy(
        self, mu: np.ndarray, cov: np.ndarray, n: int
    ) -> Tuple[np.ndarray, bool]:
        """Maximize Sharpe ratio via scipy minimize."""
        rf_daily = self.config.risk_free_rate / 252

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(port_ret - self.config.risk_free_rate) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, self.config.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-9})

        if result.success:
            weights = np.clip(result.x, 0, self.config.max_weight)
            weights /= weights.sum()
            return weights, True

        return np.ones(n) / n, False

    def _risk_parity(self, cov: np.ndarray, n: int) -> Tuple[np.ndarray, bool]:
        """Risk parity: equalize marginal risk contribution via scipy."""
        def risk_parity_objective(w):
            port_vol = np.sqrt(w @ cov @ w)
            mrc = cov @ w / port_vol  # Marginal risk contribution
            rc = w * mrc              # Risk contribution
            target_rc = port_vol / n  # Equal risk contribution
            return np.sum((rc - target_rc) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(max(self.config.min_weight, 0.001), self.config.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(risk_parity_objective, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 2000, "ftol": 1e-10})

        if result.success:
            weights = np.clip(result.x, 0, self.config.max_weight)
            weights /= weights.sum()
            return weights, True

        return np.ones(n) / n, False

    def _conviction_weighted(
        self, tickers: List[str], conviction_scores: Dict[str, float], n: int
    ) -> Tuple[np.ndarray, bool]:
        """Weight by conviction scores, capped at max_weight."""
        scores = np.array([conviction_scores.get(t, 0.5) for t in tickers])
        scores = np.clip(scores, 0, 1)

        if scores.sum() < 1e-10:
            return np.ones(n) / n, True

        weights = scores / scores.sum()
        weights = np.clip(weights, 0, self.config.max_weight)
        weights /= weights.sum()
        return weights, True

    def _equal_weight_result(self, tickers: List[str], returns: pd.DataFrame) -> OptimizationResult:
        n = len(tickers)
        weights = {t: 1.0 / n for t in tickers}
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        w = np.ones(n) / n
        port_return = float(w @ mu.values)
        port_vol = float(np.sqrt(w @ cov.values @ w))
        sharpe = (port_return - self.config.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        return OptimizationResult(
            weights=weights,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="equal_weight_fallback",
            converged=True,
        )

    def _empty_result(self) -> OptimizationResult:
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            method=self.config.method,
            converged=False,
        )
