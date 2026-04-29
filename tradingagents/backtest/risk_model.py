"""Factor Risk Model.

Implements a Fama-French style factor risk model that decomposes portfolio risk into:
  - Market factor (beta to SPY/market)
  - Size factor (SMB proxy)
  - Value factor (HML proxy)
  - Momentum factor (MOM proxy)
  - Idiosyncratic / stock-specific risk

Also provides VaR and CVaR calculations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FactorExposures:
    """Factor exposures (betas) for a single asset or portfolio."""
    ticker: str
    market_beta: float
    size_beta: float = 0.0
    value_beta: float = 0.0
    momentum_beta: float = 0.0
    alpha: float = 0.0                 # Annualized alpha
    r_squared: float = 0.0
    residual_vol: float = 0.0          # Idiosyncratic volatility (annualized)
    total_vol: float = 0.0             # Total annualized volatility

    @property
    def systematic_risk_pct(self) -> float:
        """Fraction of total variance explained by factors."""
        return self.r_squared

    @property
    def idiosyncratic_risk_pct(self) -> float:
        return 1.0 - self.r_squared


@dataclass
class RiskDecomposition:
    """Portfolio-level risk decomposition."""
    portfolio_vol: float               # Total annualized portfolio volatility
    factor_vol: float                  # Volatility from systematic factors
    idiosyncratic_vol: float           # Volatility from stock-specific risk
    market_contribution: float         # Fraction from market factor
    size_contribution: float = 0.0
    value_contribution: float = 0.0
    momentum_contribution: float = 0.0
    var_95: float = 0.0                # 1-day 95% VaR (as fraction of portfolio)
    var_99: float = 0.0                # 1-day 99% VaR
    cvar_95: float = 0.0               # 1-day 95% CVaR (Expected Shortfall)
    asset_exposures: Dict[str, FactorExposures] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Risk Decomposition ===",
            f"Portfolio Vol (ann): {self.portfolio_vol*100:.2f}%",
            f"  Systematic:        {self.factor_vol*100:.2f}% ({self.market_contribution*100:.1f}% market)",
            f"  Idiosyncratic:     {self.idiosyncratic_vol*100:.2f}%",
            f"1-Day VaR (95%):     {self.var_95*100:.2f}%",
            f"1-Day VaR (99%):     {self.var_99*100:.2f}%",
            f"1-Day CVaR (95%):    {self.cvar_95*100:.2f}%",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factor Risk Model
# ---------------------------------------------------------------------------

class FactorRiskModel:
    """Fama-French style factor risk model.

    Usage:
        model = FactorRiskModel()
        exposures = model.compute_exposures(asset_returns, factor_returns)
        decomposition = model.decompose_portfolio(weights, asset_returns, factor_returns)
    """

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days

    def compute_exposures(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> FactorExposures:
        """Compute factor exposures for a single asset.

        Args:
            asset_returns: Daily returns for the asset.
            factor_returns: DataFrame with columns ['market', 'smb', 'hml', 'mom'].
                            All columns optional except 'market'.

        Returns:
            FactorExposures with betas, alpha, R², and residual vol.
        """
        ticker = asset_returns.name or "UNKNOWN"

        # Align and trim
        combined = pd.concat([asset_returns, factor_returns], axis=1).dropna()
        combined = combined.tail(self.lookback_days)

        if len(combined) < 30:
            return FactorExposures(
                ticker=str(ticker),
                market_beta=1.0,
                total_vol=float(asset_returns.std() * np.sqrt(252)) if len(asset_returns) > 1 else 0.0,
            )

        y = combined.iloc[:, 0].values
        factor_cols = [c for c in factor_returns.columns if c in combined.columns]
        X = combined[factor_cols].values

        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])

        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            return FactorExposures(ticker=str(ticker), market_beta=1.0)

        alpha_daily = coeffs[0]
        betas = coeffs[1:]

        # R-squared
        y_pred = X_with_const @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Residual volatility
        resid = y - y_pred
        residual_vol = float(np.std(resid) * np.sqrt(252))
        total_vol = float(np.std(y) * np.sqrt(252))

        # Map betas to named factors
        factor_map = {name: float(betas[i]) for i, name in enumerate(factor_cols)}

        return FactorExposures(
            ticker=str(ticker),
            market_beta=factor_map.get("market", 1.0),
            size_beta=factor_map.get("smb", 0.0),
            value_beta=factor_map.get("hml", 0.0),
            momentum_beta=factor_map.get("mom", 0.0),
            alpha=float(alpha_daily * 252),
            r_squared=float(np.clip(r_squared, 0, 1)),
            residual_vol=residual_vol,
            total_vol=total_vol,
        )

    def decompose_portfolio(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> RiskDecomposition:
        """Decompose portfolio risk into factor and idiosyncratic components.

        Args:
            weights: Dict of ticker -> portfolio weight.
            asset_returns: DataFrame of daily returns, columns=tickers.
            factor_returns: DataFrame with factor return columns.

        Returns:
            RiskDecomposition with full breakdown.
        """
        tickers = [t for t in weights if t in asset_returns.columns]
        if not tickers:
            return RiskDecomposition(
                portfolio_vol=0.0, factor_vol=0.0, idiosyncratic_vol=0.0,
                market_contribution=0.0
            )

        w = np.array([weights[t] for t in tickers])
        w = w / w.sum()  # Normalize

        ret_df = asset_returns[tickers].tail(self.lookback_days).dropna(how="all")
        factor_df = factor_returns.tail(self.lookback_days)

        # Compute portfolio daily returns
        port_returns = (ret_df * w).sum(axis=1)
        port_vol = float(port_returns.std() * np.sqrt(252))

        # Compute per-asset exposures
        asset_exposures = {}
        for ticker in tickers:
            if ticker in asset_returns.columns:
                asset_exposures[ticker] = self.compute_exposures(
                    asset_returns[ticker], factor_df
                )

        # Portfolio-level factor exposures (weighted average)
        port_market_beta = sum(weights.get(t, 0) * asset_exposures[t].market_beta for t in tickers)
        port_size_beta = sum(weights.get(t, 0) * asset_exposures[t].size_beta for t in tickers)
        port_value_beta = sum(weights.get(t, 0) * asset_exposures[t].value_beta for t in tickers)
        port_mom_beta = sum(weights.get(t, 0) * asset_exposures[t].momentum_beta for t in tickers)

        # Factor volatilities
        factor_vols = {}
        for col in factor_df.columns:
            factor_vols[col] = float(factor_df[col].std() * np.sqrt(252))

        market_vol = factor_vols.get("market", 0.16)  # Default 16% market vol
        smb_vol = factor_vols.get("smb", 0.10)
        hml_vol = factor_vols.get("hml", 0.10)
        mom_vol = factor_vols.get("mom", 0.15)

        # Factor variance contributions (assuming uncorrelated factors for simplicity)
        market_var = (port_market_beta * market_vol) ** 2
        size_var = (port_size_beta * smb_vol) ** 2
        value_var = (port_value_beta * hml_vol) ** 2
        mom_var = (port_mom_beta * mom_vol) ** 2
        factor_var = market_var + size_var + value_var + mom_var

        # Idiosyncratic variance (weighted sum of residual variances)
        idio_var = sum(
            (weights.get(t, 0) ** 2) * (asset_exposures[t].residual_vol ** 2)
            for t in tickers
        )

        total_var = port_vol ** 2
        factor_vol = float(np.sqrt(factor_var))
        idio_vol = float(np.sqrt(idio_var))

        # Contributions as fraction of total variance
        market_contribution = market_var / total_var if total_var > 0 else 0.0

        # VaR and CVaR from historical simulation
        var_95, var_99, cvar_95 = self._compute_var(port_returns)

        return RiskDecomposition(
            portfolio_vol=port_vol,
            factor_vol=factor_vol,
            idiosyncratic_vol=idio_vol,
            market_contribution=float(np.clip(market_contribution, 0, 1)),
            size_contribution=float(np.clip(size_var / total_var if total_var > 0 else 0, 0, 1)),
            value_contribution=float(np.clip(value_var / total_var if total_var > 0 else 0, 0, 1)),
            momentum_contribution=float(np.clip(mom_var / total_var if total_var > 0 else 0, 0, 1)),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            asset_exposures=asset_exposures,
        )

    def _compute_var(self, returns: pd.Series) -> Tuple[float, float, float]:
        """Compute 1-day VaR and CVaR via historical simulation."""
        if len(returns) < 20:
            return 0.0, 0.0, 0.0
        sorted_returns = np.sort(returns.dropna().values)
        n = len(sorted_returns)
        var_95 = float(-np.percentile(sorted_returns, 5))
        var_99 = float(-np.percentile(sorted_returns, 1))
        cutoff_95 = int(n * 0.05)
        cvar_95 = float(-sorted_returns[:max(cutoff_95, 1)].mean()) if cutoff_95 > 0 else var_95
        return var_95, var_99, cvar_95

    @staticmethod
    def build_proxy_factors(market_returns: pd.Series) -> pd.DataFrame:
        """Build proxy factor returns from market data.

        When Fama-French data is unavailable, construct proxies:
          - market: the market return itself
          - smb: rolling small-cap proxy (high vol = small, low vol = large)
          - hml: value proxy (negative momentum as value proxy)
          - mom: 12-1 month momentum

        Args:
            market_returns: Daily returns for the market index (e.g., SPY).

        Returns:
            DataFrame with columns ['market', 'smb', 'hml', 'mom'].
        """
        df = pd.DataFrame({"market": market_returns})

        # SMB proxy: negative of rolling 20-day realized vol (high vol ≈ small cap)
        df["smb"] = -df["market"].rolling(20).std().fillna(0) * 5

        # HML proxy: negative of 12-month momentum (value = anti-momentum)
        df["hml"] = -df["market"].rolling(252).mean().fillna(0) * 50

        # Momentum: 12-1 month momentum (skip last month)
        df["mom"] = df["market"].rolling(231).mean().shift(21).fillna(0) * 50

        return df.dropna()
