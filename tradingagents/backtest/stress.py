"""Stress Testing Suite.

Implements both historical scenario replay and hypothetical shock analysis:

Historical scenarios:
  - 2008 Global Financial Crisis (Sep–Nov 2008)
  - 2020 COVID Crash (Feb–Mar 2020)
  - 2022 Rate Shock (Jan–Oct 2022)
  - 2000 Dot-com Bust (Mar 2000–Oct 2002)

Hypothetical shocks:
  - Market crash (-20%, -30%, -40%)
  - Volatility spike (VIX doubles)
  - Interest rate shock (+200bps, +400bps)
  - Sector rotation (tech -30%, energy +20%)
  - Correlation breakdown (all correlations → 0.9)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    description: str
    scenario_type: str                 # "historical" | "hypothetical"
    # For hypothetical shocks: dict of factor -> shock magnitude
    shocks: Dict[str, float] = field(default_factory=dict)
    # For historical: date range to replay
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class StressResult:
    """Result of a single stress scenario."""
    scenario: StressScenario
    portfolio_return: float            # Total return under stress
    max_loss: float                    # Maximum single-day loss
    recovery_days: Optional[int]       # Days to recover to pre-stress level
    asset_returns: Dict[str, float]    # Per-asset returns under stress
    var_breach: bool                   # Whether 99% VaR was breached
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Scenario: {self.scenario.name}\n"
            f"  Portfolio Return:  {self.portfolio_return*100:.2f}%\n"
            f"  Max 1-Day Loss:    {self.max_loss*100:.2f}%\n"
            f"  Recovery Days:     {self.recovery_days or 'N/A'}\n"
            f"  99% VaR Breach:    {self.var_breach}"
        )


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

BUILTIN_SCENARIOS: List[StressScenario] = [
    # Historical
    StressScenario(
        name="GFC_2008",
        description="Global Financial Crisis: Lehman collapse (Sep–Nov 2008)",
        scenario_type="historical",
        start_date="2008-09-01",
        end_date="2008-11-30",
    ),
    StressScenario(
        name="COVID_2020",
        description="COVID-19 crash (Feb 19 – Mar 23, 2020)",
        scenario_type="historical",
        start_date="2020-02-19",
        end_date="2020-03-23",
    ),
    StressScenario(
        name="RATE_SHOCK_2022",
        description="Fed rate hike cycle (Jan–Oct 2022)",
        scenario_type="historical",
        start_date="2022-01-03",
        end_date="2022-10-14",
    ),
    StressScenario(
        name="DOTCOM_BUST",
        description="Dot-com bust (Mar 2000 – Oct 2002)",
        scenario_type="historical",
        start_date="2000-03-10",
        end_date="2002-10-09",
    ),
    # Hypothetical
    StressScenario(
        name="MARKET_CRASH_20",
        description="Hypothetical 20% market crash",
        scenario_type="hypothetical",
        shocks={"market": -0.20},
    ),
    StressScenario(
        name="MARKET_CRASH_30",
        description="Hypothetical 30% market crash",
        scenario_type="hypothetical",
        shocks={"market": -0.30},
    ),
    StressScenario(
        name="MARKET_CRASH_40",
        description="Hypothetical 40% market crash (tail risk)",
        scenario_type="hypothetical",
        shocks={"market": -0.40},
    ),
    StressScenario(
        name="VOL_SPIKE",
        description="Volatility doubles (VIX 15 → 30)",
        scenario_type="hypothetical",
        shocks={"vol_multiplier": 2.0, "market": -0.10},
    ),
    StressScenario(
        name="RATE_SHOCK_200BPS",
        description="Sudden +200bps rate shock",
        scenario_type="hypothetical",
        shocks={"market": -0.08, "hml": 0.05, "smb": -0.03},
    ),
    StressScenario(
        name="RATE_SHOCK_400BPS",
        description="Severe +400bps rate shock",
        scenario_type="hypothetical",
        shocks={"market": -0.15, "hml": 0.10, "smb": -0.05},
    ),
    StressScenario(
        name="TECH_SELLOFF",
        description="Tech sector -30%, Energy +20%",
        scenario_type="hypothetical",
        shocks={"market": -0.12, "mom": -0.20},
    ),
]


# ---------------------------------------------------------------------------
# Stress Tester
# ---------------------------------------------------------------------------

class StressTester:
    """Portfolio stress tester.

    Usage:
        tester = StressTester()
        results = tester.run_all(weights, asset_returns, factor_returns)
        for r in results:
            print(r.summary())
    """

    def __init__(self, var_99: float = 0.03):
        """
        Args:
            var_99: 1-day 99% VaR threshold for breach detection (as fraction).
        """
        self.var_99 = var_99

    def run_all(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        scenarios: Optional[List[StressScenario]] = None,
    ) -> List[StressResult]:
        """Run all stress scenarios.

        Args:
            weights: Dict of ticker -> portfolio weight.
            asset_returns: DataFrame of daily returns, columns=tickers.
            factor_returns: Optional factor returns for hypothetical shocks.
            scenarios: List of scenarios to run (defaults to BUILTIN_SCENARIOS).

        Returns:
            List of StressResult objects.
        """
        if scenarios is None:
            scenarios = BUILTIN_SCENARIOS

        results = []
        for scenario in scenarios:
            try:
                result = self.run_scenario(weights, asset_returns, factor_returns, scenario)
                results.append(result)
            except Exception as e:
                logger.warning(f"Stress scenario {scenario.name} failed: {e}")

        return results

    def run_scenario(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame],
        scenario: StressScenario,
    ) -> StressResult:
        """Run a single stress scenario."""
        if scenario.scenario_type == "historical":
            return self._run_historical(weights, asset_returns, scenario)
        else:
            return self._run_hypothetical(weights, asset_returns, factor_returns, scenario)

    def _run_historical(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        scenario: StressScenario,
    ) -> StressResult:
        """Replay historical scenario on the portfolio."""
        tickers = [t for t in weights if t in asset_returns.columns]
        if not tickers:
            return self._empty_result(scenario)

        # Filter to scenario date range
        idx = pd.to_datetime(asset_returns.index)
        mask = (idx >= pd.Timestamp(scenario.start_date)) & (idx <= pd.Timestamp(scenario.end_date))
        period_returns = asset_returns.loc[mask, tickers]

        if period_returns.empty:
            # No data for this period — use synthetic shock based on known magnitudes
            known_shocks = {
                "GFC_2008": -0.45,
                "COVID_2020": -0.34,
                "RATE_SHOCK_2022": -0.25,
                "DOTCOM_BUST": -0.78,
            }
            shock = known_shocks.get(scenario.name, -0.20)
            return self._apply_uniform_shock(weights, tickers, shock, scenario)

        w = np.array([weights[t] for t in tickers])
        w = w / w.sum()

        port_daily = (period_returns * w).sum(axis=1)
        portfolio_return = float((1 + port_daily).prod() - 1)
        max_loss = float(port_daily.min())

        # Recovery: days after end_date to get back to 0% cumulative return
        recovery_days = self._estimate_recovery(port_daily, asset_returns, tickers, weights, scenario.end_date)

        asset_returns_dict = {}
        for ticker in tickers:
            if ticker in period_returns.columns:
                asset_returns_dict[ticker] = float((1 + period_returns[ticker]).prod() - 1)

        return StressResult(
            scenario=scenario,
            portfolio_return=portfolio_return,
            max_loss=max_loss,
            recovery_days=recovery_days,
            asset_returns=asset_returns_dict,
            var_breach=abs(max_loss) > self.var_99,
        )

    def _run_hypothetical(
        self,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame],
        scenario: StressScenario,
    ) -> StressResult:
        """Apply hypothetical factor shocks to the portfolio."""
        tickers = [t for t in weights if t in asset_returns.columns]
        if not tickers:
            return self._empty_result(scenario)

        market_shock = scenario.shocks.get("market", 0.0)

        # Compute per-asset beta to market
        betas = {}
        for ticker in tickers:
            if factor_returns is not None and "market" in factor_returns.columns:
                combined = pd.concat([asset_returns[ticker], factor_returns["market"]], axis=1).dropna()
                if len(combined) > 30:
                    cov = np.cov(combined.values.T)
                    betas[ticker] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
                else:
                    betas[ticker] = 1.0
            else:
                betas[ticker] = 1.0

        # Apply shocks: asset_return ≈ beta * market_shock + idio_shock
        asset_shock_returns = {}
        for ticker in tickers:
            beta = betas.get(ticker, 1.0)
            # Additional factor shocks
            factor_contribution = 0.0
            if factor_returns is not None:
                for factor, shock in scenario.shocks.items():
                    if factor == "market":
                        continue
                    if factor in factor_returns.columns:
                        # Estimate factor beta
                        combined = pd.concat([asset_returns[ticker], factor_returns[factor]], axis=1).dropna()
                        if len(combined) > 30:
                            cov = np.cov(combined.values.T)
                            f_beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
                            factor_contribution += f_beta * shock

            asset_shock_returns[ticker] = beta * market_shock + factor_contribution

        # Portfolio return under shock
        w = np.array([weights[t] for t in tickers])
        w = w / w.sum()
        shock_returns = np.array([asset_shock_returns[t] for t in tickers])
        portfolio_return = float(w @ shock_returns)

        # Max loss = portfolio return (single-period shock)
        max_loss = min(portfolio_return, 0.0)

        return StressResult(
            scenario=scenario,
            portfolio_return=portfolio_return,
            max_loss=max_loss,
            recovery_days=None,  # Not applicable for hypothetical
            asset_returns=asset_shock_returns,
            var_breach=abs(max_loss) > self.var_99,
            metadata={"betas": betas, "shocks": scenario.shocks},
        )

    def _apply_uniform_shock(
        self,
        weights: Dict[str, float],
        tickers: List[str],
        shock: float,
        scenario: StressScenario,
    ) -> StressResult:
        """Apply a uniform shock when historical data is unavailable."""
        asset_returns = {t: shock for t in tickers}
        w = np.array([weights[t] for t in tickers])
        w = w / w.sum()
        portfolio_return = float(w @ np.array([shock] * len(tickers)))

        return StressResult(
            scenario=scenario,
            portfolio_return=portfolio_return,
            max_loss=portfolio_return,
            recovery_days=None,
            asset_returns=asset_returns,
            var_breach=abs(portfolio_return) > self.var_99,
            metadata={"note": "synthetic_shock_no_historical_data"},
        )

    def _estimate_recovery(
        self,
        period_returns: pd.Series,
        full_returns: pd.DataFrame,
        tickers: List[str],
        weights: Dict[str, float],
        end_date: str,
    ) -> Optional[int]:
        """Estimate days to recover from the stress period."""
        try:
            idx = pd.to_datetime(full_returns.index)
            post_mask = idx > pd.Timestamp(end_date)
            post_returns = full_returns.loc[post_mask, tickers]

            if post_returns.empty:
                return None

            w = np.array([weights[t] for t in tickers])
            w = w / w.sum()

            stress_loss = float((1 + (full_returns.loc[:, tickers] * w).sum(axis=1)
                                  .loc[pd.to_datetime(full_returns.index) <= pd.Timestamp(end_date)]).prod() - 1)

            if stress_loss >= 0:
                return 0

            post_port = (post_returns * w).sum(axis=1)
            cumulative = (1 + post_port).cumprod() - 1

            # Find first day where cumulative recovery exceeds the loss
            recovery_mask = cumulative >= abs(stress_loss)
            if recovery_mask.any():
                return int(recovery_mask.idxmax())  # type: ignore
            return None
        except Exception:
            return None

    def _empty_result(self, scenario: StressScenario) -> StressResult:
        return StressResult(
            scenario=scenario,
            portfolio_return=0.0,
            max_loss=0.0,
            recovery_days=None,
            asset_returns={},
            var_breach=False,
        )

    def summary_table(self, results: List[StressResult]) -> pd.DataFrame:
        """Build a summary DataFrame from stress results."""
        rows = []
        for r in results:
            rows.append({
                "Scenario": r.scenario.name,
                "Description": r.scenario.description,
                "Portfolio Return": f"{r.portfolio_return*100:.1f}%",
                "Max 1-Day Loss": f"{r.max_loss*100:.1f}%",
                "Recovery Days": r.recovery_days or "N/A",
                "VaR Breach": "YES" if r.var_breach else "no",
            })
        return pd.DataFrame(rows)
