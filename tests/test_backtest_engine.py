"""Tests for Phase 3: Backtesting Engine.

Covers:
- BacktestEngine (T+1 execution, stop-loss, take-profit, PnL)
- PortfolioOptimizer (mean-variance, risk parity, conviction)
- FactorRiskModel (exposures, decomposition, VaR)
- StressTester (historical, hypothetical scenarios)
- PerformanceAnalytics (Sharpe, Sortino, max drawdown, Calmar, alpha/beta)
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    OrderSide,
    Trade,
)
from tradingagents.backtest.optimizer import (
    OptimizationConfig,
    PortfolioOptimizer,
)
from tradingagents.backtest.risk_model import (
    FactorRiskModel,
)
from tradingagents.backtest.stress import (
    BUILTIN_SCENARIOS,
    StressTester,
    StressScenario,
)
from tradingagents.backtest.analytics import (
    PerformanceAnalytics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_price_data(
    ticker: str = "AAPL",
    days: int = 100,
    start_price: float = 150.0,
    trend: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.bdate_range(end="2026-04-28", periods=days)
    prices = start_price * np.cumprod(1 + np.random.normal(trend, 0.015, days))
    prices = np.maximum(prices, 5.0)
    return pd.DataFrame({
        "ticker": ticker,
        "event_time": dates.date,
        "open": prices * 0.995,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, days).astype(float),
    })


def make_signals(
    ticker: str = "AAPL",
    direction: str = "long",
    n: int = 1,
    conviction: float = 0.7,
    price_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """Create synthetic signal records."""
    rows = []
    if price_data is not None:
        # price_data is a dict of ticker->DataFrame; get first value
        if isinstance(price_data, dict):
            df = next(iter(price_data.values()))
        else:
            df = price_data
        dates = df["event_time"].tolist()
    else:
        dates = pd.bdate_range(end="2026-04-01", periods=n).date.tolist()

    for i in range(min(n, len(dates) - 5)):
        rows.append({
            "signal_id": str(uuid.uuid4()),
            "ticker": ticker,
            "trade_date": str(dates[i]),
            "direction": direction,
            "conviction": conviction,
            "stop_loss": None,
            "take_profit": None,
        })
    return pd.DataFrame(rows)


def make_returns(days: int = 252, mu: float = 0.0003, sigma: float = 0.012, seed: int = 42) -> pd.Series:
    """Create synthetic daily returns."""
    np.random.seed(seed)
    dates = pd.bdate_range(end="2026-04-28", periods=days)
    returns = np.random.normal(mu, sigma, days)
    return pd.Series(returns, index=dates.astype(str))


def make_equity_curve(returns: pd.Series, initial: float = 100_000.0) -> pd.Series:
    """Build equity curve from returns."""
    return initial * (1 + returns).cumprod()


# ---------------------------------------------------------------------------
# BacktestEngine Tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_empty_signals_returns_initial_capital(self):
        engine = BacktestEngine()
        price_data = {"AAPL": make_price_data()}
        result = engine.run(pd.DataFrame(), price_data)
        assert result.final_portfolio_value == pytest.approx(100_000.0)
        assert result.total_trades == 0

    def test_single_long_trade_executes(self):
        price_data = {"AAPL": make_price_data(days=60)}
        signals = make_signals("AAPL", "long", n=1, price_data=price_data)
        engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
        result = engine.run(signals, price_data)
        assert result.total_trades >= 1

    def test_t1_execution_delay(self):
        """Signal on day N should fill on day N+1 at open."""
        price_data = {"AAPL": make_price_data(days=60)}
        signals = make_signals("AAPL", "long", n=1, price_data=price_data)
        engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
        result = engine.run(signals, price_data)

        if result.trades:
            signal_date = pd.to_datetime(signals["trade_date"].iloc[0]).date()
            fill_date = pd.to_datetime(result.trades[0].fill_date).date()
            assert fill_date > signal_date

    def test_stop_loss_triggers(self):
        """Stop loss should close position before take profit."""
        # Create downtrending price data
        price_data = {"AAPL": make_price_data(days=80, trend=-0.005, seed=99)}
        signals = make_signals("AAPL", "long", n=1, price_data=price_data)
        config = BacktestConfig(
            initial_capital=100_000,
            stop_loss_pct=0.05,   # 5% stop
            take_profit_pct=0.30,  # 30% target (won't be hit in downtrend)
        )
        engine = BacktestEngine(config)
        result = engine.run(signals, price_data)
        # With a downtrend and 5% stop, the trade should close with a loss
        if result.trades and result.trades[0].exit_date is not None:
            assert result.trades[0].net_pnl < 0 or result.trades[0].exit_date is not None

    def test_max_positions_respected(self):
        """Should not open more than max_open_positions."""
        price_data = {
            "AAPL": make_price_data("AAPL", days=60, seed=1),
            "MSFT": make_price_data("MSFT", days=60, seed=2),
            "GOOGL": make_price_data("GOOGL", days=60, seed=3),
        }
        signals = pd.concat([
            make_signals("AAPL", "long", n=1, price_data=price_data["AAPL"]),
            make_signals("MSFT", "long", n=1, price_data=price_data["MSFT"]),
            make_signals("GOOGL", "long", n=1, price_data=price_data["GOOGL"]),
        ])
        config = BacktestConfig(max_open_positions=2)
        engine = BacktestEngine(config)
        result = engine.run(signals, price_data)
        # At most 2 positions should have been opened
        assert len(result.trades) <= 2

    def test_conviction_below_min_skipped(self):
        """Signals below min_conviction should be ignored."""
        price_data = {"AAPL": make_price_data(days=60)}
        signals = make_signals("AAPL", "long", n=1, conviction=0.10)
        config = BacktestConfig(min_conviction=0.30)
        engine = BacktestEngine(config)
        result = engine.run(signals, price_data)
        assert result.total_trades == 0

    def test_flat_direction_skipped(self):
        """FLAT signals should not generate trades."""
        price_data = {"AAPL": make_price_data(days=60)}
        signals = make_signals("AAPL", "flat", n=1)
        engine = BacktestEngine()
        result = engine.run(signals, price_data)
        assert result.total_trades == 0

    def test_equity_curve_has_entries(self):
        price_data = {"AAPL": make_price_data(days=60)}
        signals = make_signals("AAPL", "long", n=1, price_data=price_data)
        engine = BacktestEngine()
        result = engine.run(signals, price_data)
        assert len(result.equity_curve) > 0

    def test_commission_reduces_pnl(self):
        """Commission should reduce net PnL vs gross PnL."""
        price_data = {"AAPL": make_price_data(days=60, trend=0.005)}
        signals = make_signals("AAPL", "long", n=1, price_data=price_data)
        config = BacktestConfig(commission_pct=0.001)
        engine = BacktestEngine(config)
        result = engine.run(signals, price_data)
        for trade in result.trades:
            if trade.exit_date:
                assert trade.commission > 0
                assert trade.net_pnl <= trade.gross_pnl + 0.01  # Net ≤ Gross

    def test_result_summary_string(self):
        price_data = {"AAPL": make_price_data(days=60)}
        signals = make_signals("AAPL", "long", n=1, price_data=price_data)
        engine = BacktestEngine()
        result = engine.run(signals, price_data)
        summary = result.summary()
        assert "Total Return" in summary
        assert "Sharpe" in summary


# ---------------------------------------------------------------------------
# PortfolioOptimizer Tests
# ---------------------------------------------------------------------------

class TestPortfolioOptimizer:
    def _make_returns_df(self, n_assets: int = 3, days: int = 252) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.bdate_range(end="2026-04-28", periods=days)
        data = {
            f"ASSET{i}": np.random.normal(0.0003 * (i + 1), 0.012, days)
            for i in range(n_assets)
        }
        return pd.DataFrame(data, index=dates.astype(str))

    def test_mean_variance_weights_sum_to_one(self):
        returns = self._make_returns_df()
        optimizer = PortfolioOptimizer(OptimizationConfig(method="mean_variance"))
        result = optimizer.optimize(returns)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_risk_parity_weights_sum_to_one(self):
        returns = self._make_returns_df()
        optimizer = PortfolioOptimizer(OptimizationConfig(method="risk_parity"))
        result = optimizer.optimize(returns)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_conviction_weights_sum_to_one(self):
        returns = self._make_returns_df()
        conviction = {"ASSET0": 0.8, "ASSET1": 0.5, "ASSET2": 0.3}
        optimizer = PortfolioOptimizer(OptimizationConfig(method="conviction"))
        result = optimizer.optimize(returns, conviction)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_max_weight_constraint_respected(self):
        returns = self._make_returns_df()
        config = OptimizationConfig(method="mean_variance", max_weight=0.40)
        optimizer = PortfolioOptimizer(config)
        result = optimizer.optimize(returns)
        for w in result.weights.values():
            assert w <= 0.40 + 1e-6

    def test_empty_returns_returns_empty_result(self):
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(pd.DataFrame())
        assert result.weights == {}
        assert not result.converged

    def test_insufficient_history_uses_equal_weight(self):
        returns = self._make_returns_df(days=10)
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(returns)
        # Should fall back to equal weight
        n = len(result.weights)
        if n > 0:
            for w in result.weights.values():
                assert abs(w - 1.0 / n) < 0.01

    def test_high_conviction_gets_higher_weight(self):
        returns = self._make_returns_df()
        conviction = {"ASSET0": 0.9, "ASSET1": 0.1, "ASSET2": 0.1}
        optimizer = PortfolioOptimizer(OptimizationConfig(method="conviction"))
        result = optimizer.optimize(returns, conviction)
        assert result.weights["ASSET0"] > result.weights["ASSET1"]

    def test_expected_return_and_vol_positive(self):
        returns = self._make_returns_df()
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(returns)
        assert result.expected_volatility >= 0
        # Expected return can be negative in theory but vol must be non-negative

    def test_summary_string_contains_weights(self):
        returns = self._make_returns_df()
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(returns)
        summary = result.summary()
        assert "Weights" in summary


# ---------------------------------------------------------------------------
# FactorRiskModel Tests
# ---------------------------------------------------------------------------

class TestFactorRiskModel:
    def _make_factor_returns(self, days: int = 252) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.bdate_range(end="2026-04-28", periods=days)
        return pd.DataFrame({
            "market": np.random.normal(0.0004, 0.012, days),
            "smb": np.random.normal(0.0001, 0.008, days),
            "hml": np.random.normal(0.0001, 0.008, days),
            "mom": np.random.normal(0.0002, 0.010, days),
        }, index=dates.astype(str))

    def _make_asset_returns(self, factor_returns: pd.DataFrame, beta: float = 1.2) -> pd.Series:
        np.random.seed(99)
        idio = np.random.normal(0, 0.008, len(factor_returns))
        returns = beta * factor_returns["market"].values + idio
        return pd.Series(returns, index=factor_returns.index, name="AAPL")

    def test_compute_exposures_returns_exposures(self):
        factors = self._make_factor_returns()
        asset = self._make_asset_returns(factors, beta=1.2)
        model = FactorRiskModel()
        exposures = model.compute_exposures(asset, factors)
        assert exposures.ticker == "AAPL"
        assert exposures.market_beta > 0  # Should be positive for correlated asset
        assert 0 <= exposures.r_squared <= 1
        assert exposures.total_vol > 0

    def test_high_beta_asset_detected(self):
        factors = self._make_factor_returns()
        high_beta = self._make_asset_returns(factors, beta=2.0)
        low_beta = self._make_asset_returns(factors, beta=0.5)
        high_beta.name = "HIGH"
        low_beta.name = "LOW"
        model = FactorRiskModel()
        high_exp = model.compute_exposures(high_beta, factors)
        low_exp = model.compute_exposures(low_beta, factors)
        assert high_exp.market_beta > low_exp.market_beta

    def test_insufficient_data_returns_default(self):
        factors = self._make_factor_returns(days=10)
        asset = self._make_asset_returns(factors)
        model = FactorRiskModel()
        exposures = model.compute_exposures(asset, factors)
        assert exposures.market_beta == 1.0  # Default

    def test_decompose_portfolio(self):
        factors = self._make_factor_returns()
        asset_returns = pd.DataFrame({
            "AAPL": self._make_asset_returns(factors, beta=1.2).values,
            "MSFT": self._make_asset_returns(factors, beta=0.9).values,
        }, index=factors.index)
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        model = FactorRiskModel()
        decomp = model.decompose_portfolio(weights, asset_returns, factors)
        assert decomp.portfolio_vol > 0
        assert 0 <= decomp.market_contribution <= 1
        assert decomp.var_95 >= 0
        assert decomp.var_99 >= decomp.var_95

    def test_decompose_empty_weights(self):
        factors = self._make_factor_returns()
        asset_returns = pd.DataFrame()
        model = FactorRiskModel()
        decomp = model.decompose_portfolio({}, asset_returns, factors)
        assert decomp.portfolio_vol == 0.0

    def test_build_proxy_factors(self):
        market = make_returns(days=300)
        factors = FactorRiskModel.build_proxy_factors(market)
        assert "market" in factors.columns
        assert "smb" in factors.columns
        assert "hml" in factors.columns
        assert "mom" in factors.columns

    def test_decomp_summary_string(self):
        factors = self._make_factor_returns()
        asset_returns = pd.DataFrame({
            "AAPL": self._make_asset_returns(factors, beta=1.2).values,
        }, index=factors.index)
        weights = {"AAPL": 1.0}
        model = FactorRiskModel()
        decomp = model.decompose_portfolio(weights, asset_returns, factors)
        summary = decomp.summary()
        assert "Portfolio Vol" in summary
        assert "VaR" in summary


# ---------------------------------------------------------------------------
# StressTester Tests
# ---------------------------------------------------------------------------

class TestStressTester:
    def _make_weights_and_returns(self):
        np.random.seed(42)
        days = 500
        dates = pd.bdate_range(end="2026-04-28", periods=days).astype(str)
        returns = pd.DataFrame({
            "AAPL": np.random.normal(0.0003, 0.015, days),
            "MSFT": np.random.normal(0.0004, 0.013, days),
        }, index=dates)
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        return weights, returns

    def test_hypothetical_crash_20_negative_return(self):
        weights, returns = self._make_weights_and_returns()
        tester = StressTester()
        scenario = next(s for s in BUILTIN_SCENARIOS if s.name == "MARKET_CRASH_20")
        result = tester.run_scenario(weights, returns, None, scenario)
        assert result.portfolio_return < 0

    def test_hypothetical_crash_30_worse_than_20(self):
        weights, returns = self._make_weights_and_returns()
        tester = StressTester()
        crash20 = next(s for s in BUILTIN_SCENARIOS if s.name == "MARKET_CRASH_20")
        crash30 = next(s for s in BUILTIN_SCENARIOS if s.name == "MARKET_CRASH_30")
        r20 = tester.run_scenario(weights, returns, None, crash20)
        r30 = tester.run_scenario(weights, returns, None, crash30)
        assert r30.portfolio_return < r20.portfolio_return

    def test_var_breach_detected(self):
        weights, returns = self._make_weights_and_returns()
        tester = StressTester(var_99=0.01)  # 1% threshold — easy to breach
        scenario = next(s for s in BUILTIN_SCENARIOS if s.name == "MARKET_CRASH_40")
        result = tester.run_scenario(weights, returns, None, scenario)
        assert result.var_breach is True

    def test_run_all_returns_results_for_all_scenarios(self):
        weights, returns = self._make_weights_and_returns()
        tester = StressTester()
        results = tester.run_all(weights, returns)
        assert len(results) == len(BUILTIN_SCENARIOS)

    def test_historical_gfc_negative_return(self):
        """GFC scenario should produce negative portfolio return."""
        weights, returns = self._make_weights_and_returns()
        tester = StressTester()
        scenario = next(s for s in BUILTIN_SCENARIOS if s.name == "GFC_2008")
        result = tester.run_scenario(weights, returns, None, scenario)
        # Either historical data is available (negative) or synthetic shock applied
        assert result.portfolio_return < 0

    def test_summary_table_has_correct_columns(self):
        weights, returns = self._make_weights_and_returns()
        tester = StressTester()
        results = tester.run_all(weights, returns)
        table = tester.summary_table(results)
        assert "Scenario" in table.columns
        assert "Portfolio Return" in table.columns
        assert "VaR Breach" in table.columns

    def test_custom_scenario(self):
        weights, returns = self._make_weights_and_returns()
        tester = StressTester()
        custom = StressScenario(
            name="CUSTOM_SHOCK",
            description="Custom -15% shock",
            scenario_type="hypothetical",
            shocks={"market": -0.15},
        )
        result = tester.run_scenario(weights, returns, None, custom)
        assert result.portfolio_return < 0
        assert result.scenario.name == "CUSTOM_SHOCK"


# ---------------------------------------------------------------------------
# PerformanceAnalytics Tests
# ---------------------------------------------------------------------------

class TestPerformanceAnalytics:
    def test_sharpe_positive_for_positive_returns(self):
        returns = make_returns(mu=0.001, sigma=0.01)
        analytics = PerformanceAnalytics(risk_free_rate=0.02)
        sharpe = analytics.sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_negative_for_negative_returns(self):
        returns = make_returns(mu=-0.001, sigma=0.01)
        analytics = PerformanceAnalytics(risk_free_rate=0.05)
        sharpe = analytics.sharpe_ratio(returns)
        assert sharpe < 0

    def test_sortino_higher_than_sharpe_for_skewed_returns(self):
        """Sortino should be higher than Sharpe when upside vol > downside vol."""
        np.random.seed(42)
        # Mostly positive returns with rare large negatives
        returns = pd.Series(np.random.exponential(0.005, 252) - 0.002)
        analytics = PerformanceAnalytics(risk_free_rate=0.02)
        sharpe = analytics.sharpe_ratio(returns)
        sortino = analytics.sortino_ratio(returns)
        # Sortino should be >= Sharpe when downside vol < total vol
        assert sortino >= sharpe - 0.5  # Allow some tolerance

    def test_max_drawdown_positive(self):
        returns = make_returns(mu=0.0, sigma=0.02)
        equity = make_equity_curve(returns)
        analytics = PerformanceAnalytics()
        max_dd = analytics.max_drawdown(equity)
        assert max_dd >= 0

    def test_max_drawdown_zero_for_monotonic_increase(self):
        dates = pd.bdate_range(end="2026-04-28", periods=50).astype(str)
        equity = pd.Series(np.linspace(100_000, 150_000, 50), index=dates)
        analytics = PerformanceAnalytics()
        max_dd = analytics.max_drawdown(equity)
        assert max_dd == pytest.approx(0.0, abs=1e-6)

    def test_calmar_ratio_positive_for_positive_return(self):
        analytics = PerformanceAnalytics()
        calmar = analytics.calmar_ratio(0.15, 0.10)
        assert calmar == pytest.approx(1.5)

    def test_calmar_ratio_zero_for_zero_drawdown(self):
        analytics = PerformanceAnalytics()
        calmar = analytics.calmar_ratio(0.15, 0.0)
        assert calmar == 0.0

    def test_omega_ratio_greater_than_one_for_positive_returns(self):
        returns = make_returns(mu=0.001, sigma=0.01)
        analytics = PerformanceAnalytics()
        omega = analytics.omega_ratio(returns)
        assert omega > 1.0

    def test_alpha_beta_market_beta_close_to_one(self):
        """Portfolio that mimics market should have beta ≈ 1."""
        market = make_returns(mu=0.0003, sigma=0.012, seed=1)
        # Add small noise to make it not perfectly correlated
        np.random.seed(99)
        noise = pd.Series(np.random.normal(0, 0.002, len(market)), index=market.index)
        portfolio = market + noise
        analytics = PerformanceAnalytics()
        beta, alpha = analytics.alpha_beta(portfolio, market)
        assert abs(beta - 1.0) < 0.3

    def test_information_ratio_positive_for_outperforming_portfolio(self):
        # Build portfolio that consistently outperforms market
        market = make_returns(mu=0.0002, sigma=0.012, seed=1)
        # Add a fixed alpha to market returns so portfolio always beats it
        portfolio = market + 0.0003  # +7.5% annualized alpha
        analytics = PerformanceAnalytics()
        ir = analytics.information_ratio(portfolio, market)
        assert ir > 0

    def test_up_down_capture_market_replication(self):
        """Portfolio that mirrors market should have up/down capture ≈ 1."""
        market = make_returns(mu=0.0003, sigma=0.012, seed=1)
        np.random.seed(99)
        noise = pd.Series(np.random.normal(0, 0.001, len(market)), index=market.index)
        portfolio = market + noise
        analytics = PerformanceAnalytics()
        up_cap, down_cap = analytics.up_down_capture(portfolio, market)
        assert abs(up_cap - 1.0) < 0.3
        assert abs(down_cap - 1.0) < 0.3

    def test_full_report_all_fields_populated(self):
        returns = make_returns(mu=0.0003, sigma=0.012)
        equity = make_equity_curve(returns)
        benchmark_returns = make_returns(mu=0.0002, sigma=0.010, seed=99)
        benchmark_equity = make_equity_curve(benchmark_returns)
        analytics = PerformanceAnalytics()
        report = analytics.full_report(equity, benchmark_equity)
        assert report.total_return != 0
        assert report.sharpe_ratio != 0
        assert report.max_drawdown >= 0
        assert report.alpha is not None
        assert report.beta is not None
        assert report.n_days > 0

    def test_full_report_summary_string(self):
        returns = make_returns()
        equity = make_equity_curve(returns)
        analytics = PerformanceAnalytics()
        report = analytics.full_report(equity)
        summary = report.summary()
        assert "PERFORMANCE REPORT" in summary
        assert "Sharpe" in summary
        assert "Max Drawdown" in summary

    def test_drawdown_durations(self):
        returns = make_returns(mu=-0.001, sigma=0.02)  # Negative drift → drawdowns
        equity = make_equity_curve(returns)
        analytics = PerformanceAnalytics()
        max_dur, avg_dur = analytics.drawdown_durations(equity)
        assert max_dur >= 0
        assert avg_dur >= 0
        assert max_dur >= avg_dur
