"""Performance Analytics.

Computes institutional-grade performance metrics from equity curves and trade logs:
  - Return metrics: total, annualized, CAGR
  - Risk-adjusted: Sharpe, Sortino, Calmar, Omega, Information Ratio
  - Drawdown: max drawdown, average drawdown, drawdown duration
  - Trade statistics: win rate, profit factor, avg win/loss, expectancy
  - Benchmark comparison: alpha, beta, tracking error, up/down capture
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    # Returns
    total_return: float
    annualized_return: float
    cagr: float

    # Risk
    annualized_volatility: float
    downside_deviation: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown_duration_days: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Benchmark
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    up_capture: Optional[float] = None
    down_capture: Optional[float] = None
    benchmark_return: Optional[float] = None

    # Trade stats
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    avg_holding_days: float = 0.0

    # Metadata
    start_date: str = ""
    end_date: str = ""
    n_days: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "PERFORMANCE REPORT",
            "=" * 50,
            f"Period:             {self.start_date} → {self.end_date} ({self.n_days} days)",
            "",
            "── Returns ──────────────────────────────────",
            f"Total Return:       {self.total_return*100:.2f}%",
            f"Annualized Return:  {self.annualized_return*100:.2f}%",
            f"CAGR:               {self.cagr*100:.2f}%",
            "",
            "── Risk ─────────────────────────────────────",
            f"Annualized Vol:     {self.annualized_volatility*100:.2f}%",
            f"Downside Dev:       {self.downside_deviation*100:.2f}%",
            f"Max Drawdown:       {self.max_drawdown*100:.2f}%",
            f"Avg Drawdown:       {self.avg_drawdown*100:.2f}%",
            f"Max DD Duration:    {self.max_drawdown_duration_days} days",
            "",
            "── Risk-Adjusted ────────────────────────────",
            f"Sharpe Ratio:       {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:      {self.sortino_ratio:.3f}",
            f"Calmar Ratio:       {self.calmar_ratio:.3f}",
            f"Omega Ratio:        {self.omega_ratio:.3f}",
        ]

        if self.benchmark_return is not None:
            lines += [
                "",
                "── vs Benchmark ─────────────────────────────",
                f"Benchmark Return:   {self.benchmark_return*100:.2f}%",
                f"Alpha:              {(self.alpha or 0)*100:.2f}%",
                f"Beta:               {self.beta:.3f}" if self.beta else "Beta:               N/A",
                f"Information Ratio:  {self.information_ratio:.3f}" if self.information_ratio else "Info Ratio:         N/A",
                f"Tracking Error:     {(self.tracking_error or 0)*100:.2f}%",
                f"Up Capture:         {(self.up_capture or 0)*100:.1f}%",
                f"Down Capture:       {(self.down_capture or 0)*100:.1f}%",
            ]

        if self.total_trades > 0:
            lines += [
                "",
                "── Trade Statistics ─────────────────────────",
                f"Total Trades:       {self.total_trades}",
                f"Win Rate:           {self.win_rate*100:.1f}%",
                f"Profit Factor:      {self.profit_factor:.2f}",
                f"Avg Win:            {self.avg_win*100:.2f}%",
                f"Avg Loss:           {self.avg_loss*100:.2f}%",
                f"Expectancy:         {self.expectancy*100:.3f}%",
                f"Avg Holding Days:   {self.avg_holding_days:.1f}",
            ]

        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Performance Analytics
# ---------------------------------------------------------------------------

class PerformanceAnalytics:
    """Compute institutional-grade performance metrics.

    Usage:
        analytics = PerformanceAnalytics(risk_free_rate=0.05)
        report = analytics.full_report(equity_series, benchmark_series, trades)
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """Annualized Sharpe ratio."""
        excess = daily_returns - self.rf_daily
        if len(excess) < 2 or excess.std() < 1e-10:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    def sortino_ratio(self, daily_returns: pd.Series, target: float = 0.0) -> float:
        """Annualized Sortino ratio (penalizes only downside deviation)."""
        excess = daily_returns - self.rf_daily
        downside = excess[excess < target]
        if len(downside) < 2 or downside.std() < 1e-10:
            return 0.0
        downside_dev = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        ann_return = float(excess.mean() * TRADING_DAYS_PER_YEAR)
        return ann_return / downside_dev if downside_dev > 0 else 0.0

    def calmar_ratio(self, annualized_return: float, max_dd: float) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        return annualized_return / max_dd if max_dd > 0 else 0.0

    def omega_ratio(self, daily_returns: pd.Series, threshold: float = 0.0) -> float:
        """Omega ratio = probability-weighted gains / probability-weighted losses."""
        excess = daily_returns - self.rf_daily
        gains = excess[excess > threshold].sum()
        losses = abs(excess[excess <= threshold].sum())
        return float(gains / losses) if losses > 0 else float("inf")

    def max_drawdown(self, equity_series: pd.Series) -> float:
        """Maximum drawdown as a positive fraction."""
        if len(equity_series) < 2:
            return 0.0
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        return float(abs(drawdowns.min()))

    def drawdown_series(self, equity_series: pd.Series) -> pd.Series:
        """Full drawdown time series (negative values)."""
        rolling_max = equity_series.expanding().max()
        return (equity_series - rolling_max) / rolling_max

    def avg_drawdown(self, equity_series: pd.Series) -> float:
        """Average drawdown (only during drawdown periods)."""
        dd = self.drawdown_series(equity_series)
        in_drawdown = dd[dd < 0]
        return float(abs(in_drawdown.mean())) if len(in_drawdown) > 0 else 0.0

    def drawdown_durations(self, equity_series: pd.Series) -> Tuple[int, float]:
        """Returns (max_duration_days, avg_duration_days)."""
        dd = self.drawdown_series(equity_series)
        in_dd = dd < 0

        durations = []
        current_duration = 0
        for is_dd in in_dd:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)

        if not durations:
            return 0, 0.0
        return int(max(durations)), float(np.mean(durations))

    def alpha_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Compute Jensen's alpha and market beta via OLS."""
        combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(combined) < 30:
            return 0.0, 1.0

        y = combined.iloc[:, 0].values
        x = combined.iloc[:, 1].values

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            beta = float(slope)
            alpha_daily = float(intercept)
            alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR
            return beta, alpha_annual
        except Exception:
            return 0.0, 1.0

    def information_ratio(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Information ratio = active return / tracking error."""
        combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(combined) < 20:
            return 0.0
        active = combined.iloc[:, 0] - combined.iloc[:, 1]
        te = float(active.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        active_return = float(active.mean() * TRADING_DAYS_PER_YEAR)
        return active_return / te if te > 0 else 0.0

    def tracking_error(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Annualized tracking error."""
        combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(combined) < 20:
            return 0.0
        active = combined.iloc[:, 0] - combined.iloc[:, 1]
        return float(active.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    def up_down_capture(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Up capture and down capture ratios."""
        combined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(combined) < 20:
            return 1.0, 1.0

        port = combined.iloc[:, 0]
        bench = combined.iloc[:, 1]

        up_bench = bench[bench > 0]
        down_bench = bench[bench < 0]

        up_port = port[bench > 0]
        down_port = port[bench < 0]

        up_capture = float(up_port.mean() / up_bench.mean()) if len(up_bench) > 0 and up_bench.mean() != 0 else 1.0
        down_capture = float(down_port.mean() / down_bench.mean()) if len(down_bench) > 0 and down_bench.mean() != 0 else 1.0

        return up_capture, down_capture

    def downside_deviation(self, daily_returns: pd.Series, target: float = 0.0) -> float:
        """Annualized downside deviation."""
        downside = daily_returns[daily_returns < target]
        if len(downside) < 2:
            return 0.0
        return float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        equity_series: pd.Series,
        benchmark_series: Optional[pd.Series] = None,
        trades: Optional[List] = None,
    ) -> PerformanceReport:
        """Generate a complete performance report.

        Args:
            equity_series: Portfolio value time series (indexed by date strings).
            benchmark_series: Optional benchmark value series for comparison.
            trades: Optional list of Trade objects for trade statistics.

        Returns:
            PerformanceReport with all metrics.
        """
        if len(equity_series) < 2:
            return self._empty_report()

        daily_returns = equity_series.pct_change().dropna()
        n_days = len(equity_series)
        n_years = n_days / TRADING_DAYS_PER_YEAR

        initial = float(equity_series.iloc[0])
        final = float(equity_series.iloc[-1])
        total_return = (final - initial) / initial
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        cagr = ann_return  # Same as annualized return for continuous compounding

        ann_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        dd_dev = self.downside_deviation(daily_returns)
        max_dd = self.max_drawdown(equity_series)
        avg_dd = self.avg_drawdown(equity_series)
        max_dd_dur, avg_dd_dur = self.drawdown_durations(equity_series)

        sharpe = self.sharpe_ratio(daily_returns)
        sortino = self.sortino_ratio(daily_returns)
        calmar = self.calmar_ratio(ann_return, max_dd)
        omega = self.omega_ratio(daily_returns)

        # Benchmark metrics
        alpha = beta = ir = te = up_cap = down_cap = bm_return = None
        if benchmark_series is not None and len(benchmark_series) > 10:
            bm_returns = benchmark_series.pct_change().dropna()
            bm_return = float((benchmark_series.iloc[-1] / benchmark_series.iloc[0]) - 1)
            beta_val, alpha_val = self.alpha_beta(daily_returns, bm_returns)
            beta = beta_val
            alpha = alpha_val
            ir = self.information_ratio(daily_returns, bm_returns)
            te = self.tracking_error(daily_returns, bm_returns)
            up_cap, down_cap = self.up_down_capture(daily_returns, bm_returns)

        # Trade statistics
        total_trades = win_rate = profit_factor = avg_win = avg_loss = expectancy = avg_holding = 0.0
        if trades:
            closed = [t for t in trades if getattr(t, "exit_date", None) is not None]
            if closed:
                total_trades = len(closed)
                trade_returns = []
                for t in closed:
                    if hasattr(t, "fill_price") and t.fill_price and t.quantity:
                        cost = t.fill_price * t.quantity
                        trade_returns.append(t.net_pnl / cost if cost > 0 else 0.0)

                if trade_returns:
                    wins = [r for r in trade_returns if r > 0]
                    losses = [r for r in trade_returns if r <= 0]
                    win_rate = len(wins) / len(trade_returns)
                    avg_win = float(np.mean(wins)) if wins else 0.0
                    avg_loss = float(np.mean(losses)) if losses else 0.0
                    gross_profit = sum(t.net_pnl for t in closed if t.net_pnl > 0)
                    gross_loss = abs(sum(t.net_pnl for t in closed if t.net_pnl <= 0))
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
                    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
                    avg_holding = float(np.mean([getattr(t, "holding_days", 0) for t in closed]))

        start_date = str(equity_series.index[0]) if len(equity_series) > 0 else ""
        end_date = str(equity_series.index[-1]) if len(equity_series) > 0 else ""

        return PerformanceReport(
            total_return=total_return,
            annualized_return=ann_return,
            cagr=cagr,
            annualized_volatility=ann_vol,
            downside_deviation=dd_dev,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_dd_dur,
            avg_drawdown_duration_days=avg_dd_dur,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            alpha=alpha,
            beta=beta,
            information_ratio=ir,
            tracking_error=te,
            up_capture=up_cap,
            down_capture=down_cap,
            benchmark_return=bm_return,
            total_trades=int(total_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            avg_holding_days=avg_holding,
            start_date=start_date,
            end_date=end_date,
            n_days=n_days,
        )

    def _empty_report(self) -> PerformanceReport:
        return PerformanceReport(
            total_return=0.0, annualized_return=0.0, cagr=0.0,
            annualized_volatility=0.0, downside_deviation=0.0,
            max_drawdown=0.0, avg_drawdown=0.0,
            max_drawdown_duration_days=0, avg_drawdown_duration_days=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0, omega_ratio=0.0,
        )
