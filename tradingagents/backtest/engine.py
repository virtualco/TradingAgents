"""Core Backtesting Engine.

Event-driven simulation that replays signals from the SignalRegistry against
point-in-time OHLCV data. Enforces:
  - T+1 execution (signal on day N fills at open on day N+1)
  - No lookahead bias (uses only data available at trade_date)
  - Realistic transaction costs (commission + slippage)
  - Position sizing via fixed fractional or signal-conviction weighting
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


@dataclass
class Trade:
    """A single executed trade."""
    trade_id: str
    ticker: str
    signal_id: str
    side: OrderSide
    quantity: float
    fill_price: float
    fill_date: str
    commission: float
    slippage: float
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    holding_days: int = 0
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None


@dataclass
class Position:
    """An open position."""
    ticker: str
    signal_id: str
    entry_date: str
    entry_price: float
    quantity: float
    direction: str          # "long" or "short"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    conviction: float = 0.5
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001          # 10 bps per trade
    slippage_pct: float = 0.0005           # 5 bps slippage
    max_position_pct: float = 0.10         # Max 10% per position
    max_open_positions: int = 20
    position_sizing: str = "conviction"    # "fixed" | "conviction" | "equal"
    fixed_position_pct: float = 0.05       # Used when sizing="fixed"
    allow_short: bool = False
    execution_delay_days: int = 1          # T+1 execution
    min_conviction: float = 0.30           # Ignore signals below this
    stop_loss_pct: Optional[float] = 0.08  # 8% stop loss (None = disabled)
    take_profit_pct: Optional[float] = 0.20  # 20% take profit (None = disabled)
    benchmark_ticker: str = "SPY"
    risk_free_rate: float = 0.05           # 5% annual


@dataclass
class EquityCurvePoint:
    date: str
    portfolio_value: float
    cash: float
    positions_value: float
    daily_return: float
    drawdown: float


@dataclass
class BacktestResult:
    """Full results of a backtest run."""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: List[EquityCurvePoint]
    final_portfolio_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_holding_days: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Backtest Summary ===",
            f"Total Return:       {self.total_return*100:.2f}%",
            f"Annualized Return:  {self.annualized_return*100:.2f}%",
            f"Sharpe Ratio:       {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:      {self.sortino_ratio:.3f}",
            f"Max Drawdown:       {self.max_drawdown*100:.2f}%",
            f"Calmar Ratio:       {self.calmar_ratio:.3f}",
            f"Win Rate:           {self.win_rate*100:.1f}%",
            f"Profit Factor:      {self.profit_factor:.2f}",
            f"Total Trades:       {self.total_trades}",
            f"Avg Holding Days:   {self.avg_holding_days:.1f}",
        ]
        if self.benchmark_return is not None:
            lines.append(f"Benchmark Return:   {self.benchmark_return*100:.2f}%")
        if self.alpha is not None:
            lines.append(f"Alpha:              {self.alpha*100:.2f}%")
        if self.beta is not None:
            lines.append(f"Beta:               {self.beta:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtesting engine.

    Usage:
        engine = BacktestEngine(config=BacktestConfig())
        result = engine.run(signals_df, price_data)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._reset()

    def _reset(self):
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}   # ticker -> Position
        self.trades: List[Trade] = []
        self.equity_curve: List[EquityCurvePoint] = []
        self._trade_counter = 0

    def run(
        self,
        signals: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        benchmark_prices: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            signals: DataFrame with columns [signal_id, ticker, trade_date,
                     direction, conviction, stop_loss, take_profit].
            price_data: Dict of ticker -> OHLCV DataFrame with 'event_time',
                        'open', 'high', 'low', 'close' columns.
            benchmark_prices: Optional Series of benchmark close prices indexed
                              by date string for alpha/beta calculation.

        Returns:
            BacktestResult with full trade log and equity curve.
        """
        self._reset()

        if signals.empty:
            return self._empty_result()

        # Build a sorted list of all trading dates
        all_dates = self._get_all_dates(price_data)
        if not all_dates:
            return self._empty_result()

        # Index signals by execution date (T+1)
        pending_signals = self._index_signals(signals, all_dates)

        prev_value = self.config.initial_capital

        for trade_date in all_dates:
            date_str = str(trade_date)

            # 1. Update current prices for open positions
            self._update_positions(date_str, price_data)

            # 2. Check stop-loss / take-profit exits
            self._check_exits(date_str, price_data)

            # 3. Execute pending signals (T+1 fill at open)
            if date_str in pending_signals:
                for sig in pending_signals[date_str]:
                    self._execute_signal(sig, date_str, price_data)

            # 4. Record equity curve point
            portfolio_value = self._portfolio_value()
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
            drawdown = self._current_drawdown(portfolio_value)

            self.equity_curve.append(EquityCurvePoint(
                date=date_str,
                portfolio_value=portfolio_value,
                cash=self.cash,
                positions_value=portfolio_value - self.cash,
                daily_return=daily_return,
                drawdown=drawdown,
            ))
            prev_value = portfolio_value

        # Close all remaining positions at last available price
        last_date = str(all_dates[-1])
        self._close_all_positions(last_date, price_data)

        return self._build_result(benchmark_prices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_all_dates(self, price_data: Dict[str, pd.DataFrame]) -> List:
        """Get sorted list of all trading dates across all tickers."""
        all_dates = set()
        for df in price_data.values():
            if df.empty or "event_time" not in df.columns:
                continue
            dates = pd.to_datetime(df["event_time"]).dt.date.tolist()
            all_dates.update(dates)
        return sorted(all_dates)

    def _index_signals(
        self, signals: pd.DataFrame, all_dates: List
    ) -> Dict[str, List[Dict]]:
        """Index signals by their T+1 execution date."""
        date_set = {str(d) for d in all_dates}
        pending: Dict[str, List[Dict]] = {}

        for _, row in signals.iterrows():
            conviction = float(row.get("conviction", 0.5))
            if conviction < self.config.min_conviction:
                continue

            direction = str(row.get("direction", "flat")).lower()
            if direction in ("flat", "neutral"):
                continue

            if not self.config.allow_short and direction in ("short", "sell"):
                continue

            # T+1 execution
            signal_date = pd.to_datetime(row["trade_date"]).date()
            exec_date = self._next_trading_date(signal_date, all_dates)
            if exec_date is None:
                continue

            exec_str = str(exec_date)
            if exec_str not in pending:
                pending[exec_str] = []

            pending[exec_str].append({
                "signal_id": str(row.get("signal_id", "")),
                "ticker": str(row["ticker"]).upper(),
                "direction": direction,
                "conviction": conviction,
                "stop_loss": row.get("stop_loss"),
                "take_profit": row.get("take_profit"),
            })

        return pending

    def _next_trading_date(self, signal_date: date, all_dates: List) -> Optional[date]:
        """Find the next trading date after signal_date."""
        for d in all_dates:
            if d > signal_date:
                return d
        return None

    def _get_open_price(self, ticker: str, date_str: str, price_data: Dict) -> Optional[float]:
        """Get the open price for a ticker on a given date."""
        df = price_data.get(ticker)
        if df is None or df.empty:
            return None
        mask = pd.to_datetime(df["event_time"]).dt.date.astype(str) == date_str
        row = df[mask]
        if row.empty:
            return None
        return float(row["open"].iloc[0])

    def _get_close_price(self, ticker: str, date_str: str, price_data: Dict) -> Optional[float]:
        """Get the close price for a ticker on a given date."""
        df = price_data.get(ticker)
        if df is None or df.empty:
            return None
        mask = pd.to_datetime(df["event_time"]).dt.date.astype(str) == date_str
        row = df[mask]
        if row.empty:
            return None
        return float(row["close"].iloc[0])

    def _get_day_range(self, ticker: str, date_str: str, price_data: Dict) -> Optional[Tuple[float, float]]:
        """Get (low, high) for a ticker on a given date."""
        df = price_data.get(ticker)
        if df is None or df.empty:
            return None
        mask = pd.to_datetime(df["event_time"]).dt.date.astype(str) == date_str
        row = df[mask]
        if row.empty:
            return None
        return float(row["low"].iloc[0]), float(row["high"].iloc[0])

    def _execute_signal(self, sig: Dict, date_str: str, price_data: Dict):
        """Execute a signal — open or close a position."""
        ticker = sig["ticker"]
        direction = sig["direction"]

        # Skip if already have an open position in this ticker
        if ticker in self.positions:
            return

        # Skip if at max positions
        if len(self.positions) >= self.config.max_open_positions:
            logger.debug(f"Max positions reached, skipping {ticker}")
            return

        # Get fill price (open of execution day + slippage)
        fill_price = self._get_open_price(ticker, date_str, price_data)
        if fill_price is None or fill_price <= 0:
            return

        slippage = fill_price * self.config.slippage_pct
        if direction in ("buy", "long"):
            fill_price += slippage
        else:
            fill_price -= slippage

        # Determine position size
        position_value = self._compute_position_size(sig["conviction"])
        if position_value <= 0 or position_value > self.cash:
            return

        quantity = position_value / fill_price
        commission = position_value * self.config.commission_pct

        if commission > self.cash - position_value:
            return

        self.cash -= position_value + commission

        # Compute stop/target prices
        stop_loss = None
        take_profit = None
        if self.config.stop_loss_pct and direction in ("buy", "long"):
            stop_loss = fill_price * (1 - self.config.stop_loss_pct)
        if self.config.take_profit_pct and direction in ("buy", "long"):
            take_profit = fill_price * (1 + self.config.take_profit_pct)

        # Override with signal-level stops if provided
        if sig.get("stop_loss") and not pd.isna(sig["stop_loss"]):
            stop_loss = float(sig["stop_loss"])
        if sig.get("take_profit") and not pd.isna(sig["take_profit"]):
            take_profit = float(sig["take_profit"])

        self.positions[ticker] = Position(
            ticker=ticker,
            signal_id=sig["signal_id"],
            entry_date=date_str,
            entry_price=fill_price,
            quantity=quantity,
            direction="long" if direction in ("buy", "long") else "short",
            stop_loss=stop_loss,
            take_profit=take_profit,
            conviction=sig["conviction"],
            current_price=fill_price,
        )

        self._trade_counter += 1
        self.trades.append(Trade(
            trade_id=f"T{self._trade_counter:05d}",
            ticker=ticker,
            signal_id=sig["signal_id"],
            side=OrderSide.BUY if direction in ("buy", "long") else OrderSide.SHORT,
            quantity=quantity,
            fill_price=fill_price,
            fill_date=date_str,
            commission=commission,
            slippage=slippage * quantity,
        ))

    def _compute_position_size(self, conviction: float) -> float:
        """Compute the dollar value to allocate to a position."""
        portfolio_value = self._portfolio_value()

        if self.config.position_sizing == "fixed":
            pct = self.config.fixed_position_pct
        elif self.config.position_sizing == "conviction":
            # Scale between 2% and max_position_pct based on conviction
            min_pct = 0.02
            pct = min_pct + (self.config.max_position_pct - min_pct) * conviction
        else:  # equal
            n_slots = self.config.max_open_positions
            pct = 1.0 / n_slots

        pct = min(pct, self.config.max_position_pct)
        return portfolio_value * pct

    def _update_positions(self, date_str: str, price_data: Dict):
        """Update current prices and unrealized PnL for all open positions."""
        for ticker, pos in self.positions.items():
            close = self._get_close_price(ticker, date_str, price_data)
            if close is not None:
                pos.current_price = close
                if pos.direction == "long":
                    pos.unrealized_pnl = (close - pos.entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.entry_price - close) * pos.quantity

    def _check_exits(self, date_str: str, price_data: Dict):
        """Check stop-loss and take-profit exits for all open positions."""
        to_close = []
        for ticker, pos in self.positions.items():
            day_range = self._get_day_range(ticker, date_str, price_data)
            if day_range is None:
                continue
            low, high = day_range

            if pos.direction == "long":
                if pos.stop_loss and low <= pos.stop_loss:
                    to_close.append((ticker, pos.stop_loss, "stop_loss"))
                elif pos.take_profit and high >= pos.take_profit:
                    to_close.append((ticker, pos.take_profit, "take_profit"))
            else:
                if pos.take_profit and low <= pos.take_profit:
                    to_close.append((ticker, pos.take_profit, "take_profit"))
                elif pos.stop_loss and high >= pos.stop_loss:
                    to_close.append((ticker, pos.stop_loss, "stop_loss"))

        for ticker, exit_price, reason in to_close:
            self._close_position(ticker, exit_price, date_str, reason)

    def _close_position(self, ticker: str, exit_price: float, date_str: str, reason: str = "signal"):
        """Close an open position and record the trade."""
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return

        slippage = exit_price * self.config.slippage_pct
        if pos.direction == "long":
            exit_price -= slippage
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            exit_price += slippage
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        proceeds = exit_price * pos.quantity
        commission = proceeds * self.config.commission_pct
        net_pnl = gross_pnl - commission

        self.cash += proceeds - commission

        # Update the opening trade record
        for trade in reversed(self.trades):
            if trade.ticker == ticker and trade.signal_id == pos.signal_id and trade.exit_date is None:
                entry_date = pd.to_datetime(pos.entry_date).date()
                exit_date_obj = pd.to_datetime(date_str).date()
                trade.holding_days = (exit_date_obj - entry_date).days
                trade.exit_price = exit_price
                trade.exit_date = date_str
                trade.gross_pnl = gross_pnl
                trade.net_pnl = net_pnl
                break

    def _close_all_positions(self, date_str: str, price_data: Dict):
        """Close all remaining open positions at the last available price."""
        tickers = list(self.positions.keys())
        for ticker in tickers:
            close = self._get_close_price(ticker, date_str, price_data)
            if close is None:
                close = self.positions[ticker].current_price
            self._close_position(ticker, close, date_str, "end_of_backtest")

    def _portfolio_value(self) -> float:
        """Compute total portfolio value (cash + positions)."""
        positions_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        return self.cash + positions_value

    def _current_drawdown(self, current_value: float) -> float:
        """Compute current drawdown from peak."""
        if not self.equity_curve:
            return 0.0
        peak = max(p.portfolio_value for p in self.equity_curve)
        peak = max(peak, current_value)
        if peak <= 0:
            return 0.0
        return (peak - current_value) / peak

    def _build_result(self, benchmark_prices: Optional[pd.Series]) -> BacktestResult:
        """Build the final BacktestResult from accumulated state."""
        from .analytics import PerformanceAnalytics
        analytics = PerformanceAnalytics(self.config.risk_free_rate)

        equity_series = pd.Series(
            [p.portfolio_value for p in self.equity_curve],
            index=[p.date for p in self.equity_curve],
        )
        daily_returns = pd.Series(
            [p.daily_return for p in self.equity_curve],
            index=[p.date for p in self.equity_curve],
        )

        final_value = equity_series.iloc[-1] if len(equity_series) > 0 else self.config.initial_capital
        total_return = (final_value - self.config.initial_capital) / self.config.initial_capital

        n_days = len(self.equity_curve)
        n_years = n_days / 252.0
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        closed_trades = [t for t in self.trades if t.exit_date is not None]
        wins = [t for t in closed_trades if t.net_pnl > 0]
        losses = [t for t in closed_trades if t.net_pnl <= 0]

        win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_holding = np.mean([t.holding_days for t in closed_trades]) if closed_trades else 0.0
        trade_returns = [t.net_pnl / (t.fill_price * t.quantity) for t in closed_trades if t.fill_price * t.quantity > 0]
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0.0
        best_trade = max(trade_returns) if trade_returns else 0.0
        worst_trade = min(trade_returns) if trade_returns else 0.0

        sharpe = analytics.sharpe_ratio(daily_returns)
        sortino = analytics.sortino_ratio(daily_returns)
        max_dd = analytics.max_drawdown(equity_series)
        calmar = annualized_return / max_dd if max_dd > 0 else 0.0

        # Benchmark comparison
        benchmark_return = None
        alpha = None
        beta = None
        if benchmark_prices is not None and len(benchmark_prices) > 1:
            bm_returns = benchmark_prices.pct_change().dropna()
            benchmark_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
            beta, alpha = analytics.alpha_beta(daily_returns, bm_returns)

        return BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            final_portfolio_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(closed_trades),
            avg_holding_days=float(avg_holding),
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
        )

    def _empty_result(self) -> BacktestResult:
        """Return an empty result when no signals are provided."""
        return BacktestResult(
            config=self.config,
            trades=[],
            equity_curve=[],
            final_portfolio_value=self.config.initial_capital,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_holding_days=0.0,
            avg_trade_return=0.0,
            best_trade=0.0,
            worst_trade=0.0,
        )
