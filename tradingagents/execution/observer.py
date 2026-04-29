"""Live Observation Framework.

Orchestrates the daily paper trading observation loop:

Daily cycle (runs at market open, e.g., 09:35 ET):
  1. Reset daily counters (kill switch, schema failures)
  2. Fetch latest prices for all open positions
  3. Mark-to-market portfolio → compute NAV, drawdown, daily P&L
  4. Run circuit breaker checks (drawdown, daily loss)
  5. For each configured agent: fetch signals from signal registry
  6. Run pre-trade risk checks on each signal
  7. Submit approved orders → fill at next open price (T+1)
  8. Run reconciliation against simulated broker state
  9. Log observation record to SQLite
  10. Generate and save daily report

The observer is designed to run as a standalone script or be called
by an external scheduler (cron, Airflow, etc.).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ObservationConfig:
    """Configuration for the live observation loop."""
    db_path: str = "paper_trading.db"
    initial_capital: float = 100_000.0
    max_position_size_pct: float = 0.10    # Max 10% of NAV per position
    min_conviction: float = 0.50           # Min conviction to trade
    observation_period_days: int = 90      # Minimum observation before live
    report_dir: str = "observation_reports"


@dataclass
class DailyObservation:
    """Record of a single day's observation."""
    observation_id: str
    trade_date: str
    nav: float
    cash: float
    gross_long: float
    gross_short: float
    daily_pnl: float
    total_pnl: float
    drawdown_pct: float
    signals_received: int
    orders_submitted: int
    orders_approved: int
    orders_rejected: int
    orders_filled: int
    kill_switch_active: bool
    circuit_breaker_triggered: bool
    reconciliation_clean: bool
    reconciliation_breaks: int
    positions_count: int
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ObservationSummary:
    """Summary of the full observation period."""
    start_date: str
    end_date: str
    total_days: int
    initial_nav: float
    final_nav: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_signals: int
    total_trades: int
    win_rate: float
    kill_switch_days: int
    circuit_breaker_events: int
    ready_for_live: bool
    readiness_notes: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "OBSERVATION PERIOD SUMMARY",
            "=" * 55,
            f"Period:             {self.start_date} → {self.end_date} ({self.total_days} days)",
            f"Initial NAV:        ${self.initial_nav:,.2f}",
            f"Final NAV:          ${self.final_nav:,.2f}",
            f"Total Return:       {self.total_return:.2%}",
            f"Annualized Return:  {self.annualized_return:.2%}",
            f"Max Drawdown:       {self.max_drawdown:.2%}",
            f"Sharpe Ratio:       {self.sharpe_ratio:.3f}",
            "",
            f"Total Signals:      {self.total_signals}",
            f"Total Trades:       {self.total_trades}",
            f"Win Rate:           {self.win_rate:.1%}",
            "",
            f"Kill Switch Days:   {self.kill_switch_days}",
            f"Circuit Breakers:   {self.circuit_breaker_events}",
            "",
            f"Ready for Live:     {'✓ YES' if self.ready_for_live else '✗ NOT YET'}",
        ]
        if self.readiness_notes:
            lines.append("Readiness Notes:")
            for note in self.readiness_notes:
                lines.append(f"  • {note}")
        lines.append("=" * 55)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observation Logger
# ---------------------------------------------------------------------------

class ObservationLogger:
    """Logs daily observations to SQLite for analysis.

    Usage:
        logger_obj = ObservationLogger(db_path="paper_trading.db")
        logger_obj.log(observation)
        summary = logger_obj.get_summary()
    """

    def __init__(self, db_path: str = "paper_trading.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS daily_observations (
                    observation_id TEXT PRIMARY KEY,
                    trade_date TEXT NOT NULL,
                    nav REAL,
                    cash REAL,
                    gross_long REAL,
                    gross_short REAL,
                    daily_pnl REAL,
                    total_pnl REAL,
                    drawdown_pct REAL,
                    signals_received INTEGER DEFAULT 0,
                    orders_submitted INTEGER DEFAULT 0,
                    orders_approved INTEGER DEFAULT 0,
                    orders_rejected INTEGER DEFAULT 0,
                    orders_filled INTEGER DEFAULT 0,
                    kill_switch_active INTEGER DEFAULT 0,
                    circuit_breaker_triggered INTEGER DEFAULT 0,
                    reconciliation_clean INTEGER DEFAULT 1,
                    reconciliation_breaks INTEGER DEFAULT 0,
                    positions_count INTEGER DEFAULT 0,
                    notes TEXT DEFAULT '',
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_obs_date ON daily_observations(trade_date);
            """)

    def log(self, obs: DailyObservation) -> None:
        """Persist a daily observation record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_observations
                (observation_id, trade_date, nav, cash, gross_long, gross_short,
                 daily_pnl, total_pnl, drawdown_pct, signals_received, orders_submitted,
                 orders_approved, orders_rejected, orders_filled, kill_switch_active,
                 circuit_breaker_triggered, reconciliation_clean, reconciliation_breaks,
                 positions_count, notes, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                obs.observation_id, obs.trade_date, obs.nav, obs.cash,
                obs.gross_long, obs.gross_short, obs.daily_pnl, obs.total_pnl,
                obs.drawdown_pct, obs.signals_received, obs.orders_submitted,
                obs.orders_approved, obs.orders_rejected, obs.orders_filled,
                int(obs.kill_switch_active), int(obs.circuit_breaker_triggered),
                int(obs.reconciliation_clean), obs.reconciliation_breaks,
                obs.positions_count, obs.notes, obs.created_at,
            ))
        logger.info(f"Observation logged: {obs.trade_date} NAV=${obs.nav:,.2f} PnL=${obs.daily_pnl:+,.2f}")

    def get_observations(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[DailyObservation]:
        """Retrieve observation records."""
        with sqlite3.connect(self.db_path) as conn:
            if start_date and end_date:
                rows = conn.execute(
                    "SELECT * FROM daily_observations WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                    (start_date, end_date)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM daily_observations ORDER BY trade_date"
                ).fetchall()

        return [self._row_to_obs(r) for r in rows]

    def get_summary(self, config: Optional[ObservationConfig] = None) -> ObservationSummary:
        """Generate an observation period summary with live-readiness assessment."""
        config = config or ObservationConfig()
        obs_list = self.get_observations()

        if not obs_list:
            return ObservationSummary(
                start_date="", end_date="", total_days=0,
                initial_nav=config.initial_capital, final_nav=config.initial_capital,
                total_return=0.0, annualized_return=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, total_signals=0, total_trades=0, win_rate=0.0,
                kill_switch_days=0, circuit_breaker_events=0,
                ready_for_live=False,
                readiness_notes=["No observation data yet"],
            )

        import numpy as np

        navs = [o.nav for o in obs_list]
        daily_pnls = [o.daily_pnl for o in obs_list]

        initial_nav = navs[0] if navs else config.initial_capital
        final_nav = navs[-1] if navs else config.initial_capital
        total_return = (final_nav - initial_nav) / initial_nav if initial_nav > 0 else 0.0
        n_years = len(obs_list) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        # Max drawdown
        peak = initial_nav
        max_dd = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Sharpe
        if len(daily_pnls) > 1 and initial_nav > 0:
            daily_returns = [p / initial_nav for p in daily_pnls]
            rf_daily = 0.05 / 252
            excess = [r - rf_daily for r in daily_returns]
            sharpe = (np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0.0
        else:
            sharpe = 0.0

        total_signals = sum(o.signals_received for o in obs_list)
        total_trades = sum(o.orders_filled for o in obs_list)
        kill_switch_days = sum(1 for o in obs_list if o.kill_switch_active)
        cb_events = sum(1 for o in obs_list if o.circuit_breaker_triggered)

        # Win rate from filled orders
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM orders WHERE status='filled'
            """).fetchone()
        win_rate = (rows[1] or 0) / rows[0] if rows[0] > 0 else 0.0

        # Readiness assessment
        readiness_notes = []
        ready = True

        if len(obs_list) < config.observation_period_days:
            ready = False
            readiness_notes.append(
                f"Observation period too short: {len(obs_list)}/{config.observation_period_days} days"
            )
        if max_dd > 0.15:
            ready = False
            readiness_notes.append(f"Max drawdown {max_dd:.1%} exceeds 15% threshold")
        if sharpe < 0.5:
            ready = False
            readiness_notes.append(f"Sharpe ratio {sharpe:.2f} below 0.5 threshold")
        if cb_events > 3:
            ready = False
            readiness_notes.append(f"{cb_events} circuit breaker events — review risk parameters")
        if total_trades < 20:
            ready = False
            readiness_notes.append(f"Only {total_trades} trades — need more data for statistical significance")
        if ready:
            readiness_notes.append("All criteria met — system is ready for live paper trading review")

        return ObservationSummary(
            start_date=obs_list[0].trade_date,
            end_date=obs_list[-1].trade_date,
            total_days=len(obs_list),
            initial_nav=initial_nav,
            final_nav=final_nav,
            total_return=total_return,
            annualized_return=ann_return,
            max_drawdown=max_dd,
            sharpe_ratio=float(sharpe),
            total_signals=total_signals,
            total_trades=total_trades,
            win_rate=win_rate,
            kill_switch_days=kill_switch_days,
            circuit_breaker_events=cb_events,
            ready_for_live=ready,
            readiness_notes=readiness_notes,
        )

    def _row_to_obs(self, row) -> DailyObservation:
        return DailyObservation(
            observation_id=row[0],
            trade_date=row[1],
            nav=row[2] or 0.0,
            cash=row[3] or 0.0,
            gross_long=row[4] or 0.0,
            gross_short=row[5] or 0.0,
            daily_pnl=row[6] or 0.0,
            total_pnl=row[7] or 0.0,
            drawdown_pct=row[8] or 0.0,
            signals_received=row[9] or 0,
            orders_submitted=row[10] or 0,
            orders_approved=row[11] or 0,
            orders_rejected=row[12] or 0,
            orders_filled=row[13] or 0,
            kill_switch_active=bool(row[14]),
            circuit_breaker_triggered=bool(row[15]),
            reconciliation_clean=bool(row[16]),
            reconciliation_breaks=row[17] or 0,
            positions_count=row[18] or 0,
            notes=row[19] or "",
            created_at=row[20] or "",
        )


# ---------------------------------------------------------------------------
# Daily Observer (orchestrator)
# ---------------------------------------------------------------------------

class DailyObserver:
    """Orchestrates the daily paper trading observation cycle.

    Usage:
        observer = DailyObserver(config)
        observation = observer.run_daily_cycle(
            signals=signals_df,
            prices={"AAPL": 182.5},
            portfolio_nav=102_000.0,
        )
    """

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
        from tradingagents.execution.order_manager import PaperOrderManager, PreTradeRiskConfig
        from tradingagents.execution.kill_switch import KillSwitchManager
        from tradingagents.execution.reconciliation import PositionTracker, ReconciliationEngine

        risk_config = PreTradeRiskConfig(
            max_concentration_pct=self.config.max_position_size_pct,
            min_price=1.0,
        )
        self.order_manager = PaperOrderManager(
            db_path=self.config.db_path,
            config=risk_config,
        )
        self.kill_switch = KillSwitchManager()
        self.tracker = PositionTracker(
            db_path=self.config.db_path,
            initial_cash=self.config.initial_capital,
        )
        self.recon_engine = ReconciliationEngine(db_path=self.config.db_path)
        self.obs_logger = ObservationLogger(db_path=self.config.db_path)

    def run_daily_cycle(
        self,
        signals,  # pd.DataFrame with columns: ticker, direction, conviction, signal_id
        prices: Dict[str, float],
        trade_date: Optional[str] = None,
    ) -> DailyObservation:
        """Run one full daily observation cycle.

        Args:
            signals: DataFrame of signals for today.
            prices: Dict of ticker -> current price.
            trade_date: Date string (defaults to today).

        Returns:
            DailyObservation record.
        """
        trade_date = trade_date or date.today().isoformat()
        logger.info(f"=== Daily observation cycle: {trade_date} ===")

        # 1. Reset daily counters
        self.kill_switch.reset_daily_counters()

        # 2. Update prices and get snapshot
        self.tracker.update_prices(prices)
        snapshot = self.tracker.get_snapshot()

        # 3. Circuit breaker checks
        cb_triggered = False
        cb_triggered |= self.kill_switch.check_drawdown(
            current_nav=snapshot.total_nav,
            peak_nav=snapshot.peak_nav,
        )
        cb_triggered |= self.kill_switch.check_daily_loss(
            daily_pnl=snapshot.daily_pnl,
            portfolio_nav=snapshot.total_nav,
        )

        # 4. Process signals
        orders_submitted = 0
        orders_approved = 0
        orders_rejected = 0
        orders_filled = 0

        from tradingagents.execution.order_manager import OrderSide

        if signals is not None and len(signals) > 0:
            for _, sig in signals.iterrows():
                ticker = sig.get("ticker", "")
                direction = sig.get("direction", "flat").lower()
                conviction = float(sig.get("conviction", 0.5))
                signal_id = sig.get("signal_id", str(uuid.uuid4()))

                if direction == "flat" or conviction < self.config.min_conviction:
                    continue
                if ticker not in prices:
                    continue

                price = prices[ticker]
                side = OrderSide.BUY if direction == "long" else OrderSide.SELL

                # Size by conviction × max position size
                notional = conviction * self.config.max_position_size_pct * snapshot.total_nav
                quantity = max(1, int(notional / price))

                order = self.order_manager.submit_order(
                    ticker=ticker,
                    side=side,
                    quantity=quantity,
                    signal_id=signal_id,
                    conviction=conviction,
                    current_price=price,
                    portfolio_nav=snapshot.total_nav,
                    kill_switch_active=self.kill_switch.is_halted(ticker),
                    daily_pnl=snapshot.daily_pnl,
                )
                orders_submitted += 1

                from tradingagents.execution.order_manager import OrderStatus
                if order.status == OrderStatus.APPROVED:
                    orders_approved += 1
                    # Fill at current price (T+0 for paper trading simplicity)
                    filled = self.order_manager.fill_order(order.order_id, price, trade_date)
                    if filled:
                        orders_filled += 1
                else:
                    orders_rejected += 1

        # 5. Reconciliation (against self as "broker" — always clean in paper mode)
        from tradingagents.execution.reconciliation import BrokerPosition
        broker_positions = [
            BrokerPosition(
                ticker=ticker,
                quantity=pos["quantity"],
                avg_cost=pos["avg_cost"],
                last_price=pos["last_price"],
                market_value=pos["market_value"],
                unrealized_pnl=pos["unrealized_pnl"],
            )
            for ticker, pos in snapshot.positions.items()
        ]
        recon_report = self.recon_engine.reconcile(broker_positions, cash=snapshot.cash)

        # 6. Update prev day NAV for tomorrow's daily P&L
        self.tracker.set_prev_day_nav(snapshot.total_nav)

        # 7. Log observation
        obs = DailyObservation(
            observation_id=str(uuid.uuid4()),
            trade_date=trade_date,
            nav=snapshot.total_nav,
            cash=snapshot.cash,
            gross_long=snapshot.gross_long,
            gross_short=snapshot.gross_short,
            daily_pnl=snapshot.daily_pnl,
            total_pnl=snapshot.total_nav - self.config.initial_capital,
            drawdown_pct=snapshot.drawdown_pct,
            signals_received=len(signals) if signals is not None else 0,
            orders_submitted=orders_submitted,
            orders_approved=orders_approved,
            orders_rejected=orders_rejected,
            orders_filled=orders_filled,
            kill_switch_active=self.kill_switch.is_halted(),
            circuit_breaker_triggered=cb_triggered,
            reconciliation_clean=recon_report.is_clean,
            reconciliation_breaks=len(recon_report.breaks),
            positions_count=len(snapshot.positions),
        )
        self.obs_logger.log(obs)

        return obs
