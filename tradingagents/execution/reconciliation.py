"""Broker Reconciliation & Position Tracker.

Compares internal paper trading positions against a broker snapshot
(or simulated broker state) and identifies discrepancies:

Reconciliation checks:
  - Missing positions (internal has, broker doesn't)
  - Extra positions (broker has, internal doesn't)
  - Quantity mismatches (same ticker, different quantity)
  - Price staleness (last price > N hours old)
  - Unrealized P&L drift (internal vs broker calculation differs)

Position tracker:
  - Mark-to-market all positions with latest prices
  - Compute portfolio NAV, gross/net exposure
  - Track daily P&L and running peak NAV (for drawdown monitoring)
  - Generate position summary report
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BrokerPosition:
    """A position as reported by the broker (or simulated broker)."""
    ticker: str
    quantity: float
    avg_cost: float
    last_price: float
    market_value: float
    unrealized_pnl: float


@dataclass
class ReconciliationBreak:
    """A discrepancy between internal and broker positions."""
    ticker: str
    break_type: str          # "missing_internal" | "missing_broker" | "qty_mismatch" | "pnl_drift"
    internal_qty: Optional[float]
    broker_qty: Optional[float]
    internal_pnl: Optional[float]
    broker_pnl: Optional[float]
    severity: str            # "critical" | "warning" | "info"
    message: str


@dataclass
class ReconciliationReport:
    """Full reconciliation report."""
    run_at: str
    total_internal_positions: int
    total_broker_positions: int
    breaks: List[ReconciliationBreak]
    is_clean: bool
    internal_nav: float
    broker_nav: float
    nav_difference: float
    nav_difference_pct: float

    def summary(self) -> str:
        lines = [
            f"=== Reconciliation Report ({self.run_at}) ===",
            f"Internal positions: {self.total_internal_positions}",
            f"Broker positions:   {self.total_broker_positions}",
            f"Breaks found:       {len(self.breaks)}",
            f"Internal NAV:       ${self.internal_nav:,.2f}",
            f"Broker NAV:         ${self.broker_nav:,.2f}",
            f"NAV difference:     ${self.nav_difference:,.2f} ({self.nav_difference_pct:.2%})",
            f"Status:             {'✓ CLEAN' if self.is_clean else '✗ BREAKS FOUND'}",
        ]
        if self.breaks:
            lines.append("\nBreaks:")
            for b in self.breaks:
                lines.append(f"  [{b.severity.upper()}] {b.ticker}: {b.message}")
        return "\n".join(lines)


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    snapshot_at: str
    cash: float
    positions: Dict[str, Dict]
    gross_long: float
    gross_short: float
    net_exposure: float
    total_nav: float
    peak_nav: float
    drawdown_pct: float
    daily_pnl: float
    total_realized_pnl: float
    total_unrealized_pnl: float

    def summary(self) -> str:
        lines = [
            f"=== Portfolio Snapshot ({self.snapshot_at}) ===",
            f"Cash:               ${self.cash:,.2f}",
            f"Total NAV:          ${self.total_nav:,.2f}",
            f"Peak NAV:           ${self.peak_nav:,.2f}",
            f"Drawdown:           {self.drawdown_pct:.2%}",
            f"Daily P&L:          ${self.daily_pnl:+,.2f}",
            f"Realized P&L:       ${self.total_realized_pnl:+,.2f}",
            f"Unrealized P&L:     ${self.total_unrealized_pnl:+,.2f}",
            f"Gross Long:         ${self.gross_long:,.2f}",
            f"Gross Short:        ${self.gross_short:,.2f}",
            f"Net Exposure:       ${self.net_exposure:+,.2f}",
            f"Open Positions:     {len(self.positions)}",
        ]
        if self.positions:
            lines.append("\nPositions:")
            for ticker, pos in self.positions.items():
                qty = pos.get("quantity", 0)
                mv = pos.get("market_value", 0)
                pnl = pos.get("unrealized_pnl", 0)
                lines.append(f"  {ticker:8s} qty={qty:+.0f}  MV=${mv:,.0f}  uPnL=${pnl:+,.0f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Position Tracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """Mark-to-market position tracker with NAV and drawdown monitoring.

    Usage:
        tracker = PositionTracker(db_path="paper_trading.db", initial_cash=100_000)
        tracker.update_prices({"AAPL": 182.50, "MSFT": 415.20})
        snapshot = tracker.get_snapshot()
        print(snapshot.summary())
    """

    def __init__(self, db_path: str = "paper_trading.db", initial_cash: float = 100_000.0):
        self.db_path = db_path
        self.initial_cash = initial_cash
        self._peak_nav: float = initial_cash
        self._prev_day_nav: float = initial_cash

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update last prices for all positions.

        Args:
            prices: Dict of ticker -> current price.
        """
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for ticker, price in prices.items():
                conn.execute("""
                    UPDATE positions SET last_price=?, last_updated=?
                    WHERE ticker=?
                """, (price, now, ticker.upper()))

            # Recompute unrealized P&L for all positions
            rows = conn.execute(
                "SELECT ticker, quantity, avg_cost, last_price FROM positions WHERE quantity != 0"
            ).fetchall()

            for row in rows:
                ticker, qty, avg_cost, last_price = row
                if last_price is not None and avg_cost is not None:
                    if qty > 0:
                        upnl = qty * (last_price - avg_cost)
                    else:
                        upnl = abs(qty) * (avg_cost - last_price)
                    conn.execute(
                        "UPDATE positions SET unrealized_pnl=? WHERE ticker=?",
                        (upnl, ticker)
                    )

    def get_cash(self) -> float:
        """Compute remaining cash from fills."""
        with sqlite3.connect(self.db_path) as conn:
            # Cash = initial - (buys + commissions) + (sells)
            rows = conn.execute("""
                SELECT side, quantity, fill_price, commission
                FROM orders WHERE status='filled' AND fill_price IS NOT NULL
            """).fetchall()

        cash = self.initial_cash
        for side, qty, fill_price, commission in rows:
            notional = qty * fill_price
            if side in ("buy", "cover"):
                cash -= notional + commission
            elif side in ("sell", "short"):
                cash += notional - commission
        return cash

    def get_snapshot(self) -> PortfolioSnapshot:
        """Generate a current portfolio snapshot."""
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT ticker, quantity, avg_cost, last_price, unrealized_pnl, realized_pnl
                FROM positions WHERE quantity != 0
            """).fetchall()

        positions = {}
        gross_long = 0.0
        gross_short = 0.0
        total_unrealized = 0.0
        total_realized = 0.0

        for row in rows:
            ticker, qty, avg_cost, last_price, upnl, rpnl = row
            last_price = last_price or avg_cost or 0.0
            market_value = qty * last_price
            upnl = upnl or 0.0
            rpnl = rpnl or 0.0

            positions[ticker] = {
                "ticker": ticker,
                "quantity": qty,
                "avg_cost": avg_cost or 0.0,
                "last_price": last_price,
                "market_value": market_value,
                "unrealized_pnl": upnl,
                "realized_pnl": rpnl,
            }

            if qty > 0:
                gross_long += market_value
            else:
                gross_short += abs(market_value)

            total_unrealized += upnl
            total_realized += rpnl

        cash = self.get_cash()
        total_nav = cash + gross_long - gross_short
        net_exposure = gross_long - gross_short

        # Update peak NAV
        if total_nav > self._peak_nav:
            self._peak_nav = total_nav

        drawdown_pct = (self._peak_nav - total_nav) / self._peak_nav if self._peak_nav > 0 else 0.0
        daily_pnl = total_nav - self._prev_day_nav

        return PortfolioSnapshot(
            snapshot_at=now,
            cash=cash,
            positions=positions,
            gross_long=gross_long,
            gross_short=gross_short,
            net_exposure=net_exposure,
            total_nav=total_nav,
            peak_nav=self._peak_nav,
            drawdown_pct=max(0.0, drawdown_pct),
            daily_pnl=daily_pnl,
            total_realized_pnl=total_realized,
            total_unrealized_pnl=total_unrealized,
        )

    def set_prev_day_nav(self, nav: float) -> None:
        """Set previous day NAV for daily P&L calculation."""
        self._prev_day_nav = nav


# ---------------------------------------------------------------------------
# Reconciliation Engine
# ---------------------------------------------------------------------------

class ReconciliationEngine:
    """Compare internal positions against broker snapshot.

    Usage:
        engine = ReconciliationEngine(db_path="paper_trading.db")
        broker_positions = [BrokerPosition("AAPL", 100, 175.0, 182.5, 18250, 750)]
        report = engine.reconcile(broker_positions, cash=80_000)
        print(report.summary())
    """

    def __init__(
        self,
        db_path: str = "paper_trading.db",
        qty_tolerance: float = 0.01,
        pnl_tolerance_pct: float = 0.02,
        price_staleness_hours: float = 4.0,
    ):
        self.db_path = db_path
        self.qty_tolerance = qty_tolerance
        self.pnl_tolerance_pct = pnl_tolerance_pct
        self.price_staleness_hours = price_staleness_hours

    def reconcile(
        self,
        broker_positions: List[BrokerPosition],
        cash: Optional[float] = None,
    ) -> ReconciliationReport:
        """Run full reconciliation against broker snapshot.

        Args:
            broker_positions: List of positions as reported by broker.
            cash: Broker-reported cash balance.

        Returns:
            ReconciliationReport with all breaks.
        """
        now = datetime.utcnow().isoformat()
        internal = self._get_internal_positions()
        broker = {p.ticker.upper(): p for p in broker_positions}

        breaks = []

        # Check for missing internal positions
        for ticker, bp in broker.items():
            if ticker not in internal:
                breaks.append(ReconciliationBreak(
                    ticker=ticker,
                    break_type="missing_internal",
                    internal_qty=None,
                    broker_qty=bp.quantity,
                    internal_pnl=None,
                    broker_pnl=bp.unrealized_pnl,
                    severity="critical",
                    message=f"Broker has {bp.quantity} shares but internal has no position",
                ))

        # Check for extra internal positions
        for ticker, ip in internal.items():
            if ticker not in broker:
                breaks.append(ReconciliationBreak(
                    ticker=ticker,
                    break_type="missing_broker",
                    internal_qty=ip["quantity"],
                    broker_qty=None,
                    internal_pnl=ip.get("unrealized_pnl"),
                    broker_pnl=None,
                    severity="critical",
                    message=f"Internal has {ip['quantity']} shares but broker has no position",
                ))

        # Check quantity and P&L mismatches
        for ticker in set(internal.keys()) & set(broker.keys()):
            ip = internal[ticker]
            bp = broker[ticker]

            qty_diff = abs(ip["quantity"] - bp.quantity)
            if qty_diff > self.qty_tolerance:
                breaks.append(ReconciliationBreak(
                    ticker=ticker,
                    break_type="qty_mismatch",
                    internal_qty=ip["quantity"],
                    broker_qty=bp.quantity,
                    internal_pnl=ip.get("unrealized_pnl"),
                    broker_pnl=bp.unrealized_pnl,
                    severity="critical",
                    message=f"Qty mismatch: internal={ip['quantity']:.2f}, broker={bp.quantity:.2f} (diff={qty_diff:.2f})",
                ))

            # P&L drift check
            int_pnl = ip.get("unrealized_pnl", 0) or 0
            if abs(bp.unrealized_pnl) > 0:
                pnl_diff_pct = abs(int_pnl - bp.unrealized_pnl) / abs(bp.unrealized_pnl)
                if pnl_diff_pct > self.pnl_tolerance_pct:
                    breaks.append(ReconciliationBreak(
                        ticker=ticker,
                        break_type="pnl_drift",
                        internal_qty=ip["quantity"],
                        broker_qty=bp.quantity,
                        internal_pnl=int_pnl,
                        broker_pnl=bp.unrealized_pnl,
                        severity="warning",
                        message=f"P&L drift {pnl_diff_pct:.1%}: internal=${int_pnl:+,.0f}, broker=${bp.unrealized_pnl:+,.0f}",
                    ))

        # Check price staleness
        staleness_breaks = self._check_price_staleness()
        breaks.extend(staleness_breaks)

        # NAV comparison
        internal_nav = sum(
            p.get("market_value", 0) for p in internal.values()
        ) + (cash or 0)
        broker_nav = sum(bp.market_value for bp in broker_positions) + (cash or 0)
        nav_diff = internal_nav - broker_nav
        nav_diff_pct = nav_diff / broker_nav if broker_nav > 0 else 0.0

        return ReconciliationReport(
            run_at=now,
            total_internal_positions=len(internal),
            total_broker_positions=len(broker),
            breaks=breaks,
            is_clean=len(breaks) == 0,
            internal_nav=internal_nav,
            broker_nav=broker_nav,
            nav_difference=nav_diff,
            nav_difference_pct=nav_diff_pct,
        )

    def _get_internal_positions(self) -> Dict[str, Dict]:
        """Get all non-zero internal positions."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT ticker, quantity, avg_cost, last_price, unrealized_pnl, realized_pnl, last_updated
                FROM positions WHERE quantity != 0
            """).fetchall()

        positions = {}
        for row in rows:
            ticker = row[0]
            last_price = row[3] or row[2] or 0.0
            positions[ticker] = {
                "ticker": ticker,
                "quantity": row[1],
                "avg_cost": row[2],
                "last_price": last_price,
                "market_value": row[1] * last_price,
                "unrealized_pnl": row[4],
                "realized_pnl": row[5],
                "last_updated": row[6],
            }
        return positions

    def _check_price_staleness(self) -> List[ReconciliationBreak]:
        """Check for positions with stale prices."""
        breaks = []
        cutoff = (datetime.utcnow() - timedelta(hours=self.price_staleness_hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT ticker, quantity, last_updated
                FROM positions
                WHERE quantity != 0 AND (last_updated IS NULL OR last_updated < ?)
            """, (cutoff,)).fetchall()

        for ticker, qty, last_updated in rows:
            breaks.append(ReconciliationBreak(
                ticker=ticker,
                break_type="stale_price",
                internal_qty=qty,
                broker_qty=None,
                internal_pnl=None,
                broker_pnl=None,
                severity="warning",
                message=f"Price not updated since {last_updated or 'never'} (>{self.price_staleness_hours}h old)",
            ))

        return breaks
