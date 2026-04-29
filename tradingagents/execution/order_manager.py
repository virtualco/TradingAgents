"""Paper Trading Order Manager.

Manages the full lifecycle of paper trading orders with institutional-grade
pre-trade risk checks:

Pre-trade checks (in order):
  1. Kill switch — abort if system-wide halt is active
  2. Position limit — max shares / notional per ticker
  3. Concentration limit — max % of portfolio in single name
  4. Gross exposure — max total long + short notional
  5. Net exposure — max net long/short notional
  6. Daily loss limit — halt if daily P&L exceeds threshold
  7. Liquidity check — order size vs average daily volume (ADV)
  8. Duplicate signal guard — no repeat signal within cooldown window

Order states: PENDING → APPROVED | REJECTED → FILLED | CANCELLED
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FILLED = "filled"
    CANCELLED = "cancelled"


class RejectionReason(str, Enum):
    KILL_SWITCH = "kill_switch"
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    LIQUIDITY = "liquidity"
    DUPLICATE_SIGNAL = "duplicate_signal"
    INSUFFICIENT_CASH = "insufficient_cash"
    UNKNOWN_TICKER = "unknown_ticker"


@dataclass
class PreTradeRiskConfig:
    """Configuration for pre-trade risk checks."""
    max_position_notional: float = 50_000.0      # Max $ per ticker
    max_concentration_pct: float = 0.20           # Max 20% of portfolio in one name
    max_gross_exposure: float = 200_000.0         # Max total long + short notional
    max_net_exposure_pct: float = 1.0             # Max net exposure as % of NAV
    daily_loss_limit_pct: float = 0.05            # Halt if daily loss > 5% NAV
    max_adv_pct: float = 0.01                     # Max 1% of 30-day ADV per order
    signal_cooldown_hours: int = 24               # Min hours between same-ticker signals
    min_price: float = 1.0                        # Min stock price (avoid penny stocks)
    max_order_notional: float = 25_000.0          # Max single order size


@dataclass
class Order:
    """A paper trading order."""
    order_id: str
    ticker: str
    side: OrderSide
    quantity: float
    limit_price: Optional[float]
    signal_id: str
    conviction: float
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: Optional[str] = None
    fill_price: Optional[float] = None
    fill_date: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notional: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["status"] = self.status.value
        return d


@dataclass
class PreTradeCheckResult:
    """Result of a pre-trade risk check."""
    passed: bool
    reason: Optional[RejectionReason] = None
    message: str = ""
    risk_metrics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Order Manager
# ---------------------------------------------------------------------------

class PaperOrderManager:
    """Paper trading order manager with full pre-trade risk stack.

    Usage:
        manager = PaperOrderManager(db_path="paper_trading.db")
        result = manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            signal_id="sig-123",
            conviction=0.75,
            current_price=180.0,
            portfolio_nav=100_000.0,
        )
    """

    def __init__(
        self,
        db_path: str = "paper_trading.db",
        config: Optional[PreTradeRiskConfig] = None,
        commission_pct: float = 0.001,
    ):
        self.db_path = db_path
        self.config = config or PreTradeRiskConfig()
        self.commission_pct = commission_pct
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    limit_price REAL,
                    signal_id TEXT NOT NULL,
                    conviction REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    rejection_reason TEXT,
                    fill_price REAL,
                    fill_date TEXT,
                    created_at TEXT NOT NULL,
                    notional REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    quantity REAL NOT NULL DEFAULT 0,
                    avg_cost REAL NOT NULL DEFAULT 0,
                    last_price REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    last_updated TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_pnl (
                    trade_date TEXT PRIMARY KEY,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);
            """)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        signal_id: str,
        conviction: float,
        current_price: float,
        portfolio_nav: float,
        adv_30d: Optional[float] = None,
        kill_switch_active: bool = False,
        daily_pnl: float = 0.0,
    ) -> Order:
        """Submit an order through the full pre-trade risk stack.

        Args:
            ticker: Stock ticker symbol.
            side: BUY, SELL, SHORT, or COVER.
            quantity: Number of shares.
            signal_id: ID of the originating signal.
            conviction: Agent conviction score (0-1).
            current_price: Current market price.
            portfolio_nav: Current portfolio net asset value.
            adv_30d: 30-day average daily volume (shares). None skips liquidity check.
            kill_switch_active: Whether the system-wide kill switch is active.
            daily_pnl: Today's realized + unrealized P&L (negative = loss).

        Returns:
            Order object with status APPROVED or REJECTED.
        """
        order_id = str(uuid.uuid4())
        notional = quantity * current_price

        order = Order(
            order_id=order_id,
            ticker=ticker.upper(),
            side=side,
            quantity=quantity,
            limit_price=current_price,
            signal_id=signal_id,
            conviction=conviction,
            notional=notional,
        )

        # Run pre-trade checks
        check = self._run_pretrade_checks(
            order=order,
            current_price=current_price,
            portfolio_nav=portfolio_nav,
            adv_30d=adv_30d,
            kill_switch_active=kill_switch_active,
            daily_pnl=daily_pnl,
        )

        if check.passed:
            order.status = OrderStatus.APPROVED
            order.commission = notional * self.commission_pct
        else:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = check.reason.value if check.reason else "unknown"
            logger.warning(f"Order rejected [{ticker}]: {check.message}")

        self._save_order(order)
        return order

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        fill_date: Optional[str] = None,
    ) -> Optional[Order]:
        """Mark an approved order as filled.

        Args:
            order_id: The order to fill.
            fill_price: Actual fill price.
            fill_date: Fill date (defaults to today).

        Returns:
            Updated Order or None if not found.
        """
        order = self.get_order(order_id)
        if order is None or order.status != OrderStatus.APPROVED:
            return None

        fill_date = fill_date or date.today().isoformat()
        order.fill_price = fill_price
        order.fill_date = fill_date
        order.status = OrderStatus.FILLED
        order.commission = order.quantity * fill_price * self.commission_pct

        # Update position
        self._update_position(order)

        # Save
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE orders SET status=?, fill_price=?, fill_date=?, commission=?
                WHERE order_id=?
            """, (order.status.value, fill_price, fill_date, order.commission, order_id))

        logger.info(f"Order filled: {order.ticker} {order.side.value} {order.quantity} @ {fill_price:.2f}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending or approved order."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "UPDATE orders SET status='cancelled' WHERE order_id=? AND status IN ('pending','approved')",
                (order_id,)
            ).rowcount
        return rows > 0

    # ------------------------------------------------------------------
    # Pre-trade risk checks
    # ------------------------------------------------------------------

    def _run_pretrade_checks(
        self,
        order: Order,
        current_price: float,
        portfolio_nav: float,
        adv_30d: Optional[float],
        kill_switch_active: bool,
        daily_pnl: float,
    ) -> PreTradeCheckResult:
        """Run all pre-trade checks in priority order."""

        # 1. Kill switch
        if kill_switch_active:
            return PreTradeCheckResult(
                passed=False,
                reason=RejectionReason.KILL_SWITCH,
                message="System-wide kill switch is active — all orders halted",
            )

        # 2. Min price
        if current_price < self.config.min_price:
            return PreTradeCheckResult(
                passed=False,
                reason=RejectionReason.UNKNOWN_TICKER,
                message=f"Price {current_price:.2f} below minimum {self.config.min_price:.2f}",
            )

        # 3. Order notional limit
        if order.notional > self.config.max_order_notional:
            return PreTradeCheckResult(
                passed=False,
                reason=RejectionReason.POSITION_LIMIT,
                message=f"Order notional ${order.notional:,.0f} exceeds max ${self.config.max_order_notional:,.0f}",
            )

        # 4. Position limit (existing + new)
        existing_notional = self._get_position_notional(order.ticker, current_price)
        if order.side in (OrderSide.BUY, OrderSide.SHORT):
            projected_notional = existing_notional + order.notional
            if projected_notional > self.config.max_position_notional:
                return PreTradeCheckResult(
                    passed=False,
                    reason=RejectionReason.POSITION_LIMIT,
                    message=f"Position notional ${projected_notional:,.0f} would exceed limit ${self.config.max_position_notional:,.0f}",
                )

        # 5. Concentration limit
        if portfolio_nav > 0:
            concentration = order.notional / portfolio_nav
            if concentration > self.config.max_concentration_pct:
                return PreTradeCheckResult(
                    passed=False,
                    reason=RejectionReason.CONCENTRATION_LIMIT,
                    message=f"Order concentration {concentration:.1%} exceeds limit {self.config.max_concentration_pct:.1%}",
                )

        # 6. Daily loss limit
        if portfolio_nav > 0 and daily_pnl < 0:
            loss_pct = abs(daily_pnl) / portfolio_nav
            if loss_pct >= self.config.daily_loss_limit_pct:
                return PreTradeCheckResult(
                    passed=False,
                    reason=RejectionReason.DAILY_LOSS_LIMIT,
                    message=f"Daily loss {loss_pct:.1%} exceeds limit {self.config.daily_loss_limit_pct:.1%} — trading halted",
                )

        # 7. Liquidity check
        if adv_30d is not None and adv_30d > 0:
            adv_pct = order.quantity / adv_30d
            if adv_pct > self.config.max_adv_pct:
                return PreTradeCheckResult(
                    passed=False,
                    reason=RejectionReason.LIQUIDITY,
                    message=f"Order size {adv_pct:.2%} of ADV exceeds limit {self.config.max_adv_pct:.2%}",
                )

        # 8. Duplicate signal guard
        if self._is_duplicate_signal(order.ticker, order.side):
            return PreTradeCheckResult(
                passed=False,
                reason=RejectionReason.DUPLICATE_SIGNAL,
                message=f"Duplicate signal for {order.ticker} within {self.config.signal_cooldown_hours}h cooldown",
            )

        return PreTradeCheckResult(
            passed=True,
            message="All pre-trade checks passed",
            risk_metrics={
                "notional": order.notional,
                "concentration_pct": order.notional / portfolio_nav if portfolio_nav > 0 else 0,
                "existing_position_notional": existing_notional,
            },
        )

    def _is_duplicate_signal(self, ticker: str, side: OrderSide) -> bool:
        """Check if a same-direction signal was submitted within the cooldown window."""
        cutoff = (datetime.utcnow() - timedelta(hours=self.config.signal_cooldown_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT COUNT(*) FROM orders
                WHERE ticker=? AND side=? AND status IN ('approved','filled')
                AND created_at >= ?
            """, (ticker, side.value, cutoff)).fetchone()
        return row[0] > 0

    def _get_position_notional(self, ticker: str, current_price: float) -> float:
        """Get current position notional for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT quantity FROM positions WHERE ticker=?", (ticker,)
            ).fetchone()
        if row:
            return abs(row[0]) * current_price
        return 0.0

    def _update_position(self, order: Order):
        """Update position table after a fill."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT quantity, avg_cost, realized_pnl FROM positions WHERE ticker=?",
                (order.ticker,)
            ).fetchone()

            if row is None:
                qty, avg_cost, realized_pnl = 0.0, 0.0, 0.0
            else:
                qty, avg_cost, realized_pnl = row

            fill_price = order.fill_price or 0.0

            if order.side == OrderSide.BUY:
                new_qty = qty + order.quantity
                new_avg_cost = (qty * avg_cost + order.quantity * fill_price) / new_qty if new_qty > 0 else fill_price
                new_realized = realized_pnl
            elif order.side == OrderSide.SELL:
                sold_qty = min(order.quantity, qty)
                new_qty = qty - sold_qty
                new_avg_cost = avg_cost
                new_realized = realized_pnl + sold_qty * (fill_price - avg_cost) - order.commission
            elif order.side == OrderSide.SHORT:
                new_qty = qty - order.quantity
                new_avg_cost = fill_price
                new_realized = realized_pnl
            elif order.side == OrderSide.COVER:
                covered_qty = min(order.quantity, abs(qty))
                new_qty = qty + covered_qty
                new_avg_cost = avg_cost
                new_realized = realized_pnl + covered_qty * (avg_cost - fill_price) - order.commission
            else:
                return

            conn.execute("""
                INSERT INTO positions (ticker, quantity, avg_cost, last_price, realized_pnl, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    quantity=excluded.quantity,
                    avg_cost=excluded.avg_cost,
                    last_price=excluded.last_price,
                    realized_pnl=excluded.realized_pnl,
                    last_updated=excluded.last_updated
            """, (order.ticker, new_qty, new_avg_cost, fill_price, new_realized,
                  datetime.utcnow().isoformat()))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> Optional[Order]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id=?", (order_id,)
            ).fetchone()
        return self._row_to_order(row) if row else None

    def get_open_orders(self) -> List[Order]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE status IN ('pending','approved')"
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def get_positions(self) -> Dict[str, Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM positions WHERE quantity != 0").fetchall()
        positions = {}
        for row in rows:
            ticker = row[0]
            positions[ticker] = {
                "ticker": ticker,
                "quantity": row[1],
                "avg_cost": row[2],
                "last_price": row[3],
                "unrealized_pnl": row[4],
                "realized_pnl": row[5],
                "last_updated": row[6],
            }
        return positions

    def get_order_history(self, ticker: Optional[str] = None, limit: int = 100) -> List[Order]:
        with sqlite3.connect(self.db_path) as conn:
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM orders WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
                    (ticker, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def get_rejection_stats(self) -> Dict[str, int]:
        """Get counts of rejections by reason."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT rejection_reason, COUNT(*) as cnt
                FROM orders WHERE status='rejected'
                GROUP BY rejection_reason
            """).fetchall()
        return {row[0]: row[1] for row in rows}

    def _save_order(self, order: Order):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO orders
                (order_id, ticker, side, quantity, limit_price, signal_id, conviction,
                 status, rejection_reason, fill_price, fill_date, created_at, notional,
                 commission, net_pnl, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                order.order_id, order.ticker, order.side.value, order.quantity,
                order.limit_price, order.signal_id, order.conviction,
                order.status.value, order.rejection_reason, order.fill_price,
                order.fill_date, order.created_at, order.notional,
                order.commission, order.net_pnl,
                json.dumps(order.metadata),
            ))

    def _row_to_order(self, row) -> Order:
        return Order(
            order_id=row[0],
            ticker=row[1],
            side=OrderSide(row[2]),
            quantity=row[3],
            limit_price=row[4],
            signal_id=row[5],
            conviction=row[6],
            status=OrderStatus(row[7]),
            rejection_reason=row[8],
            fill_price=row[9],
            fill_date=row[10],
            created_at=row[11],
            notional=row[12] or 0.0,
            commission=row[13] or 0.0,
            net_pnl=row[14] or 0.0,
            metadata=json.loads(row[15] or "{}"),
        )
