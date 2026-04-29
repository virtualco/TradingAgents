"""Phase 4 Tests: Execution Engine — Order Manager, Kill Switch, Reconciliation, Observer."""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.execution.order_manager import (
    Order, OrderSide, OrderStatus, PaperOrderManager, PreTradeRiskConfig, RejectionReason
)
from tradingagents.execution.kill_switch import (
    CircuitBreakerConfig, CircuitBreakerType, HaltLevel, KillSwitchManager
)
from tradingagents.execution.reconciliation import (
    BrokerPosition, PortfolioSnapshot, PositionTracker, ReconciliationEngine
)
from tradingagents.execution.observer import (
    DailyObservation, DailyObserver, ObservationConfig, ObservationLogger
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_paper_trading.db")


@pytest.fixture
def tmp_ks_file(tmp_path):
    """Temporary kill switch state file."""
    return str(tmp_path / "kill_switch_state.json")


@pytest.fixture
def order_manager(tmp_db):
    """PaperOrderManager with tight risk limits for testing."""
    config = PreTradeRiskConfig(
        max_position_notional=20_000.0,
        max_concentration_pct=0.25,
        max_gross_exposure=100_000.0,
        daily_loss_limit_pct=0.10,
        max_adv_pct=0.05,
        signal_cooldown_hours=0,   # Disable cooldown for tests
        min_price=0.50,
        max_order_notional=15_000.0,
    )
    return PaperOrderManager(db_path=tmp_db, config=config, commission_pct=0.001)


@pytest.fixture
def kill_switch(tmp_ks_file):
    """KillSwitchManager with low thresholds for testing."""
    config = CircuitBreakerConfig(
        max_drawdown_pct=0.05,
        max_daily_loss_pct=0.03,
        max_consecutive_losses=3,
        max_signals_per_hour=5,
        max_schema_failures=2,
        auto_reset_hours=1,
    )
    return KillSwitchManager(state_file=tmp_ks_file, config=config)


@pytest.fixture
def position_tracker(tmp_db):
    """PositionTracker with $100k initial capital."""
    return PositionTracker(db_path=tmp_db, initial_cash=100_000.0)


@pytest.fixture
def recon_engine(tmp_db):
    """ReconciliationEngine."""
    return ReconciliationEngine(db_path=tmp_db, qty_tolerance=0.01, pnl_tolerance_pct=0.05)


# ---------------------------------------------------------------------------
# Order Manager Tests
# ---------------------------------------------------------------------------

class TestPaperOrderManager:

    def test_submit_buy_order_approved(self, order_manager):
        """A valid BUY order within all limits should be approved."""
        order = order_manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            signal_id="sig-001",
            conviction=0.8,
            current_price=180.0,
            portfolio_nav=100_000.0,
        )
        assert order.status == OrderStatus.APPROVED
        assert order.ticker == "AAPL"
        assert order.notional == pytest.approx(9_000.0)
        assert order.commission == pytest.approx(9.0, rel=0.01)

    def test_kill_switch_rejects_order(self, order_manager):
        """Kill switch active should reject all orders."""
        order = order_manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            signal_id="sig-002",
            conviction=0.9,
            current_price=180.0,
            portfolio_nav=100_000.0,
            kill_switch_active=True,
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.KILL_SWITCH.value

    def test_position_limit_rejection(self, order_manager):
        """Order exceeding max position notional should be rejected."""
        order = order_manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=200,           # 200 * 180 = 36,000 > max_order_notional 15,000
            signal_id="sig-003",
            conviction=0.9,
            current_price=180.0,
            portfolio_nav=100_000.0,
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.POSITION_LIMIT.value

    def test_concentration_limit_rejection(self, order_manager):
        """Order exceeding concentration limit should be rejected."""
        # 80 * 180 = 14,400 / 50,000 NAV = 28.8% > 25% limit
        order = order_manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=80,
            signal_id="sig-004",
            conviction=0.9,
            current_price=180.0,
            portfolio_nav=50_000.0,
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.CONCENTRATION_LIMIT.value

    def test_daily_loss_limit_rejection(self, order_manager):
        """Order when daily loss exceeds limit should be rejected."""
        order = order_manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            signal_id="sig-005",
            conviction=0.9,
            current_price=180.0,
            portfolio_nav=100_000.0,
            daily_pnl=-12_000.0,   # 12% loss > 10% limit
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.DAILY_LOSS_LIMIT.value

    def test_liquidity_check_rejection(self, order_manager):
        """Order exceeding ADV limit should be rejected."""
        # 1000 shares / 10,000 ADV = 10% > 5% limit
        order = order_manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=1000,
            signal_id="sig-006",
            conviction=0.9,
            current_price=10.0,    # Low price to keep notional within limits
            portfolio_nav=100_000.0,
            adv_30d=10_000.0,
        )
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == RejectionReason.LIQUIDITY.value

    def test_min_price_rejection(self, order_manager):
        """Order on sub-penny stock should be rejected."""
        order = order_manager.submit_order(
            ticker="PENNY",
            side=OrderSide.BUY,
            quantity=100,
            signal_id="sig-007",
            conviction=0.9,
            current_price=0.10,    # Below min_price=0.50
            portfolio_nav=100_000.0,
        )
        assert order.status == OrderStatus.REJECTED

    def test_fill_order(self, order_manager):
        """Filling an approved order should update its status."""
        order = order_manager.submit_order(
            ticker="MSFT",
            side=OrderSide.BUY,
            quantity=20,
            signal_id="sig-008",
            conviction=0.75,
            current_price=415.0,
            portfolio_nav=200_000.0,
        )
        assert order.status == OrderStatus.APPROVED

        filled = order_manager.fill_order(order.order_id, fill_price=416.0)
        assert filled is not None
        assert filled.status == OrderStatus.FILLED
        assert filled.fill_price == 416.0

    def test_cancel_order(self, order_manager):
        """Cancelling an approved order should update its status."""
        order = order_manager.submit_order(
            ticker="GOOGL",
            side=OrderSide.BUY,
            quantity=5,
            signal_id="sig-009",
            conviction=0.7,
            current_price=170.0,
            portfolio_nav=100_000.0,
        )
        assert order.status == OrderStatus.APPROVED
        cancelled = order_manager.cancel_order(order.order_id)
        assert cancelled is True

        retrieved = order_manager.get_order(order.order_id)
        assert retrieved.status == OrderStatus.CANCELLED

    def test_get_order_history(self, order_manager):
        """Order history should return all submitted orders."""
        for i in range(3):
            order_manager.submit_order(
                ticker="TSLA",
                side=OrderSide.BUY,
                quantity=5,
                signal_id=f"sig-hist-{i}",
                conviction=0.6,
                current_price=250.0,
                portfolio_nav=100_000.0,
            )
        history = order_manager.get_order_history(ticker="TSLA")
        assert len(history) == 3

    def test_rejection_stats(self, order_manager):
        """Rejection stats should count by reason."""
        # Submit one order that will be rejected by kill switch
        order_manager.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            signal_id="sig-rej-1",
            conviction=0.9,
            current_price=180.0,
            portfolio_nav=100_000.0,
            kill_switch_active=True,
        )
        stats = order_manager.get_rejection_stats()
        assert stats.get(RejectionReason.KILL_SWITCH.value, 0) >= 1

    def test_sell_order_updates_position(self, order_manager):
        """Selling shares should reduce position quantity."""
        # Buy first
        buy = order_manager.submit_order(
            ticker="NVDA",
            side=OrderSide.BUY,
            quantity=30,
            signal_id="sig-buy-nvda",
            conviction=0.85,
            current_price=450.0,
            portfolio_nav=500_000.0,
        )
        order_manager.fill_order(buy.order_id, fill_price=450.0)

        # Then sell
        sell = order_manager.submit_order(
            ticker="NVDA",
            side=OrderSide.SELL,
            quantity=10,
            signal_id="sig-sell-nvda",
            conviction=0.85,
            current_price=460.0,
            portfolio_nav=500_000.0,
        )
        order_manager.fill_order(sell.order_id, fill_price=460.0)

        positions = order_manager.get_positions()
        assert "NVDA" in positions
        assert positions["NVDA"]["quantity"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Kill Switch Tests
# ---------------------------------------------------------------------------

class TestKillSwitch:

    def test_initial_state_not_halted(self, kill_switch):
        """Kill switch should not be halted initially."""
        assert not kill_switch.is_halted()
        assert not kill_switch.is_halted("AAPL")

    def test_manual_halt_system(self, kill_switch):
        """Manual system halt should block all trading."""
        kill_switch.manual_halt("Risk review")
        assert kill_switch.is_halted()
        assert kill_switch.is_halted("AAPL")

    def test_manual_halt_ticker(self, kill_switch):
        """Ticker halt should only block that ticker."""
        kill_switch.halt_ticker("AAPL", "Earnings halt")
        assert kill_switch.is_halted("AAPL")
        assert not kill_switch.is_halted("MSFT")
        assert not kill_switch.is_halted()

    def test_resume_ticker(self, kill_switch):
        """Resuming a ticker should unblock it."""
        kill_switch.halt_ticker("TSLA")
        kill_switch.resume_ticker("TSLA")
        assert not kill_switch.is_halted("TSLA")

    def test_drawdown_circuit_breaker(self, kill_switch):
        """Drawdown exceeding threshold should trigger circuit breaker."""
        triggered = kill_switch.check_drawdown(current_nav=93_000, peak_nav=100_000)
        assert triggered is True
        assert kill_switch.is_halted()
        assert kill_switch.state.halt_level == HaltLevel.CIRCUIT_BREAKER

    def test_drawdown_below_threshold(self, kill_switch):
        """Drawdown below threshold should not trigger circuit breaker."""
        triggered = kill_switch.check_drawdown(current_nav=97_000, peak_nav=100_000)
        assert triggered is False
        assert not kill_switch.is_halted()

    def test_daily_loss_circuit_breaker(self, kill_switch):
        """Daily loss exceeding threshold should trigger circuit breaker."""
        triggered = kill_switch.check_daily_loss(daily_pnl=-4_000, portfolio_nav=100_000)
        assert triggered is True
        assert kill_switch.is_halted()

    def test_consecutive_losses_circuit_breaker(self, kill_switch):
        """3 consecutive losses should trigger circuit breaker."""
        kill_switch.record_signal_outcome("AAPL", was_profitable=False)
        kill_switch.record_signal_outcome("MSFT", was_profitable=False)
        triggered = kill_switch.record_signal_outcome("GOOGL", was_profitable=False)
        assert triggered is True
        assert kill_switch.is_halted()

    def test_profitable_signal_resets_consecutive_losses(self, kill_switch):
        """A profitable signal should reset the consecutive loss counter."""
        kill_switch.record_signal_outcome("AAPL", was_profitable=False)
        kill_switch.record_signal_outcome("MSFT", was_profitable=False)
        kill_switch.record_signal_outcome("GOOGL", was_profitable=True)
        assert kill_switch.state.consecutive_losses == 0

    def test_schema_failure_circuit_breaker(self, kill_switch):
        """2 schema failures should trigger circuit breaker."""
        kill_switch.record_schema_failure()
        triggered = kill_switch.record_schema_failure()
        assert triggered is True
        assert kill_switch.is_halted()

    def test_reset_clears_halt(self, kill_switch):
        """Reset should clear all halts."""
        kill_switch.manual_halt("Test halt")
        kill_switch.reset("Test reset")
        assert not kill_switch.is_halted()
        assert kill_switch.state.halt_level == HaltLevel.NONE

    def test_halt_history_recorded(self, kill_switch):
        """Halt events should be recorded in history."""
        kill_switch.manual_halt("First halt")
        kill_switch.reset()
        kill_switch.check_drawdown(current_nav=90_000, peak_nav=100_000)
        assert len(kill_switch.state.halt_history) >= 2

    def test_state_persistence(self, tmp_ks_file):
        """Kill switch state should persist across instances."""
        ks1 = KillSwitchManager(state_file=tmp_ks_file)
        ks1.manual_halt("Persistent halt")

        ks2 = KillSwitchManager(state_file=tmp_ks_file)
        assert ks2.is_halted()
        assert ks2.state.halt_reason == "Persistent halt"


# ---------------------------------------------------------------------------
# Position Tracker & Reconciliation Tests
# ---------------------------------------------------------------------------

class TestPositionTracker:

    def test_initial_cash(self, tmp_db):
        """Initial cash should equal initial capital."""
        # Initialize DB via order manager first
        PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(signal_cooldown_hours=0))
        tracker = PositionTracker(db_path=tmp_db, initial_cash=100_000.0)
        cash = tracker.get_cash()
        assert cash == pytest.approx(100_000.0)

    def test_snapshot_empty_portfolio(self, tmp_db):
        """Empty portfolio snapshot should have correct NAV."""
        PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(signal_cooldown_hours=0))
        tracker = PositionTracker(db_path=tmp_db, initial_cash=100_000.0)
        snapshot = tracker.get_snapshot()
        assert snapshot.total_nav == pytest.approx(100_000.0)
        assert snapshot.gross_long == 0.0
        assert snapshot.positions == {}

    def test_update_prices(self, tmp_db):
        """Updating prices should mark positions to market."""
        # Use order manager to create a position
        om = PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(
            signal_cooldown_hours=0, max_order_notional=100_000, max_position_notional=100_000
        ))
        order = om.submit_order("AAPL", OrderSide.BUY, 100, "sig-x", 0.8, 180.0, 500_000.0)
        om.fill_order(order.order_id, 180.0)

        tracker = PositionTracker(db_path=tmp_db, initial_cash=500_000.0)
        tracker.update_prices({"AAPL": 190.0})
        snapshot = tracker.get_snapshot()

        assert "AAPL" in snapshot.positions
        assert snapshot.positions["AAPL"]["last_price"] == pytest.approx(190.0)
        assert snapshot.positions["AAPL"]["unrealized_pnl"] == pytest.approx(1000.0)  # 100 * (190-180)

    def test_drawdown_calculation(self, tmp_db):
        """Drawdown should be computed from peak NAV."""
        PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(signal_cooldown_hours=0))
        tracker = PositionTracker(db_path=tmp_db, initial_cash=100_000.0)
        tracker._peak_nav = 110_000.0
        tracker._prev_day_nav = 100_000.0
        snapshot = tracker.get_snapshot()
        # NAV = 100k cash, peak = 110k → drawdown ≈ 9.09%
        assert snapshot.drawdown_pct == pytest.approx((110_000 - 100_000) / 110_000, rel=0.01)


class TestReconciliationEngine:

    def test_clean_reconciliation(self, tmp_db, recon_engine):
        """Identical internal and broker positions should be clean."""
        om = PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(
            signal_cooldown_hours=0, max_order_notional=100_000, max_position_notional=100_000
        ))
        order = om.submit_order("AAPL", OrderSide.BUY, 100, "sig-r1", 0.8, 180.0, 500_000.0)
        om.fill_order(order.order_id, 180.0)

        broker_positions = [BrokerPosition("AAPL", 100, 180.0, 180.0, 18_000.0, 0.0)]
        report = recon_engine.reconcile(broker_positions, cash=482_000.0)

        # Only stale price warnings, no critical breaks
        critical_breaks = [b for b in report.breaks if b.severity == "critical"]
        assert len(critical_breaks) == 0
        assert report.total_internal_positions == 1
        assert report.total_broker_positions == 1

    def test_missing_internal_position(self, tmp_db):
        """Broker position not in internal should be a critical break."""
        # Init DB
        PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(signal_cooldown_hours=0))
        engine = ReconciliationEngine(db_path=tmp_db)
        broker_positions = [BrokerPosition("TSLA", 50, 250.0, 260.0, 13_000.0, 500.0)]
        report = engine.reconcile(broker_positions)

        critical_breaks = [b for b in report.breaks if b.break_type == "missing_internal"]
        assert len(critical_breaks) == 1
        assert critical_breaks[0].ticker == "TSLA"

    def test_quantity_mismatch(self, tmp_db, recon_engine):
        """Quantity mismatch between internal and broker should be flagged."""
        om = PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(
            signal_cooldown_hours=0, max_order_notional=100_000, max_position_notional=100_000
        ))
        order = om.submit_order("MSFT", OrderSide.BUY, 100, "sig-r2", 0.8, 400.0, 500_000.0)
        om.fill_order(order.order_id, 400.0)

        # Broker reports 90 shares (mismatch)
        broker_positions = [BrokerPosition("MSFT", 90, 400.0, 405.0, 36_450.0, 450.0)]
        report = recon_engine.reconcile(broker_positions)

        qty_breaks = [b for b in report.breaks if b.break_type == "qty_mismatch"]
        assert len(qty_breaks) == 1
        assert qty_breaks[0].internal_qty == pytest.approx(100.0)
        assert qty_breaks[0].broker_qty == pytest.approx(90.0)

    def test_reconciliation_report_summary(self, tmp_db):
        """Summary string should be generated without error."""
        PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(signal_cooldown_hours=0))
        engine = ReconciliationEngine(db_path=tmp_db)
        report = engine.reconcile([])
        summary = report.summary()
        assert "Reconciliation Report" in summary
        assert "CLEAN" in summary


# ---------------------------------------------------------------------------
# Observation Logger Tests
# ---------------------------------------------------------------------------

class TestObservationLogger:

    def test_log_and_retrieve(self, tmp_db):
        """Logged observations should be retrievable."""
        logger_obj = ObservationLogger(db_path=tmp_db)
        obs = DailyObservation(
            observation_id=str(uuid.uuid4()),
            trade_date="2026-01-15",
            nav=102_000.0,
            cash=80_000.0,
            gross_long=22_000.0,
            gross_short=0.0,
            daily_pnl=2_000.0,
            total_pnl=2_000.0,
            drawdown_pct=0.0,
            signals_received=5,
            orders_submitted=3,
            orders_approved=2,
            orders_rejected=1,
            orders_filled=2,
            kill_switch_active=False,
            circuit_breaker_triggered=False,
            reconciliation_clean=True,
            reconciliation_breaks=0,
            positions_count=2,
        )
        logger_obj.log(obs)
        retrieved = logger_obj.get_observations()
        assert len(retrieved) == 1
        assert retrieved[0].trade_date == "2026-01-15"
        assert retrieved[0].nav == pytest.approx(102_000.0)

    def test_summary_no_data(self, tmp_db):
        """Summary with no data should return not-ready."""
        logger_obj = ObservationLogger(db_path=tmp_db)
        summary = logger_obj.get_summary()
        assert summary.ready_for_live is False
        assert summary.total_days == 0

    def test_summary_with_data(self, tmp_db):
        """Summary with sufficient data should compute metrics."""
        # Init orders table via order manager
        PaperOrderManager(db_path=tmp_db, config=PreTradeRiskConfig(signal_cooldown_hours=0))
        logger_obj = ObservationLogger(db_path=tmp_db)
        # Log 10 days of observations
        nav = 100_000.0
        for i in range(10):
            nav += 500.0  # Steady gains
            obs = DailyObservation(
                observation_id=str(uuid.uuid4()),
                trade_date=f"2026-01-{i+1:02d}",
                nav=nav,
                cash=nav * 0.8,
                gross_long=nav * 0.2,
                gross_short=0.0,
                daily_pnl=500.0,
                total_pnl=(i + 1) * 500.0,
                drawdown_pct=0.0,
                signals_received=3,
                orders_submitted=2,
                orders_approved=2,
                orders_rejected=0,
                orders_filled=2,
                kill_switch_active=False,
                circuit_breaker_triggered=False,
                reconciliation_clean=True,
                reconciliation_breaks=0,
                positions_count=1,
            )
            logger_obj.log(obs)

        summary = logger_obj.get_summary()
        assert summary.total_days == 10
        assert summary.total_return > 0
        assert summary.final_nav > summary.initial_nav
        # Not ready — too few days
        assert summary.ready_for_live is False
        assert any("90" in note for note in summary.readiness_notes)


# ---------------------------------------------------------------------------
# Daily Observer Integration Test
# ---------------------------------------------------------------------------

class TestDailyObserver:

    def test_daily_cycle_no_signals(self, tmp_db, tmp_ks_file):
        """Daily cycle with no signals should complete without error."""
        config = ObservationConfig(
            db_path=tmp_db,
            initial_capital=100_000.0,
            min_conviction=0.5,
        )
        observer = DailyObserver(config=config)
        observer.kill_switch.state_file = tmp_ks_file

        obs = observer.run_daily_cycle(
            signals=pd.DataFrame(),
            prices={"AAPL": 180.0},
            trade_date="2026-01-15",
        )
        assert obs.trade_date == "2026-01-15"
        assert obs.orders_submitted == 0
        assert obs.kill_switch_active is False

    def test_daily_cycle_with_signals(self, tmp_db, tmp_ks_file):
        """Daily cycle with valid signals should submit and fill orders."""
        config = ObservationConfig(
            db_path=tmp_db,
            initial_capital=100_000.0,
            min_conviction=0.5,
            max_position_size_pct=0.05,
        )
        observer = DailyObserver(config=config)
        observer.kill_switch.state_file = tmp_ks_file

        signals = pd.DataFrame([
            {"ticker": "AAPL", "direction": "long", "conviction": 0.8, "signal_id": "sig-obs-1"},
        ])
        obs = observer.run_daily_cycle(
            signals=signals,
            prices={"AAPL": 180.0},
            trade_date="2026-01-16",
        )
        assert obs.signals_received == 1
        assert obs.orders_submitted >= 1
        assert obs.orders_filled >= 1

    def test_daily_cycle_kill_switch_halts_trading(self, tmp_db, tmp_ks_file):
        """Kill switch active should prevent any orders from being filled."""
        config = ObservationConfig(db_path=tmp_db, initial_capital=100_000.0)
        observer = DailyObserver(config=config)
        observer.kill_switch.state_file = tmp_ks_file
        observer.kill_switch.manual_halt("Test halt")

        signals = pd.DataFrame([
            {"ticker": "AAPL", "direction": "long", "conviction": 0.9, "signal_id": "sig-obs-2"},
        ])
        obs = observer.run_daily_cycle(
            signals=signals,
            prices={"AAPL": 180.0},
            trade_date="2026-01-17",
        )
        assert obs.kill_switch_active is True
        assert obs.orders_filled == 0
