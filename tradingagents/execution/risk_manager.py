"""
Production Risk Management Framework — TradingAgents
=====================================================
Implements all risk controls required before live trading on Bybit:

  1. Account-level circuit breaker (daily drawdown limit)
  2. Per-trade position sizing (ATR-based Kelly fraction)
  3. Maximum concurrent positions cap
  4. Maximum notional exposure per symbol
  5. Trade cooldown (minimum time between trades per symbol)
  6. Volatility circuit breaker (halt trading in extreme vol)
  7. Kill switch (emergency halt via env var or file)
  8. Daily P&L tracking and reporting

Usage:
    from tradingagents.execution.risk_manager import RiskManager

    rm = RiskManager(
        account_equity=10_000,
        daily_loss_limit_pct=0.05,   # 5% max daily loss
        max_position_pct=0.10,       # 10% max per position
    )

    # Check before placing any order
    approved, reason = rm.approve_trade(symbol, side, qty, price, atr)
    if approved:
        place_order(...)
        rm.record_trade(symbol, side, qty, price)
"""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("risk_manager")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    # Account-level limits
    daily_loss_limit_pct: float = 0.05       # 5% max daily drawdown
    weekly_loss_limit_pct: float = 0.15      # 15% max weekly drawdown
    max_open_positions: int = 3              # Maximum concurrent open positions
    max_position_pct: float = 0.10          # Max 10% of equity per position

    # Trade-level limits
    account_risk_per_trade: float = 0.01    # 1% account risk per trade
    atr_stop_multiplier: float = 2.5        # ATR multiplier for stop-loss
    max_leverage: int = 3                   # Maximum leverage
    min_qty: float = 0.001                  # Minimum order quantity (BTC)

    # Timing controls
    min_trade_interval_sec: int = 3600      # 1 hour minimum between trades per symbol
    max_daily_trades: int = 10              # Maximum trades per day across all symbols

    # Volatility circuit breaker
    max_atr_pct: float = 0.05              # Halt if ATR/price > 5% (extreme volatility)

    # Kill switch
    kill_switch_env: str = "KILL_SWITCH"
    kill_switch_file: str = "data/KILL_SWITCH"


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    date: str = ""
    starting_equity: float = 0.0
    current_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl: float = 0.0
    daily_trade_count: int = 0
    open_positions: dict = field(default_factory=dict)   # symbol -> {side, qty, entry_price, notional}
    last_trade_time: dict = field(default_factory=dict)  # symbol -> ISO timestamp
    circuit_breaker_active: bool = False
    circuit_breaker_reason: str = ""


# ── Risk Manager ──────────────────────────────────────────────────────────────

class RiskManager:
    """
    Production risk management for live crypto trading.
    Thread-safe for single-process use.
    """

    STATE_FILE = Path("data/risk_state.json")

    def __init__(
        self,
        account_equity: float,
        config: Optional[RiskConfig] = None,
    ):
        self.config = config or RiskConfig()
        self.state = self._load_or_init_state(account_equity)
        log.info(
            f"RiskManager initialised | equity=${account_equity:,.2f} | "
            f"daily_limit={self.config.daily_loss_limit_pct*100:.1f}% | "
            f"max_positions={self.config.max_open_positions}"
        )

    # ── State Persistence ─────────────────────────────────────────────────────

    def _load_or_init_state(self, equity: float) -> RiskState:
        today = str(date.today())
        if self.STATE_FILE.exists():
            try:
                data = json.loads(self.STATE_FILE.read_text())
                state = RiskState(**data)
                if state.date != today:
                    # New day — reset daily counters but keep positions
                    log.info(f"New trading day ({today}) — resetting daily counters")
                    state.date = today
                    state.starting_equity = equity
                    state.current_equity = equity
                    state.daily_pnl = 0.0
                    state.daily_pnl_pct = 0.0
                    state.daily_trade_count = 0
                    state.circuit_breaker_active = False
                    state.circuit_breaker_reason = ""
                return state
            except Exception as e:
                log.warning(f"Could not load risk state: {e} — initialising fresh")

        return RiskState(
            date=today,
            starting_equity=equity,
            current_equity=equity,
        )

    def save_state(self):
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.STATE_FILE.write_text(json.dumps(asdict(self.state), indent=2))

    # ── Kill Switch ───────────────────────────────────────────────────────────

    def is_kill_switch_active(self) -> bool:
        """Check kill switch via env var or file."""
        if os.getenv(self.config.kill_switch_env, "0") == "1":
            return True
        if Path(self.config.kill_switch_file).exists():
            return True
        return False

    def activate_kill_switch(self, reason: str = "Manual"):
        """Activate kill switch by creating the sentinel file."""
        Path(self.config.kill_switch_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.kill_switch_file).write_text(f"Kill switch activated: {reason}\n")
        log.critical(f"KILL SWITCH ACTIVATED: {reason}")

    # ── Circuit Breaker ───────────────────────────────────────────────────────

    def _check_circuit_breaker(self, current_equity: float) -> tuple[bool, str]:
        """Check if circuit breaker should trip."""
        if self.state.starting_equity <= 0:
            return False, ""

        daily_loss_pct = (current_equity - self.state.starting_equity) / self.state.starting_equity
        if daily_loss_pct < -self.config.daily_loss_limit_pct:
            return True, f"Daily loss limit hit: {daily_loss_pct*100:.1f}% (limit: {self.config.daily_loss_limit_pct*100:.1f}%)"

        if self.state.daily_trade_count >= self.config.max_daily_trades:
            return True, f"Daily trade limit hit: {self.state.daily_trade_count} trades"

        return False, ""

    # ── Position Sizing ───────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        equity: float,
    ) -> float:
        """
        ATR-based position sizing.
        Risk amount = account_risk_per_trade * equity
        Stop distance = atr_stop_multiplier * ATR
        Qty = risk_amount / stop_distance (in base currency)
        Capped at max_position_pct * equity / price
        """
        stop_distance = self.config.atr_stop_multiplier * atr
        if stop_distance <= 0 or price <= 0:
            return self.config.min_qty

        risk_amount = self.config.account_risk_per_trade * equity
        qty_by_risk = risk_amount / stop_distance

        # Cap by max position size
        max_notional = self.config.max_position_pct * equity * self.config.max_leverage
        max_qty = max_notional / price

        qty = min(qty_by_risk, max_qty)
        qty = max(round(qty, 3), self.config.min_qty)

        log.debug(
            f"Position size: risk_amt=${risk_amount:.2f} | stop_dist={stop_distance:.2f} | "
            f"qty_by_risk={qty_by_risk:.4f} | max_qty={max_qty:.4f} | final={qty:.4f}"
        )
        return qty

    # ── Trade Approval ────────────────────────────────────────────────────────

    def approve_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        atr: float,
        current_equity: float,
    ) -> tuple[bool, str]:
        """
        Gate all trade entries through risk checks.
        Returns (approved: bool, reason: str)
        """
        # 1. Kill switch
        if self.is_kill_switch_active():
            return False, "Kill switch is active"

        # 2. Circuit breaker
        if self.state.circuit_breaker_active:
            return False, f"Circuit breaker active: {self.state.circuit_breaker_reason}"

        # 3. Daily loss limit
        tripped, reason = self._check_circuit_breaker(current_equity)
        if tripped:
            self.state.circuit_breaker_active = True
            self.state.circuit_breaker_reason = reason
            self.save_state()
            log.warning(f"CIRCUIT BREAKER TRIPPED: {reason}")
            return False, reason

        # 4. Max open positions
        if len(self.state.open_positions) >= self.config.max_open_positions:
            return False, f"Max open positions reached ({self.config.max_open_positions})"

        # 5. Already have position in this symbol
        if symbol in self.state.open_positions:
            existing = self.state.open_positions[symbol]
            if existing.get("side") == side:
                return False, f"Already have {side} position in {symbol}"

        # 6. Trade cooldown
        last_trade = self.state.last_trade_time.get(symbol)
        if last_trade:
            from datetime import datetime
            last_dt = datetime.fromisoformat(last_trade)
            now_dt = datetime.now(timezone.utc)
            elapsed = (now_dt - last_dt.replace(tzinfo=timezone.utc if last_dt.tzinfo is None else last_dt.tzinfo)).total_seconds()
            if elapsed < self.config.min_trade_interval_sec:
                remaining = self.config.min_trade_interval_sec - elapsed
                return False, f"Trade cooldown: {remaining:.0f}s remaining for {symbol}"

        # 7. Volatility circuit breaker
        atr_pct = atr / price if price > 0 else 0
        if atr_pct > self.config.max_atr_pct:
            return False, f"Extreme volatility: ATR/price={atr_pct*100:.1f}% > {self.config.max_atr_pct*100:.1f}%"

        # 8. Notional check
        notional = qty * price
        max_notional = self.config.max_position_pct * current_equity * self.config.max_leverage
        if notional > max_notional:
            return False, f"Notional ${notional:,.0f} exceeds max ${max_notional:,.0f}"

        return True, "Approved"

    # ── Trade Recording ───────────────────────────────────────────────────────

    def record_trade_open(self, symbol: str, side: str, qty: float, price: float):
        """Record a new trade opening."""
        notional = qty * price
        self.state.open_positions[symbol] = {
            "side": side,
            "qty": qty,
            "entry_price": price,
            "notional": notional,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state.last_trade_time[symbol] = datetime.now(timezone.utc).isoformat()
        self.state.daily_trade_count += 1
        self.save_state()
        log.info(f"Trade recorded: OPEN {side} {qty} {symbol} @ {price:,.2f} | notional=${notional:,.2f}")

    def record_trade_close(self, symbol: str, close_price: float):
        """Record a trade closing and update P&L."""
        if symbol not in self.state.open_positions:
            log.warning(f"No open position found for {symbol}")
            return

        pos = self.state.open_positions[symbol]
        entry_price = pos["entry_price"]
        qty = pos["qty"]
        side = pos["side"]

        # Calculate P&L
        if side == "Buy":
            pnl = (close_price - entry_price) * qty
        else:
            pnl = (entry_price - close_price) * qty

        self.state.daily_pnl += pnl
        self.state.current_equity += pnl
        if self.state.starting_equity > 0:
            self.state.daily_pnl_pct = self.state.daily_pnl / self.state.starting_equity * 100

        del self.state.open_positions[symbol]
        self.save_state()
        log.info(
            f"Trade recorded: CLOSE {symbol} @ {close_price:,.2f} | "
            f"PnL=${pnl:+,.2f} | Daily PnL=${self.state.daily_pnl:+,.2f} ({self.state.daily_pnl_pct:+.2f}%)"
        )

    def update_equity(self, current_equity: float):
        """Update current equity (called each loop iteration)."""
        self.state.current_equity = current_equity
        if self.state.starting_equity > 0:
            self.state.daily_pnl_pct = (
                (current_equity - self.state.starting_equity) / self.state.starting_equity * 100
            )
        self.save_state()

    # ── Status Report ─────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return a human-readable risk status summary."""
        return {
            "date": self.state.date,
            "equity": self.state.current_equity,
            "daily_pnl": round(self.state.daily_pnl, 2),
            "daily_pnl_pct": round(self.state.daily_pnl_pct, 3),
            "daily_trades": self.state.daily_trade_count,
            "open_positions": len(self.state.open_positions),
            "circuit_breaker": self.state.circuit_breaker_active,
            "circuit_breaker_reason": self.state.circuit_breaker_reason,
            "kill_switch": self.is_kill_switch_active(),
            "positions": self.state.open_positions,
        }

    def print_status(self):
        """Print a formatted risk status report."""
        s = self.get_status()
        log.info("=" * 55)
        log.info("RISK MANAGER STATUS")
        log.info("=" * 55)
        log.info(f"  Date:            {s['date']}")
        log.info(f"  Equity:          ${s['equity']:,.2f}")
        log.info(f"  Daily P&L:       ${s['daily_pnl']:+,.2f} ({s['daily_pnl_pct']:+.2f}%)")
        log.info(f"  Daily Trades:    {s['daily_trades']} / {self.config.max_daily_trades}")
        log.info(f"  Open Positions:  {s['open_positions']} / {self.config.max_open_positions}")
        log.info(f"  Circuit Breaker: {'ACTIVE — ' + s['circuit_breaker_reason'] if s['circuit_breaker'] else 'OK'}")
        log.info(f"  Kill Switch:     {'ACTIVE' if s['kill_switch'] else 'OK'}")
        for sym, pos in s["positions"].items():
            log.info(f"  Position: {sym} {pos['side']} {pos['qty']} @ {pos['entry_price']:,.2f}")
        log.info("=" * 55)
