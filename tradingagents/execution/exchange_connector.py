"""
CCXT Multi-Exchange Connector — TradingAgents v9.0
====================================================
Abstract exchange interface with CCXT-based adapters for Bybit, Binance, and OKX.
Provides:
  - Unified ExchangeConnector ABC (place_order, cancel_order, get_balance, get_positions)
  - CCXT-based adapters for each exchange with futures support
  - ExchangeRouter: best-execution selection + automatic failover
  - Fee/latency tracking for intelligent routing

Usage:
    from tradingagents.execution.exchange_connector import ExchangeRouter

    router = ExchangeRouter()
    router.add_exchange("bybit", BybitAdapter(api_key="...", api_secret="...", testnet=True))
    router.add_exchange("binance", BinanceAdapter(api_key="...", api_secret="..."))

    # Auto-selects best exchange based on fees + latency
    result = router.place_order("BTCUSDT", "buy", qty=0.001)

    # Or target a specific exchange
    result = router.place_order("BTCUSDT", "buy", qty=0.001, exchange="binance")
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("exchange_connector")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    log.warning("ccxt not installed — run: pip3 install ccxt")


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """Unified order result across all exchanges."""
    success: bool
    exchange: str = ""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    price: float = 0.0
    order_type: str = "market"
    status: str = ""
    fee: float = 0.0
    fee_currency: str = "USDT"
    latency_ms: float = 0.0
    error: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class Position:
    """Unified position representation."""
    exchange: str
    symbol: str
    side: str           # "long" or "short"
    size: float
    avg_price: float
    unrealised_pnl: float
    leverage: float
    mark_price: float
    liquidation_price: float = 0.0


@dataclass
class Balance:
    """Unified balance representation."""
    exchange: str
    total_equity: float
    available_balance: float
    wallet_balance: float
    currency: str = "USDT"


@dataclass
class ExchangeMetrics:
    """Tracks exchange performance for routing decisions."""
    exchange: str
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    total_orders: int = 0
    failed_orders: int = 0
    last_error: str = ""
    last_success_ts: float = 0.0


# ── Abstract Base Class ───────────────────────────────────────────────────────

class ExchangeConnector(ABC):
    """Abstract interface for exchange adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange identifier (e.g., 'bybit', 'binance', 'okx')."""
        ...

    @property
    @abstractmethod
    def maker_fee(self) -> float:
        """Maker fee rate (e.g., 0.0002 for 0.02%)."""
        ...

    @property
    @abstractmethod
    def taker_fee(self) -> float:
        """Taker fee rate (e.g., 0.0005 for 0.05%)."""
        ...

    @abstractmethod
    def ping(self) -> bool:
        """Check connectivity. Returns True if exchange is reachable."""
        ...

    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> Balance:
        """Get account balance."""
        ...

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions. If symbol is None, return all."""
        ...

    @abstractmethod
    def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """Place a market order. side: 'buy' or 'sell'."""
        ...

    @abstractmethod
    def place_limit_order(self, symbol: str, side: str, qty: float, price: float) -> OrderResult:
        """Place a limit order."""
        ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order by ID."""
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult:
        """Close all positions for a symbol."""
        ...

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict:
        """Get current ticker data (bid, ask, last, volume)."""
        ...

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        ...


# ── CCXT Base Adapter ─────────────────────────────────────────────────────────

class CCXTAdapter(ExchangeConnector):
    """Base CCXT adapter with common logic for all exchanges."""

    def __init__(self, exchange_id: str, api_key: str, api_secret: str,
                 password: str = "", testnet: bool = False, **kwargs):
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is required: pip3 install ccxt")

        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # Use futures/perpetual by default
        }
        if password:
            config["password"] = password

        self._exchange: ccxt.Exchange = getattr(ccxt, exchange_id)(config)

        if testnet:
            self._exchange.set_sandbox_mode(True)

        self._exchange_id = exchange_id
        self._testnet = testnet

    @property
    def name(self) -> str:
        return self._exchange_id

    def ping(self) -> bool:
        try:
            self._exchange.fetch_time()
            return True
        except Exception as e:
            log.warning(f"[{self.name}] ping failed: {e}")
            return False

    def get_balance(self, currency: str = "USDT") -> Balance:
        try:
            bal = self._exchange.fetch_balance()
            total = float(bal.get("total", {}).get(currency, 0))
            free = float(bal.get("free", {}).get(currency, 0))
            used = float(bal.get("used", {}).get(currency, 0))
            return Balance(
                exchange=self.name,
                total_equity=total,
                available_balance=free,
                wallet_balance=total,
                currency=currency,
            )
        except Exception as e:
            log.error(f"[{self.name}] get_balance failed: {e}")
            return Balance(exchange=self.name, total_equity=0, available_balance=0, wallet_balance=0)

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        try:
            positions = self._exchange.fetch_positions([symbol] if symbol else None)
            result = []
            for p in positions:
                size = abs(float(p.get("contracts", 0) or 0))
                if size == 0:
                    continue
                result.append(Position(
                    exchange=self.name,
                    symbol=p["symbol"],
                    side=p.get("side", "long"),
                    size=size,
                    avg_price=float(p.get("entryPrice", 0) or 0),
                    unrealised_pnl=float(p.get("unrealizedPnl", 0) or 0),
                    leverage=float(p.get("leverage", 1) or 1),
                    mark_price=float(p.get("markPrice", 0) or 0),
                    liquidation_price=float(p.get("liquidationPrice", 0) or 0),
                ))
            return result
        except Exception as e:
            log.error(f"[{self.name}] get_positions failed: {e}")
            return []

    def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        start = time.time()
        try:
            order = self._exchange.create_order(
                symbol=self._normalise_symbol(symbol),
                type="market",
                side=side.lower(),
                amount=qty,
            )
            latency = (time.time() - start) * 1000
            fee_info = order.get("fee", {}) or {}
            return OrderResult(
                success=True,
                exchange=self.name,
                order_id=order.get("id", ""),
                symbol=symbol,
                side=side,
                qty=qty,
                price=float(order.get("average", 0) or order.get("price", 0) or 0),
                order_type="market",
                status=order.get("status", "filled"),
                fee=float(fee_info.get("cost", 0) or 0),
                fee_currency=fee_info.get("currency", "USDT"),
                latency_ms=latency,
                raw=order,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            log.error(f"[{self.name}] market order failed: {e}")
            return OrderResult(success=False, exchange=self.name, symbol=symbol,
                             side=side, qty=qty, error=str(e), latency_ms=latency)

    def place_limit_order(self, symbol: str, side: str, qty: float, price: float) -> OrderResult:
        start = time.time()
        try:
            order = self._exchange.create_order(
                symbol=self._normalise_symbol(symbol),
                type="limit",
                side=side.lower(),
                amount=qty,
                price=price,
            )
            latency = (time.time() - start) * 1000
            fee_info = order.get("fee", {}) or {}
            return OrderResult(
                success=True,
                exchange=self.name,
                order_id=order.get("id", ""),
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                order_type="limit",
                status=order.get("status", "open"),
                fee=float(fee_info.get("cost", 0) or 0),
                fee_currency=fee_info.get("currency", "USDT"),
                latency_ms=latency,
                raw=order,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            log.error(f"[{self.name}] limit order failed: {e}")
            return OrderResult(success=False, exchange=self.name, symbol=symbol,
                             side=side, qty=qty, error=str(e), latency_ms=latency)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self._exchange.cancel_order(order_id, self._normalise_symbol(symbol))
            return True
        except Exception as e:
            log.error(f"[{self.name}] cancel_order failed: {e}")
            return False

    def close_position(self, symbol: str) -> OrderResult:
        """Close position by placing opposite market order."""
        positions = self.get_positions(symbol)
        if not positions:
            return OrderResult(success=True, exchange=self.name, symbol=symbol,
                             status="no_position")
        pos = positions[0]
        close_side = "sell" if pos.side == "long" else "buy"
        return self.place_market_order(symbol, close_side, pos.size)

    def get_ticker(self, symbol: str) -> dict:
        try:
            ticker = self._exchange.fetch_ticker(self._normalise_symbol(symbol))
            return {
                "bid": float(ticker.get("bid", 0) or 0),
                "ask": float(ticker.get("ask", 0) or 0),
                "last": float(ticker.get("last", 0) or 0),
                "volume": float(ticker.get("baseVolume", 0) or 0),
                "spread": float(ticker.get("ask", 0) or 0) - float(ticker.get("bid", 0) or 0),
            }
        except Exception as e:
            log.error(f"[{self.name}] get_ticker failed: {e}")
            return {"bid": 0, "ask": 0, "last": 0, "volume": 0, "spread": 0}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self._exchange.set_leverage(leverage, self._normalise_symbol(symbol))
            return True
        except Exception as e:
            log.warning(f"[{self.name}] set_leverage failed: {e}")
            return False

    def _normalise_symbol(self, symbol: str) -> str:
        """Convert BTCUSDT → BTC/USDT:USDT for CCXT perpetual format."""
        # Already in CCXT format
        if "/" in symbol:
            return symbol
        # Common crypto pairs: BTCUSDT → BTC/USDT:USDT
        for quote in ["USDT", "BUSD", "USD"]:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}:{quote}"
        return symbol


# ── Exchange-Specific Adapters ────────────────────────────────────────────────

class BybitAdapter(CCXTAdapter):
    """Bybit futures adapter via CCXT."""

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        super().__init__("bybit", api_key, api_secret, testnet=testnet)

    @property
    def maker_fee(self) -> float:
        return 0.0001  # 0.01%

    @property
    def taker_fee(self) -> float:
        return 0.0006  # 0.06%


class BinanceAdapter(CCXTAdapter):
    """Binance futures adapter via CCXT."""

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        super().__init__("binanceusdm", api_key, api_secret, testnet=testnet)

    @property
    def maker_fee(self) -> float:
        return 0.0002  # 0.02%

    @property
    def taker_fee(self) -> float:
        return 0.0005  # 0.05%


class OKXAdapter(CCXTAdapter):
    """OKX futures adapter via CCXT."""

    def __init__(self, api_key: str = "", api_secret: str = "", password: str = "",
                 testnet: bool = False):
        super().__init__("okx", api_key, api_secret, password=password, testnet=testnet)

    @property
    def maker_fee(self) -> float:
        return 0.0002  # 0.02%

    @property
    def taker_fee(self) -> float:
        return 0.0005  # 0.05%


# ── Exchange Router ───────────────────────────────────────────────────────────

class ExchangeRouter:
    """
    Intelligent exchange router with best-execution selection and failover.

    Routing strategy:
      1. If a specific exchange is requested, use it (with failover on error)
      2. Otherwise, score each exchange: lower fee + lower latency = higher score
      3. If primary fails, automatically try next-best exchange

    Failover logic:
      - On order failure, mark exchange as degraded
      - Retry on next-best exchange
      - After 5 consecutive failures, disable exchange for 5 minutes
    """

    MAX_CONSECUTIVE_FAILURES = 5
    COOLDOWN_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._exchanges: Dict[str, ExchangeConnector] = {}
        self._metrics: Dict[str, ExchangeMetrics] = {}
        self._disabled_until: Dict[str, float] = {}

    def add_exchange(self, name: str, connector: ExchangeConnector) -> None:
        """Register an exchange adapter."""
        self._exchanges[name] = connector
        self._metrics[name] = ExchangeMetrics(
            exchange=name,
            maker_fee=connector.maker_fee,
            taker_fee=connector.taker_fee,
        )
        log.info(f"[Router] Added exchange: {name} (maker={connector.maker_fee}, taker={connector.taker_fee})")

    def remove_exchange(self, name: str) -> None:
        """Remove an exchange adapter."""
        self._exchanges.pop(name, None)
        self._metrics.pop(name, None)

    @property
    def exchanges(self) -> List[str]:
        """List of registered exchange names."""
        return list(self._exchanges.keys())

    def get_metrics(self) -> Dict[str, ExchangeMetrics]:
        """Get performance metrics for all exchanges."""
        return dict(self._metrics)

    def _is_available(self, name: str) -> bool:
        """Check if exchange is available (not in cooldown)."""
        until = self._disabled_until.get(name, 0)
        if time.time() < until:
            return False
        return True

    def _rank_exchanges(self, order_type: str = "market") -> List[str]:
        """Rank exchanges by composite score (lower = better)."""
        available = [n for n in self._exchanges if self._is_available(n)]
        if not available:
            # Force all back if none available
            available = list(self._exchanges.keys())

        def score(name: str) -> float:
            m = self._metrics[name]
            fee = m.taker_fee if order_type == "market" else m.maker_fee
            latency_penalty = m.avg_latency_ms / 10000  # Normalise latency contribution
            failure_penalty = (1 - m.success_rate) * 2
            return fee + latency_penalty + failure_penalty

        return sorted(available, key=score)

    def _update_metrics(self, name: str, result: OrderResult) -> None:
        """Update exchange metrics after an order attempt."""
        m = self._metrics[name]
        m.total_orders += 1

        if result.success:
            m.failed_orders = 0  # Reset consecutive failures
            m.last_success_ts = time.time()
            # Exponential moving average for latency
            alpha = 0.2
            m.avg_latency_ms = alpha * result.latency_ms + (1 - alpha) * m.avg_latency_ms
            m.success_rate = min(1.0, m.success_rate + 0.05)
        else:
            m.failed_orders += 1
            m.last_error = result.error
            m.success_rate = max(0.0, m.success_rate - 0.1)

            # Disable if too many consecutive failures
            if m.failed_orders >= self.MAX_CONSECUTIVE_FAILURES:
                self._disabled_until[name] = time.time() + self.COOLDOWN_SECONDS
                log.warning(f"[Router] Disabled {name} for {self.COOLDOWN_SECONDS}s after {m.failed_orders} failures")

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", price: float = 0.0,
                    exchange: Optional[str] = None) -> OrderResult:
        """
        Place an order with best-execution routing and failover.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "buy" or "sell"
            qty: Order quantity
            order_type: "market" or "limit"
            price: Limit price (required for limit orders)
            exchange: Force specific exchange (optional)

        Returns:
            OrderResult with execution details
        """
        # Determine execution order
        if exchange and exchange in self._exchanges:
            candidates = [exchange] + [e for e in self._rank_exchanges(order_type) if e != exchange]
        else:
            candidates = self._rank_exchanges(order_type)

        if not candidates:
            return OrderResult(success=False, error="No exchanges available")

        # Try each candidate with failover
        for name in candidates:
            conn = self._exchanges[name]
            try:
                if order_type == "market":
                    result = conn.place_market_order(symbol, side, qty)
                else:
                    result = conn.place_limit_order(symbol, side, qty, price)

                self._update_metrics(name, result)

                if result.success:
                    log.info(f"[Router] Order filled on {name}: {symbol} {side} {qty} @ {result.price}")
                    return result
                else:
                    log.warning(f"[Router] Order failed on {name}: {result.error}, trying next...")

            except Exception as e:
                log.error(f"[Router] Exception on {name}: {e}")
                self._update_metrics(name, OrderResult(success=False, error=str(e)))

        return OrderResult(success=False, error=f"All {len(candidates)} exchanges failed")

    def get_best_price(self, symbol: str) -> Tuple[str, dict]:
        """Get best bid/ask across all exchanges."""
        best_exchange = ""
        best_ticker = {"bid": 0, "ask": float("inf"), "last": 0, "volume": 0, "spread": float("inf")}

        for name, conn in self._exchanges.items():
            if not self._is_available(name):
                continue
            ticker = conn.get_ticker(symbol)
            if ticker["spread"] < best_ticker["spread"] and ticker["bid"] > 0:
                best_ticker = ticker
                best_exchange = name

        return best_exchange, best_ticker

    def get_aggregate_balance(self, currency: str = "USDT") -> Dict[str, Balance]:
        """Get balance from all exchanges."""
        balances = {}
        for name, conn in self._exchanges.items():
            balances[name] = conn.get_balance(currency)
        return balances

    def get_all_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get positions across all exchanges."""
        all_positions = []
        for name, conn in self._exchanges.items():
            all_positions.extend(conn.get_positions(symbol))
        return all_positions

    def health_check(self) -> Dict[str, bool]:
        """Ping all exchanges and return connectivity status."""
        status = {}
        for name, conn in self._exchanges.items():
            status[name] = conn.ping()
        return status
