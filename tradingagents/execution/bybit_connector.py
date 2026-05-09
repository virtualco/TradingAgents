"""
Bybit Exchange Connector — TradingAgents
=========================================
Supports both Testnet (paper trading) and Live trading via pybit unified_trading.

Features:
  - Unified HTTP client for Bybit V5 API
  - Market and limit order placement with retry logic
  - Position and balance queries
  - Order status tracking
  - Rate-limit-aware request throttling
  - Full testnet/mainnet toggle

Usage:
    from tradingagents.execution.bybit_connector import BybitConnector

    # Testnet (paper trading)
    conn = BybitConnector(testnet=True)
    conn.set_credentials(api_key="...", api_secret="...")
    balance = conn.get_balance()
    order = conn.place_market_order("BTCUSDT", "Buy", qty=0.001)

    # Live (requires real credentials)
    conn = BybitConnector(testnet=False)
"""
from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("bybit_connector")

try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    log.warning("pybit not installed — run: pip3 install pybit")


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    price: float = 0.0
    order_type: str = ""
    status: str = ""
    error: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: str          # "Buy" or "Sell"
    size: float
    avg_price: float
    unrealised_pnl: float
    leverage: float
    mark_price: float


@dataclass
class Balance:
    total_equity: float
    available_balance: float
    wallet_balance: float
    currency: str = "USDT"


# ── Connector ─────────────────────────────────────────────────────────────────

class BybitConnector:
    """
    Bybit V5 Unified Trading connector.

    Supports:
      - Linear perpetuals (BTCUSDT, ETHUSDT, etc.)
      - Spot trading
      - Testnet and mainnet
    """

    TESTNET_URL = "https://api-testnet.bybit.com"
    MAINNET_URL = "https://api.bybit.com"

    def __init__(
        self,
        testnet: bool = True,
        category: str = "linear",   # "linear" for perps, "spot" for spot
        recv_window: int = 5000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if not PYBIT_AVAILABLE:
            raise ImportError("pybit is required: pip3 install pybit")

        self.testnet = testnet
        self.category = category
        self.recv_window = recv_window
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[HTTP] = None

        log.info(f"BybitConnector initialised | testnet={testnet} | category={category}")

    def set_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Set API credentials. Falls back to environment variables:
          BYBIT_API_KEY, BYBIT_API_SECRET
          BYBIT_TESTNET_API_KEY, BYBIT_TESTNET_API_SECRET
        """
        if self.testnet:
            key    = api_key    or os.getenv("BYBIT_TESTNET_API_KEY",    os.getenv("BYBIT_API_KEY", ""))
            secret = api_secret or os.getenv("BYBIT_TESTNET_API_SECRET", os.getenv("BYBIT_API_SECRET", ""))
        else:
            key    = api_key    or os.getenv("BYBIT_API_KEY", "")
            secret = api_secret or os.getenv("BYBIT_API_SECRET", "")

        self._client = HTTP(
            testnet=self.testnet,
            api_key=key or None,
            api_secret=secret or None,
            recv_window=self.recv_window,
        )
        mode = "TESTNET" if self.testnet else "LIVE"
        log.info(f"Credentials set | mode={mode} | key_present={bool(key)}")

    def _ensure_client(self):
        if self._client is None:
            self.set_credentials()

    def _call_with_retry(self, fn, *args, **kwargs):
        """Execute API call with exponential backoff retry."""
        self._ensure_client()
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                if result.get("retCode", -1) == 0:
                    return result
                err_msg = result.get("retMsg", "Unknown error")
                log.warning(f"API error (attempt {attempt}): {err_msg}")
                last_err = err_msg
            except Exception as e:
                log.warning(f"Request exception (attempt {attempt}): {e}")
                last_err = str(e)
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_err}")

    # ── Market Data ───────────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> dict:
        """Get latest ticker (last price, bid, ask, volume)."""
        result = self._call_with_retry(
            self._client.get_tickers,
            category=self.category,
            symbol=symbol,
        )
        items = result.get("result", {}).get("list", [])
        return items[0] if items else {}

    def get_last_price(self, symbol: str) -> float:
        """Get last traded price for a symbol."""
        ticker = self.get_ticker(symbol)
        return float(ticker.get("lastPrice", 0))

    def get_klines(
        self,
        symbol: str,
        interval: str = "60",   # "1","3","5","15","30","60","120","240","D","W"
        limit: int = 200,
    ) -> list:
        """
        Fetch OHLCV klines.
        Returns list of [timestamp, open, high, low, close, volume, turnover]
        """
        result = self._call_with_retry(
            self._client.get_kline,
            category=self.category,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        return result.get("result", {}).get("list", [])

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self, coin: str = "USDT") -> Balance:
        """Get unified trading account balance."""
        result = self._call_with_retry(
            self._client.get_wallet_balance,
            accountType="UNIFIED",
            coin=coin,
        )
        accounts = result.get("result", {}).get("list", [])
        if not accounts:
            return Balance(0, 0, 0, coin)

        account = accounts[0]
        coins = account.get("coin", [])
        coin_data = next((c for c in coins if c.get("coin") == coin), {})

        return Balance(
            total_equity=float(account.get("totalEquity", 0)),
            available_balance=float(coin_data.get("availableToWithdraw", 0)),
            wallet_balance=float(coin_data.get("walletBalance", 0)),
            currency=coin,
        )

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get open positions."""
        kwargs = {"category": self.category, "settleCoin": "USDT"}
        if symbol:
            kwargs["symbol"] = symbol
        result = self._call_with_retry(self._client.get_positions, **kwargs)
        positions = []
        for p in result.get("result", {}).get("list", []):
            size = float(p.get("size", 0))
            if size == 0:
                continue
            positions.append(Position(
                symbol=p.get("symbol", ""),
                side=p.get("side", ""),
                size=size,
                avg_price=float(p.get("avgPrice", 0)),
                unrealised_pnl=float(p.get("unrealisedPnl", 0)),
                leverage=float(p.get("leverage", 1)),
                mark_price=float(p.get("markPrice", 0)),
            ))
        return positions

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        side: str,          # "Buy" or "Sell"
        qty: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a market order."""
        try:
            result = self._call_with_retry(
                self._client.place_order,
                category=self.category,
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                reduceOnly=reduce_only,
                timeInForce="IOC",
            )
            order_info = result.get("result", {})
            return OrderResult(
                success=True,
                order_id=order_info.get("orderId", ""),
                symbol=symbol,
                side=side,
                qty=qty,
                order_type="Market",
                status="Submitted",
                raw=result,
            )
        except Exception as e:
            log.error(f"Market order failed: {e}")
            return OrderResult(success=False, symbol=symbol, side=side, qty=qty, error=str(e))

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> OrderResult:
        """Place a limit order."""
        try:
            result = self._call_with_retry(
                self._client.place_order,
                category=self.category,
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                reduceOnly=reduce_only,
                timeInForce=time_in_force,
            )
            order_info = result.get("result", {})
            return OrderResult(
                success=True,
                order_id=order_info.get("orderId", ""),
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                order_type="Limit",
                status="Submitted",
                raw=result,
            )
        except Exception as e:
            log.error(f"Limit order failed: {e}")
            return OrderResult(success=False, symbol=symbol, side=side, qty=qty, price=price, error=str(e))

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self._call_with_retry(
                self._client.cancel_order,
                category=self.category,
                symbol=symbol,
                orderId=order_id,
            )
            return True
        except Exception as e:
            log.error(f"Cancel order failed: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        try:
            self._call_with_retry(
                self._client.cancel_all_orders,
                category=self.category,
                symbol=symbol,
            )
            return True
        except Exception as e:
            log.error(f"Cancel all orders failed: {e}")
            return False

    def close_position(self, symbol: str) -> OrderResult:
        """Close all open positions for a symbol with a market order."""
        positions = self.get_positions(symbol)
        if not positions:
            log.info(f"No open position for {symbol}")
            return OrderResult(success=True, symbol=symbol, status="NoPosition")

        pos = positions[0]
        close_side = "Sell" if pos.side == "Buy" else "Buy"
        log.info(f"Closing {pos.side} position: {pos.size} {symbol} @ market")
        return self.place_market_order(symbol, close_side, pos.size, reduce_only=True)

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a linear perpetual."""
        try:
            self._call_with_retry(
                self._client.set_leverage,
                category=self.category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            log.info(f"Leverage set: {symbol} = {leverage}x")
            return True
        except Exception as e:
            log.error(f"Set leverage failed: {e}")
            return False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Test connectivity to Bybit API."""
        try:
            self._ensure_client()
            result = self._client.get_server_time()
            return result.get("retCode", -1) == 0
        except Exception as e:
            log.error(f"Ping failed: {e}")
            return False

    def get_server_time(self) -> int:
        """Get Bybit server time in milliseconds."""
        self._ensure_client()
        result = self._client.get_server_time()
        return int(result.get("result", {}).get("timeNano", 0)) // 1_000_000

    def __repr__(self) -> str:
        mode = "TESTNET" if self.testnet else "LIVE"
        return f"BybitConnector(mode={mode}, category={self.category})"
