"""Point-in-time (PIT) data schema for the TradingAgents data lake.

Every record in the data lake carries two timestamps:
- ``event_time``: when the underlying event occurred (e.g. market close date)
- ``available_at``: when this data became available to a trading system

This distinction is critical for avoiding lookahead bias in backtesting.
A fundamental filing may have an event_time of 2026-01-15 (quarter end)
but an available_at of 2026-02-10 (when the 10-Q was filed with the SEC).

Data lake layout (Parquet, partitioned):
    data/
      equities/
        ohlcv/          ← Daily OHLCV price data
        fundamentals/   ← Quarterly/annual financials
        technicals/     ← Pre-computed indicators
      news/             ← News and sentiment data
      macro/            ← Macro indicators (FRED, etc.)
      universe/         ← Universe membership snapshots
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    MACRO = "macro"
    NEWS = "news"


class DataVendor(str, Enum):
    YFINANCE = "yfinance"
    OPENBB = "openbb"
    ALPHA_VANTAGE = "alpha_vantage"
    MANUAL = "manual"


class DataFrequency(str, Enum):
    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    QUARTERLY = "1q"
    ANNUAL = "1y"


# ---------------------------------------------------------------------------
# PyArrow schemas for each data type
# ---------------------------------------------------------------------------

# OHLCV schema — daily equity price data
OHLCV_SCHEMA = pa.schema([
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("event_time", pa.date32(), nullable=False),      # market close date
    pa.field("available_at", pa.timestamp("us"), nullable=False),  # when data was available
    pa.field("vendor", pa.string(), nullable=False),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("adj_close", pa.float64()),
    pa.field("volume", pa.int64()),
    pa.field("dividends", pa.float64()),
    pa.field("stock_splits", pa.float64()),
    pa.field("raw_payload_hash", pa.string()),  # SHA-256 of raw response
])

# Fundamentals schema — quarterly/annual financials
FUNDAMENTALS_SCHEMA = pa.schema([
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("period_end", pa.date32(), nullable=False),       # fiscal period end date
    pa.field("period_type", pa.string(), nullable=False),      # "quarterly" or "annual"
    pa.field("available_at", pa.timestamp("us"), nullable=False),
    pa.field("vendor", pa.string(), nullable=False),
    pa.field("revenue", pa.float64()),
    pa.field("gross_profit", pa.float64()),
    pa.field("operating_income", pa.float64()),
    pa.field("net_income", pa.float64()),
    pa.field("eps", pa.float64()),
    pa.field("total_assets", pa.float64()),
    pa.field("total_liabilities", pa.float64()),
    pa.field("total_equity", pa.float64()),
    pa.field("free_cash_flow", pa.float64()),
    pa.field("raw_payload_hash", pa.string()),
])

# News schema — news articles and sentiment
NEWS_SCHEMA = pa.schema([
    pa.field("ticker", pa.string()),                           # nullable for global news
    pa.field("event_time", pa.timestamp("us"), nullable=False),  # article publish time
    pa.field("available_at", pa.timestamp("us"), nullable=False),
    pa.field("vendor", pa.string(), nullable=False),
    pa.field("headline", pa.string()),
    pa.field("summary", pa.string()),
    pa.field("source", pa.string()),
    pa.field("url", pa.string()),
    pa.field("sentiment_score", pa.float64()),   # -1.0 to 1.0
    pa.field("raw_payload_hash", pa.string()),
])

# Universe schema — which tickers are in the investable universe on each date
UNIVERSE_SCHEMA = pa.schema([
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("as_of_date", pa.date32(), nullable=False),
    pa.field("is_active", pa.bool_(), nullable=False),
    pa.field("delisted_date", pa.date32()),
    pa.field("market_cap", pa.float64()),
    pa.field("sector", pa.string()),
    pa.field("industry", pa.string()),
    pa.field("exchange", pa.string()),
])


# ---------------------------------------------------------------------------
# Data lake path helpers
# ---------------------------------------------------------------------------

@dataclass
class DataLakePaths:
    """Manages path conventions for the Parquet data lake."""
    root: Path = field(default_factory=lambda: Path("data"))

    def ohlcv_dir(self, ticker: str) -> Path:
        return self.root / "equities" / "ohlcv" / ticker.upper()

    def fundamentals_dir(self, ticker: str) -> Path:
        return self.root / "equities" / "fundamentals" / ticker.upper()

    def technicals_dir(self, ticker: str) -> Path:
        return self.root / "equities" / "technicals" / ticker.upper()

    def news_dir(self, ticker: Optional[str] = None) -> Path:
        if ticker:
            return self.root / "news" / ticker.upper()
        return self.root / "news" / "_global"

    def universe_dir(self) -> Path:
        return self.root / "universe"

    def ohlcv_file(self, ticker: str, year: int) -> Path:
        """Parquet file for a ticker's OHLCV data, partitioned by year."""
        return self.ohlcv_dir(ticker) / f"year={year}" / "data.parquet"

    def fundamentals_file(self, ticker: str, period_type: str) -> Path:
        return self.fundamentals_dir(ticker) / f"period={period_type}" / "data.parquet"

    def ensure_dirs(self, path: Path) -> Path:
        """Create directory tree if it doesn't exist."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------------------
# Normalizer: convert raw vendor data to PIT records
# ---------------------------------------------------------------------------

def normalize_ohlcv(
    df: pd.DataFrame,
    ticker: str,
    vendor: DataVendor,
    available_at: Optional[datetime] = None,
) -> pd.DataFrame:
    """Normalize raw OHLCV data to the PIT schema.

    Args:
        df: Raw DataFrame from vendor (yfinance, OpenBB, etc.)
        ticker: Instrument ticker symbol
        vendor: Data vendor enum
        available_at: When this data became available. Defaults to now (live data).

    Returns:
        Normalized DataFrame matching OHLCV_SCHEMA.
    """
    if available_at is None:
        available_at = datetime.utcnow()

    # Standardize column names (handle different vendor conventions)
    col_map = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
        "Dividends": "dividends", "Stock Splits": "stock_splits",
        # OpenBB conventions
        "open": "open", "high": "high", "low": "low", "close": "close",
        "adj_close": "adj_close", "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Remove timezone info from index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    normalized = pd.DataFrame({
        "ticker": ticker.upper(),
        "event_time": df.index.date,
        "available_at": available_at,
        "vendor": vendor.value,
        "open": df.get("open", pd.Series(dtype=float)),
        "high": df.get("high", pd.Series(dtype=float)),
        "low": df.get("low", pd.Series(dtype=float)),
        "close": df.get("close", pd.Series(dtype=float)),
        "adj_close": df.get("adj_close", df.get("close", pd.Series(dtype=float))),
        "volume": df.get("volume", pd.Series(dtype=float)).astype("Int64"),
        "dividends": df.get("dividends", 0.0),
        "stock_splits": df.get("stock_splits", 0.0),
        "raw_payload_hash": None,
    })

    return normalized.reset_index(drop=True)


def normalize_fundamentals(
    data: dict,
    ticker: str,
    period_end: str,
    period_type: str,
    vendor: DataVendor,
    available_at: Optional[datetime] = None,
) -> pd.DataFrame:
    """Normalize raw fundamentals to the PIT schema."""
    if available_at is None:
        available_at = datetime.utcnow()

    row = {
        "ticker": ticker.upper(),
        "period_end": pd.to_datetime(period_end).date(),
        "period_type": period_type,
        "available_at": available_at,
        "vendor": vendor.value,
        "revenue": data.get("revenue") or data.get("totalRevenue"),
        "gross_profit": data.get("grossProfit"),
        "operating_income": data.get("operatingIncome") or data.get("ebit"),
        "net_income": data.get("netIncome"),
        "eps": data.get("eps") or data.get("trailingEps"),
        "total_assets": data.get("totalAssets"),
        "total_liabilities": data.get("totalLiabilities") or data.get("totalLiab"),
        "total_equity": data.get("totalEquity") or data.get("totalStockholderEquity"),
        "free_cash_flow": data.get("freeCashflow"),
        "raw_payload_hash": None,
    }
    return pd.DataFrame([row])
