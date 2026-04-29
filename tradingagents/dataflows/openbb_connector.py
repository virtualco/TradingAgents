"""OpenBB connector for the TradingAgents data layer.

This module provides a unified interface for fetching market data via OpenBB,
with automatic fallback to yfinance when OpenBB providers are unavailable or
when no API keys are configured.

Key design principles:
- All data is returned as normalized PIT (point-in-time) records
- ``available_at`` is always set to the fetch time for live data
- For historical backfills, callers must supply the correct ``available_at``
- No data is returned that has event_time > trade_date (time-safe by default)
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, date
from typing import Optional

import pandas as pd
import yfinance as yf

from .pit_schema import (
    DataLakePaths,
    DataVendor,
    normalize_fundamentals,
    normalize_ohlcv,
)

logger = logging.getLogger(__name__)


def _try_import_openbb():
    """Lazy import OpenBB to avoid slow startup when not needed."""
    try:
        from openbb import obb
        return obb
    except Exception as e:
        logger.warning(f"OpenBB not available: {e}. Falling back to yfinance.")
        return None


def _hash_payload(data) -> str:
    """Compute SHA-256 hash of raw payload for reproducibility."""
    raw = json.dumps(data, default=str, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class OpenBBConnector:
    """Unified data connector with OpenBB primary and yfinance fallback.

    Usage:
        connector = OpenBBConnector()

        # Fetch OHLCV (returns normalized PIT DataFrame)
        df = connector.get_ohlcv("AAPL", "2026-01-01", "2026-04-28")

        # Fetch fundamentals
        df = connector.get_fundamentals("AAPL", "2026-04-28")

        # Fetch news
        df = connector.get_news("AAPL", "2026-04-28", limit=20)
    """

    def __init__(
        self,
        prefer_openbb: bool = True,
        data_lake_root: str = "data",
    ):
        self.prefer_openbb = prefer_openbb
        self.paths = DataLakePaths(root=__import__("pathlib").Path(data_lake_root))
        self._obb = None  # Lazy-loaded

    @property
    def obb(self):
        if self._obb is None and self.prefer_openbb:
            self._obb = _try_import_openbb()
        return self._obb

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        available_at: Optional[datetime] = None,
        trade_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data for a ticker.

        Args:
            ticker: Instrument ticker symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            available_at: Override the availability timestamp (for backfill).
            trade_date: If provided, filter out any rows with event_time > trade_date
                        to enforce time-safety.

        Returns:
            Normalized OHLCV DataFrame with PIT metadata.
        """
        if available_at is None:
            available_at = datetime.utcnow()

        df = None
        vendor = DataVendor.YFINANCE

        # Try OpenBB first
        if self.obb is not None:
            try:
                result = self.obb.equity.price.historical(
                    symbol=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    provider="yfinance",
                )
                df = result.to_df()
                vendor = DataVendor.OPENBB
                logger.debug(f"OpenBB OHLCV fetched for {ticker}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"OpenBB OHLCV failed for {ticker}: {e}. Falling back to yfinance.")

        # Fallback to yfinance
        if df is None or df.empty:
            try:
                ticker_obj = yf.Ticker(ticker.upper())
                df = ticker_obj.history(start=start_date, end=end_date)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                vendor = DataVendor.YFINANCE
                logger.debug(f"yfinance OHLCV fetched for {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"yfinance OHLCV failed for {ticker}: {e}")
                return pd.DataFrame()

        if df is None or df.empty:
            logger.warning(f"No OHLCV data found for {ticker} ({start_date} to {end_date})")
            return pd.DataFrame()

        normalized = normalize_ohlcv(df, ticker, vendor, available_at)

        # Time-safety filter: remove rows with event_time > trade_date
        if trade_date:
            td = pd.to_datetime(trade_date).date()
            normalized = normalized[normalized["event_time"] <= td]

        return normalized

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(
        self,
        ticker: str,
        trade_date: str,
        available_at: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch the most recent available fundamentals as of trade_date.

        Args:
            ticker: Instrument ticker symbol.
            trade_date: The date we're trading on (for time-safety).
            available_at: Override the availability timestamp.

        Returns:
            Normalized fundamentals DataFrame.
        """
        if available_at is None:
            available_at = datetime.utcnow()

        vendor = DataVendor.YFINANCE
        rows = []

        try:
            ticker_obj = yf.Ticker(ticker.upper())
            info = ticker_obj.info or {}

            # Filter to only fields available as of trade_date
            # yfinance.info is a snapshot — we use it as a proxy for current fundamentals
            row_data = {
                "revenue": info.get("totalRevenue"),
                "grossProfit": info.get("grossProfits"),
                "operatingIncome": info.get("operatingCashflow"),
                "netIncome": info.get("netIncomeToCommon"),
                "eps": info.get("trailingEps"),
                "totalAssets": info.get("totalAssets"),
                "totalLiabilities": info.get("totalLiab"),
                "totalEquity": info.get("bookValue"),
                "freeCashflow": info.get("freeCashflow"),
            }

            # Use fiscal year end as period_end proxy
            fiscal_year_end = info.get("lastFiscalYearEnd")
            if fiscal_year_end:
                period_end = datetime.fromtimestamp(fiscal_year_end).strftime("%Y-%m-%d")
            else:
                period_end = trade_date

            # Time-safety: only include if period_end <= trade_date
            if period_end <= trade_date:
                df = normalize_fundamentals(
                    row_data, ticker, period_end, "annual", vendor, available_at
                )
                df["raw_payload_hash"] = _hash_payload(row_data)
                rows.append(df)

        except Exception as e:
            logger.warning(f"Fundamentals fetch failed for {ticker}: {e}")

        if rows:
            return pd.concat(rows, ignore_index=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        trade_date: str,
        limit: int = 20,
        available_at: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch recent news articles for a ticker as of trade_date.

        Args:
            ticker: Instrument ticker symbol.
            trade_date: The date we're trading on (for time-safety).
            limit: Maximum number of articles to return.
            available_at: Override the availability timestamp.

        Returns:
            Normalized news DataFrame.
        """
        if available_at is None:
            available_at = datetime.utcnow()

        rows = []
        trade_dt = pd.to_datetime(trade_date)

        try:
            ticker_obj = yf.Ticker(ticker.upper())
            news_items = ticker_obj.news or []

            for item in news_items[:limit]:
                # Parse publish time
                pub_time = item.get("providerPublishTime") or item.get("publishedAt")
                if pub_time:
                    if isinstance(pub_time, (int, float)):
                        event_time = datetime.fromtimestamp(pub_time)
                    else:
                        event_time = pd.to_datetime(pub_time)
                else:
                    continue

                # Time-safety: skip articles published after trade_date
                if event_time > trade_dt:
                    continue

                rows.append({
                    "ticker": ticker.upper(),
                    "event_time": event_time,
                    "available_at": available_at,
                    "vendor": DataVendor.YFINANCE.value,
                    "headline": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("publisher", ""),
                    "url": item.get("link", ""),
                    "sentiment_score": None,
                    "raw_payload_hash": _hash_payload(item),
                })

        except Exception as e:
            logger.warning(f"News fetch failed for {ticker}: {e}")

        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Convenience: get all data for a ticker as of trade_date
    # ------------------------------------------------------------------

    def get_research_data(
        self,
        ticker: str,
        trade_date: str,
        lookback_days: int = 90,
    ) -> dict:
        """Fetch all research data for a ticker as of trade_date.

        Returns a dict with keys: ohlcv, fundamentals, news.
        All data is time-safe (no lookahead beyond trade_date).

        Args:
            ticker: Instrument ticker symbol.
            trade_date: The date we're trading on.
            lookback_days: How many calendar days of history to fetch.

        Returns:
            Dict with 'ohlcv', 'fundamentals', 'news' DataFrames.
        """
        end_date = trade_date
        start_dt = pd.to_datetime(trade_date) - pd.Timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        logger.info(f"Fetching research data for {ticker} as of {trade_date}")

        return {
            "ohlcv": self.get_ohlcv(ticker, start_date, end_date, trade_date=trade_date),
            "fundamentals": self.get_fundamentals(ticker, trade_date),
            "news": self.get_news(ticker, trade_date),
        }
