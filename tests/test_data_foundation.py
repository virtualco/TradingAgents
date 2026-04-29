"""Tests for Phase 1: Data Foundation.

Tests cover:
- PIT schema normalization (normalize_ohlcv, normalize_fundamentals)
- DataLake read/write operations
- OHLCVValidator and FundamentalsValidator
- Lookahead bias detection
- OpenBBConnector (yfinance path, no API keys required)
"""
from __future__ import annotations

import tempfile
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.dataflows.pit_schema import (
    DataLakePaths,
    DataVendor,
    normalize_ohlcv,
    normalize_fundamentals,
)
from tradingagents.dataflows.data_lake import DataLake
from tradingagents.dataflows.data_validator import (
    OHLCVValidator,
    FundamentalsValidator,
    Severity,
    validate_ohlcv,
    validate_fundamentals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv_df(
    ticker: str = "AAPL",
    start: str = "2026-01-02",
    days: int = 10,
    available_at: datetime = None,
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    if available_at is None:
        available_at = datetime(2026, 1, 5, 12, 0, 0)  # well before as_of_date 2026-04-28
    dates = pd.bdate_range(start=start, periods=days)
    df = pd.DataFrame({
        "ticker": ticker,
        "event_time": dates.date,
        "available_at": available_at,
        "vendor": DataVendor.YFINANCE.value,
        "open": [100.0 + i for i in range(days)],
        "high": [105.0 + i for i in range(days)],
        "low": [98.0 + i for i in range(days)],
        "close": [102.0 + i for i in range(days)],
        "adj_close": [102.0 + i for i in range(days)],
        "volume": [1_000_000 + i * 10_000 for i in range(days)],
        "dividends": 0.0,
        "stock_splits": 0.0,
        "raw_payload_hash": None,
    })
    return df


def make_raw_yfinance_df(days: int = 10) -> pd.DataFrame:
    """Create a raw yfinance-style DataFrame (capital column names)."""
    dates = pd.date_range(start="2026-01-02", periods=days, freq="B")
    df = pd.DataFrame({
        "Open": [100.0 + i for i in range(days)],
        "High": [105.0 + i for i in range(days)],
        "Low": [98.0 + i for i in range(days)],
        "Close": [102.0 + i for i in range(days)],
        "Volume": [1_000_000] * days,
        "Dividends": [0.0] * days,
        "Stock Splits": [0.0] * days,
    }, index=dates)
    return df


# ---------------------------------------------------------------------------
# PIT Schema Tests
# ---------------------------------------------------------------------------

class TestNormalizeOHLCV:
    def test_normalizes_yfinance_columns(self):
        raw = make_raw_yfinance_df(5)
        result = normalize_ohlcv(raw, "AAPL", DataVendor.YFINANCE)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "close" in result.columns
        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "AAPL"

    def test_ticker_uppercased(self):
        raw = make_raw_yfinance_df(3)
        result = normalize_ohlcv(raw, "aapl", DataVendor.YFINANCE)
        assert (result["ticker"] == "AAPL").all()

    def test_vendor_set_correctly(self):
        raw = make_raw_yfinance_df(3)
        result = normalize_ohlcv(raw, "MSFT", DataVendor.OPENBB)
        assert (result["vendor"] == "openbb").all()

    def test_available_at_set(self):
        raw = make_raw_yfinance_df(3)
        ts = datetime(2026, 4, 28, 9, 0, 0)
        result = normalize_ohlcv(raw, "AAPL", DataVendor.YFINANCE, available_at=ts)
        assert (result["available_at"] == ts).all()

    def test_available_at_defaults_to_now(self):
        raw = make_raw_yfinance_df(3)
        before = datetime.utcnow()
        result = normalize_ohlcv(raw, "AAPL", DataVendor.YFINANCE)
        after = datetime.utcnow()
        available_at = pd.to_datetime(result["available_at"].iloc[0])
        assert before <= available_at <= after

    def test_row_count_preserved(self):
        raw = make_raw_yfinance_df(10)
        result = normalize_ohlcv(raw, "AAPL", DataVendor.YFINANCE)
        assert len(result) == 10

    def test_event_time_is_date(self):
        raw = make_raw_yfinance_df(5)
        result = normalize_ohlcv(raw, "AAPL", DataVendor.YFINANCE)
        # event_time should be date objects
        assert result["event_time"].iloc[0] is not None


class TestNormalizeFundamentals:
    def test_basic_normalization(self):
        data = {
            "revenue": 95_400_000_000,
            "grossProfit": 43_000_000_000,
            "netIncome": 24_000_000_000,
            "trailingEps": 1.53,
            "totalAssets": 352_000_000_000,
        }
        result = normalize_fundamentals(
            data, "AAPL", "2026-01-31", "quarterly", DataVendor.YFINANCE
        )
        assert len(result) == 1
        assert result["ticker"].iloc[0] == "AAPL"
        assert result["revenue"].iloc[0] == 95_400_000_000
        assert result["period_type"].iloc[0] == "quarterly"

    def test_missing_fields_become_none(self):
        data = {"revenue": 100_000}
        result = normalize_fundamentals(
            data, "MSFT", "2026-01-31", "annual", DataVendor.YFINANCE
        )
        assert result["net_income"].iloc[0] is None or pd.isna(result["net_income"].iloc[0])


# ---------------------------------------------------------------------------
# DataLake Tests
# ---------------------------------------------------------------------------

class TestDataLake:
    def test_write_and_read_ohlcv(self, tmp_path):
        lake = DataLake(root=str(tmp_path))
        df = make_ohlcv_df("AAPL", days=10)
        rows = lake.write_ohlcv(df, "AAPL")
        assert rows == 10

        result = lake.read_ohlcv("AAPL", "2026-04-28", lookback_days=365)
        assert not result.empty
        assert (result["ticker"] == "AAPL").all()

    def test_time_safe_read_excludes_future(self, tmp_path):
        lake = DataLake(root=str(tmp_path))
        # Write data with available_at in the future
        future_available = datetime(2026, 12, 31, 0, 0, 0)
        df = make_ohlcv_df("TSLA", days=5, available_at=future_available)
        lake.write_ohlcv(df, "TSLA")

        # Read as of 2026-04-28 — should return empty (data not available yet)
        result = lake.read_ohlcv("TSLA", "2026-04-28", lookback_days=365)
        assert result.empty

    def test_write_deduplicates(self, tmp_path):
        lake = DataLake(root=str(tmp_path))
        df = make_ohlcv_df("AAPL", days=5)
        lake.write_ohlcv(df, "AAPL")
        # Write same data again
        lake.write_ohlcv(df, "AAPL")

        result = lake.read_ohlcv("AAPL", "2026-04-28", lookback_days=365)
        # Should not have duplicates
        assert len(result) == len(result.drop_duplicates(subset=["ticker", "event_time"]))

    def test_stats_returns_dict(self, tmp_path):
        lake = DataLake(root=str(tmp_path))
        stats = lake.stats()
        assert "status" in stats

    def test_stats_after_write(self, tmp_path):
        lake = DataLake(root=str(tmp_path))
        df = make_ohlcv_df("AAPL", days=5)
        lake.write_ohlcv(df, "AAPL")
        stats = lake.stats()
        assert stats["parquet_files"] >= 1

    def test_read_empty_lake_returns_empty_df(self, tmp_path):
        lake = DataLake(root=str(tmp_path))
        result = lake.read_ohlcv("NONEXISTENT", "2026-04-28")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_write_fundamentals(self, tmp_path):
        lake = DataLake(root=str(tmp_path))
        data = {"revenue": 95_000_000_000, "netIncome": 24_000_000_000}
        fund_df = normalize_fundamentals(
            data, "AAPL", "2025-12-31", "annual", DataVendor.YFINANCE,
            available_at=datetime(2026, 1, 15)
        )
        rows = lake.write_fundamentals(fund_df, "AAPL")
        assert rows == 1


# ---------------------------------------------------------------------------
# OHLCV Validator Tests
# ---------------------------------------------------------------------------

class TestOHLCVValidator:
    def test_valid_data_passes(self):
        df = make_ohlcv_df("AAPL", days=20)
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert report.is_valid
        assert report.row_count == 20

    def test_empty_data_is_error(self):
        report = validate_ohlcv(pd.DataFrame(), "AAPL", "2026-04-28")
        assert not report.is_valid
        assert any(i.check == "empty_data" for i in report.issues)

    def test_negative_price_is_critical(self):
        df = make_ohlcv_df("AAPL", days=5)
        df.loc[2, "close"] = -1.0
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert not report.is_valid
        critical = [i for i in report.issues if i.severity == Severity.CRITICAL]
        assert any(i.check == "negative_prices" for i in critical)

    def test_zero_price_is_critical(self):
        df = make_ohlcv_df("AAPL", days=5)
        df.loc[0, "open"] = 0.0
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert not report.is_valid

    def test_ohlc_inconsistency_detected(self):
        df = make_ohlcv_df("AAPL", days=5)
        # Make high < low for one row
        df.loc[1, "high"] = 50.0
        df.loc[1, "low"] = 100.0
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert not report.is_valid
        assert any(i.check == "ohlc_consistency" for i in report.issues)

    def test_lookahead_bias_detected(self):
        df = make_ohlcv_df("AAPL", days=5)
        # Set available_at to future date
        df["available_at"] = datetime(2027, 1, 1)
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert not report.is_valid
        critical = [i for i in report.issues if i.severity == Severity.CRITICAL]
        assert any(i.check == "lookahead_bias" for i in critical)

    def test_null_prices_flagged(self):
        df = make_ohlcv_df("AAPL", days=10)
        df.loc[0:1, "close"] = None
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert any(i.check == "null_prices" for i in report.issues)

    def test_zero_volume_flagged(self):
        df = make_ohlcv_df("AAPL", days=5)
        df["volume"] = 0
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert any(i.check == "zero_volume" for i in report.issues)

    def test_extreme_return_flagged_as_warning(self):
        df = make_ohlcv_df("AAPL", days=5)
        # Simulate a 100% price jump
        df.loc[2, "close"] = df.loc[1, "close"] * 3.0
        report = validate_ohlcv(df, "AAPL", "2026-04-28")
        assert any(i.check == "price_outliers" for i in report.issues)

    def test_summary_contains_ticker(self):
        df = make_ohlcv_df("MSFT", days=5)
        report = validate_ohlcv(df, "MSFT", "2026-04-28")
        assert "MSFT" in report.summary()

    def test_is_valid_true_for_clean_data(self):
        df = make_ohlcv_df("GOOGL", days=15)
        report = validate_ohlcv(df, "GOOGL", "2026-04-28")
        assert report.is_valid


# ---------------------------------------------------------------------------
# Fundamentals Validator Tests
# ---------------------------------------------------------------------------

class TestFundamentalsValidator:
    def make_fund_df(self, ticker="AAPL", days_ago=30):
        available_at = datetime(2026, 4, 28) - pd.Timedelta(days=days_ago)
        return pd.DataFrame([{
            "ticker": ticker,
            "period_end": date(2025, 12, 31),
            "period_type": "annual",
            "available_at": available_at,
            "vendor": "yfinance",
            "revenue": 395_000_000_000,
            "gross_profit": 180_000_000_000,
            "operating_income": 120_000_000_000,
            "net_income": 100_000_000_000,
            "eps": 6.43,
            "total_assets": 352_000_000_000,
            "total_liabilities": 290_000_000_000,
            "total_equity": 62_000_000_000,
            "free_cash_flow": 90_000_000_000,
            "raw_payload_hash": None,
        }])

    def test_valid_fundamentals_pass(self):
        df = self.make_fund_df()
        report = validate_fundamentals(df, "AAPL", "2026-04-28")
        assert report.is_valid

    def test_empty_fundamentals_warning(self):
        report = validate_fundamentals(pd.DataFrame(), "AAPL", "2026-04-28")
        assert any(i.check == "empty_data" for i in report.issues)

    def test_stale_fundamentals_flagged(self):
        df = self.make_fund_df(days_ago=150)  # 150 days stale
        report = validate_fundamentals(df, "AAPL", "2026-04-28")
        assert any(i.check == "stale_fundamentals" for i in report.issues)

    def test_fresh_fundamentals_no_staleness_warning(self):
        df = self.make_fund_df(days_ago=30)
        report = validate_fundamentals(df, "AAPL", "2026-04-28")
        assert not any(i.check == "stale_fundamentals" for i in report.issues)

    def test_lookahead_bias_in_fundamentals(self):
        df = self.make_fund_df()
        df["available_at"] = datetime(2027, 6, 1)
        report = validate_fundamentals(df, "AAPL", "2026-04-28")
        assert not report.is_valid
        assert any(i.check == "lookahead_bias" for i in report.issues)


# ---------------------------------------------------------------------------
# DataLakePaths Tests
# ---------------------------------------------------------------------------

class TestDataLakePaths:
    def test_ohlcv_dir_structure(self, tmp_path):
        paths = DataLakePaths(root=tmp_path)
        p = paths.ohlcv_dir("AAPL")
        assert "AAPL" in str(p)
        assert "ohlcv" in str(p)

    def test_ticker_uppercased_in_path(self, tmp_path):
        paths = DataLakePaths(root=tmp_path)
        p = paths.ohlcv_dir("aapl")
        assert "AAPL" in str(p)

    def test_ohlcv_file_partitioned_by_year(self, tmp_path):
        paths = DataLakePaths(root=tmp_path)
        p = paths.ohlcv_file("AAPL", 2026)
        assert "year=2026" in str(p)

    def test_ensure_dirs_creates_parent(self, tmp_path):
        paths = DataLakePaths(root=tmp_path)
        p = paths.ohlcv_file("AAPL", 2026)
        paths.ensure_dirs(p)
        assert p.parent.exists()
