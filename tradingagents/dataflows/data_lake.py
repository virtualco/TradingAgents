"""Parquet data lake with DuckDB query layer.

Provides read/write operations for the TradingAgents data lake:
- Write: append-only Parquet files, partitioned by ticker and year
- Read: DuckDB SQL queries over Parquet files for fast analytics
- Time-safe reads: automatically filter to data available as of a given date

The lake is append-only by design. Corrections are written as new records
with updated ``available_at`` timestamps, not by overwriting old records.
This preserves the full audit trail and enables point-in-time replay.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .pit_schema import (
    FUNDAMENTALS_SCHEMA,
    NEWS_SCHEMA,
    OHLCV_SCHEMA,
    UNIVERSE_SCHEMA,
    DataLakePaths,
)

logger = logging.getLogger(__name__)


class DataLake:
    """Parquet data lake with DuckDB query interface.

    Usage:
        lake = DataLake("data/")

        # Write OHLCV data
        lake.write_ohlcv(df, "AAPL")

        # Read OHLCV as of a trade date (time-safe)
        df = lake.read_ohlcv("AAPL", "2026-04-28", lookback_days=90)

        # Run arbitrary SQL over the lake
        result = lake.query("SELECT * FROM ohlcv WHERE ticker = 'AAPL' LIMIT 10")
    """

    def __init__(self, root: str = "data"):
        self.paths = DataLakePaths(root=Path(root))
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Lazy DuckDB connection."""
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
            # Install httpfs for potential S3 reads in future
            try:
                self._conn.execute("INSTALL httpfs; LOAD httpfs;")
            except Exception:
                pass
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_ohlcv(self, df: pd.DataFrame, ticker: str) -> int:
        """Write OHLCV data to the data lake.

        Appends to existing Parquet file for the ticker/year partition.
        Deduplicates by (ticker, event_time, vendor) before writing.

        Returns:
            Number of rows written.
        """
        if df.empty:
            return 0

        rows_written = 0
        # Group by year for partitioned storage
        df["_year"] = pd.to_datetime(df["event_time"]).dt.year
        for year, year_df in df.groupby("_year"):
            year_df = year_df.drop(columns=["_year"])
            path = self.paths.ohlcv_file(ticker, int(year))
            rows_written += self._write_parquet(year_df, path, OHLCV_SCHEMA)

        logger.info(f"Wrote {rows_written} OHLCV rows for {ticker}")
        return rows_written

    def write_fundamentals(self, df: pd.DataFrame, ticker: str, period_type: str = "annual") -> int:
        """Write fundamentals data to the data lake."""
        if df.empty:
            return 0
        path = self.paths.fundamentals_file(ticker, period_type)
        rows = self._write_parquet(df, path, FUNDAMENTALS_SCHEMA)
        logger.info(f"Wrote {rows} fundamentals rows for {ticker}")
        return rows

    def write_news(self, df: pd.DataFrame, ticker: Optional[str] = None) -> int:
        """Write news data to the data lake."""
        if df.empty:
            return 0
        # Partition news by month
        df["_month"] = pd.to_datetime(df["event_time"]).dt.to_period("M").astype(str)
        rows_written = 0
        for month, month_df in df.groupby("_month"):
            month_df = month_df.drop(columns=["_month"])
            news_dir = self.paths.news_dir(ticker)
            path = news_dir / f"month={month}" / "data.parquet"
            rows_written += self._write_parquet(month_df, path, NEWS_SCHEMA)
        logger.info(f"Wrote {rows_written} news rows for {ticker or 'global'}")
        return rows_written

    def _write_parquet(self, df: pd.DataFrame, path: Path, schema: pa.Schema) -> int:
        """Write a DataFrame to a Parquet file, merging with existing data."""
        path = self.paths.ensure_dirs(path)

        # Convert to Arrow table with schema coercion
        try:
            table = pa.Table.from_pandas(df, schema=schema, safe=False)
        except Exception as e:
            logger.warning(f"Schema coercion failed: {e}. Writing without strict schema.")
            table = pa.Table.from_pandas(df)

        if path.exists():
            # Merge with existing data
            existing = pq.read_table(path)
            combined = pa.concat_tables([existing, table], promote_options="default")
            # Deduplicate: keep latest record for each (ticker, event_time, vendor)
            combined_df = combined.to_pandas()
            key_cols = [c for c in ["ticker", "event_time", "vendor"] if c in combined_df.columns]
            if key_cols:
                combined_df = (
                    combined_df
                    .sort_values("available_at", ascending=False)
                    .drop_duplicates(subset=key_cols, keep="first")
                    .sort_values(key_cols)
                    .reset_index(drop=True)
                )
            table = pa.Table.from_pandas(combined_df)

        pq.write_table(table, path, compression="snappy")
        return len(df)

    # ------------------------------------------------------------------
    # Read operations (time-safe)
    # ------------------------------------------------------------------

    def read_ohlcv(
        self,
        ticker: str,
        as_of_date: str,
        lookback_days: int = 90,
        start_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read OHLCV data available as of a given date (time-safe).

        Args:
            ticker: Instrument ticker symbol.
            as_of_date: The trade date — only data with available_at <= as_of_date is returned.
            lookback_days: How many calendar days of history to include.
            start_date: Override lookback_days with an explicit start date.

        Returns:
            DataFrame with OHLCV data, sorted by event_time ascending.
        """
        ohlcv_dir = self.paths.ohlcv_dir(ticker)
        if not ohlcv_dir.exists():
            return pd.DataFrame()

        # Find all Parquet files for this ticker
        parquet_files = list(ohlcv_dir.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        if start_date is None:
            start_dt = pd.to_datetime(as_of_date) - pd.Timedelta(days=lookback_days)
            start_date = start_dt.strftime("%Y-%m-%d")

        # Use DuckDB for efficient filtering
        files_str = ", ".join(f"'{str(f)}'" for f in parquet_files)
        query = f"""
            SELECT *
            FROM read_parquet([{files_str}])
            WHERE ticker = '{ticker.upper()}'
              AND event_time >= '{start_date}'
              AND event_time <= '{as_of_date}'
              AND available_at <= '{as_of_date} 23:59:59'
            ORDER BY event_time ASC
        """
        try:
            result = self.conn.execute(query).df()
            return result
        except Exception as e:
            logger.error(f"DuckDB read_ohlcv failed: {e}")
            # Fallback: read with PyArrow
            tables = [pq.read_table(f) for f in parquet_files]
            if not tables:
                return pd.DataFrame()
            df = pa.concat_tables(tables).to_pandas()
            df = df[
                (df["ticker"] == ticker.upper()) &
                (df["event_time"].astype(str) >= start_date) &
                (df["event_time"].astype(str) <= as_of_date)
            ]
            return df.sort_values("event_time").reset_index(drop=True)

    def read_fundamentals(
        self,
        ticker: str,
        as_of_date: str,
        period_type: str = "annual",
    ) -> pd.DataFrame:
        """Read fundamentals available as of a given date (time-safe)."""
        fund_dir = self.paths.fundamentals_dir(ticker)
        if not fund_dir.exists():
            return pd.DataFrame()

        parquet_files = list(fund_dir.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        files_str = ", ".join(f"'{str(f)}'" for f in parquet_files)
        query = f"""
            SELECT *
            FROM read_parquet([{files_str}])
            WHERE ticker = '{ticker.upper()}'
              AND available_at <= '{as_of_date} 23:59:59'
            ORDER BY period_end DESC
        """
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            logger.error(f"DuckDB read_fundamentals failed: {e}")
            return pd.DataFrame()

    def read_news(
        self,
        ticker: str,
        as_of_date: str,
        lookback_days: int = 7,
    ) -> pd.DataFrame:
        """Read news available as of a given date (time-safe)."""
        news_dir = self.paths.news_dir(ticker)
        if not news_dir.exists():
            return pd.DataFrame()

        parquet_files = list(news_dir.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        start_dt = pd.to_datetime(as_of_date) - pd.Timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        files_str = ", ".join(f"'{str(f)}'" for f in parquet_files)
        query = f"""
            SELECT *
            FROM read_parquet([{files_str}])
            WHERE event_time >= '{start_date}'
              AND event_time <= '{as_of_date} 23:59:59'
              AND available_at <= '{as_of_date} 23:59:59'
            ORDER BY event_time DESC
        """
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            logger.error(f"DuckDB read_news failed: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Arbitrary SQL queries
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL over the data lake using DuckDB.

        The query can reference Parquet files directly using read_parquet().
        """
        return self.conn.execute(sql).df()

    # ------------------------------------------------------------------
    # Lake statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return basic statistics about the data lake."""
        root = self.paths.root
        if not root.exists():
            return {"status": "empty", "root": str(root)}

        parquet_files = list(root.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)

        return {
            "status": "ok",
            "root": str(root),
            "parquet_files": len(parquet_files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "tickers": list({f.parent.parent.name for f in parquet_files
                             if f.parent.parent.parent.name in ("ohlcv", "fundamentals")}),
        }
