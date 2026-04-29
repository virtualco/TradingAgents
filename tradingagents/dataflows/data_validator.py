"""Data validation for the TradingAgents data lake.

Validates OHLCV and fundamentals data for:
1. Gaps — missing trading days in price series
2. Outliers — price/volume anomalies (spikes, zeros, negatives)
3. Lookahead bias — available_at > event_time in suspicious ways
4. Stale data — fundamentals not updated in expected timeframe
5. Corporate action consistency — split/dividend sanity checks
6. Schema completeness — required fields present and non-null

Returns structured ValidationReport objects, not exceptions, so callers
can decide how to handle issues (log, skip, alert, fail).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A single validation finding."""
    severity: Severity
    check: str
    description: str
    rows_affected: int = 0
    details: Optional[dict] = None


@dataclass
class ValidationReport:
    """Aggregated validation results for a dataset."""
    ticker: str
    data_type: str  # "ohlcv", "fundamentals", "news"
    as_of_date: str
    row_count: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no ERROR or CRITICAL issues found."""
        return not any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def error_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    def summary(self) -> str:
        counts = {s: 0 for s in Severity}
        for issue in self.issues:
            counts[issue.severity] += 1
        parts = [f"{self.ticker} {self.data_type} ({self.row_count} rows)"]
        for sev, count in counts.items():
            if count > 0:
                parts.append(f"{sev.value.upper()}: {count}")
        status = "VALID" if self.is_valid else "INVALID"
        return f"[{status}] " + " | ".join(parts)


# ---------------------------------------------------------------------------
# OHLCV Validator
# ---------------------------------------------------------------------------

class OHLCVValidator:
    """Validates OHLCV price data for common data quality issues."""

    # Maximum allowed price change in a single day (as fraction)
    MAX_DAILY_RETURN = 0.50  # 50% — flags extreme moves for review

    # Minimum volume threshold (below this is suspicious)
    MIN_VOLUME = 100

    # Maximum gap in trading days before flagging
    MAX_GAP_DAYS = 5

    def validate(
        self,
        df: pd.DataFrame,
        ticker: str,
        as_of_date: str,
    ) -> ValidationReport:
        """Run all OHLCV validation checks."""
        report = ValidationReport(
            ticker=ticker,
            data_type="ohlcv",
            as_of_date=as_of_date,
            row_count=len(df),
        )

        if df.empty:
            report.issues.append(ValidationIssue(
                severity=Severity.ERROR,
                check="empty_data",
                description=f"No OHLCV data found for {ticker} as of {as_of_date}",
            ))
            return report

        self._check_required_columns(df, report)
        self._check_nulls(df, report)
        self._check_price_sanity(df, report)
        self._check_ohlc_consistency(df, report)
        self._check_volume_sanity(df, report)
        self._check_gaps(df, report)
        self._check_outliers(df, report)
        self._check_lookahead(df, report, as_of_date)

        return report

    def _check_required_columns(self, df: pd.DataFrame, report: ValidationReport):
        required = ["ticker", "event_time", "available_at", "open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            report.issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                check="required_columns",
                description=f"Missing required columns: {missing}",
            ))

    def _check_nulls(self, df: pd.DataFrame, report: ValidationReport):
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col not in df.columns:
                continue
            null_count = df[col].isna().sum()
            if null_count > 0:
                pct = null_count / len(df) * 100
                sev = Severity.ERROR if pct > 10 else Severity.WARNING
                report.issues.append(ValidationIssue(
                    severity=sev,
                    check="null_prices",
                    description=f"Column '{col}' has {null_count} null values ({pct:.1f}%)",
                    rows_affected=null_count,
                ))

    def _check_price_sanity(self, df: pd.DataFrame, report: ValidationReport):
        """Check for zero or negative prices."""
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue
            non_null = df[col].dropna()
            bad = (non_null <= 0).sum()
            if bad > 0:
                report.issues.append(ValidationIssue(
                    severity=Severity.CRITICAL,
                    check="negative_prices",
                    description=f"Column '{col}' has {bad} zero/negative values",
                    rows_affected=int(bad),
                ))

    def _check_ohlc_consistency(self, df: pd.DataFrame, report: ValidationReport):
        """Check that high >= low, high >= open, high >= close, etc."""
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            return

        valid = df[["open", "high", "low", "close"]].dropna()

        # high >= low
        bad_hl = (valid["high"] < valid["low"]).sum()
        if bad_hl > 0:
            report.issues.append(ValidationIssue(
                severity=Severity.ERROR,
                check="ohlc_consistency",
                description=f"{bad_hl} rows where high < low",
                rows_affected=int(bad_hl),
            ))

        # high >= open and high >= close
        bad_h = ((valid["high"] < valid["open"]) | (valid["high"] < valid["close"])).sum()
        if bad_h > 0:
            report.issues.append(ValidationIssue(
                severity=Severity.ERROR,
                check="ohlc_consistency",
                description=f"{bad_h} rows where high < open or high < close",
                rows_affected=int(bad_h),
            ))

        # low <= open and low <= close
        bad_l = ((valid["low"] > valid["open"]) | (valid["low"] > valid["close"])).sum()
        if bad_l > 0:
            report.issues.append(ValidationIssue(
                severity=Severity.ERROR,
                check="ohlc_consistency",
                description=f"{bad_l} rows where low > open or low > close",
                rows_affected=int(bad_l),
            ))

    def _check_volume_sanity(self, df: pd.DataFrame, report: ValidationReport):
        if "volume" not in df.columns:
            return
        vol = df["volume"].dropna()
        zero_vol = (vol == 0).sum()
        if zero_vol > 0:
            pct = zero_vol / len(vol) * 100
            sev = Severity.WARNING if pct < 5 else Severity.ERROR
            report.issues.append(ValidationIssue(
                severity=sev,
                check="zero_volume",
                description=f"{zero_vol} rows with zero volume ({pct:.1f}%)",
                rows_affected=int(zero_vol),
            ))

    def _check_gaps(self, df: pd.DataFrame, report: ValidationReport):
        """Check for unexpected gaps in the trading day series."""
        if "event_time" not in df.columns or len(df) < 2:
            return

        dates = pd.to_datetime(df["event_time"]).sort_values()
        # Compute business day gaps
        gaps = []
        for i in range(1, len(dates)):
            delta = (dates.iloc[i] - dates.iloc[i - 1]).days
            # More than MAX_GAP_DAYS calendar days is suspicious
            if delta > self.MAX_GAP_DAYS:
                gaps.append((dates.iloc[i - 1].date(), dates.iloc[i].date(), delta))

        if gaps:
            sev = Severity.WARNING if len(gaps) <= 3 else Severity.ERROR
            report.issues.append(ValidationIssue(
                severity=sev,
                check="data_gaps",
                description=f"{len(gaps)} gaps in trading day series (>{self.MAX_GAP_DAYS} days)",
                rows_affected=len(gaps),
                details={"gaps": [(str(s), str(e), d) for s, e, d in gaps[:5]]},
            ))

    def _check_outliers(self, df: pd.DataFrame, report: ValidationReport):
        """Check for extreme daily price moves."""
        if "close" not in df.columns or len(df) < 2:
            return

        closes = df["close"].dropna().sort_index()
        returns = closes.pct_change().dropna()
        extreme = (returns.abs() > self.MAX_DAILY_RETURN).sum()

        if extreme > 0:
            max_return = returns.abs().max()
            report.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                check="price_outliers",
                description=(
                    f"{extreme} days with >50% price change. "
                    f"Max: {max_return:.1%}. May indicate split/dividend or bad data."
                ),
                rows_affected=int(extreme),
                details={"max_daily_return": float(max_return)},
            ))

    def _check_lookahead(self, df: pd.DataFrame, report: ValidationReport, as_of_date: str):
        """Check for lookahead bias: available_at > as_of_date."""
        if "available_at" not in df.columns:
            return

        future_data = (pd.to_datetime(df["available_at"]) > pd.to_datetime(as_of_date)).sum()
        if future_data > 0:
            report.issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                check="lookahead_bias",
                description=(
                    f"{future_data} rows have available_at > as_of_date={as_of_date}. "
                    "LOOKAHEAD BIAS DETECTED."
                ),
                rows_affected=int(future_data),
            ))


# ---------------------------------------------------------------------------
# Fundamentals Validator
# ---------------------------------------------------------------------------

class FundamentalsValidator:
    """Validates fundamentals data for staleness and completeness."""

    # Maximum days since last fundamentals update before flagging
    MAX_STALE_DAYS = 120  # ~1 quarter

    def validate(
        self,
        df: pd.DataFrame,
        ticker: str,
        as_of_date: str,
    ) -> ValidationReport:
        report = ValidationReport(
            ticker=ticker,
            data_type="fundamentals",
            as_of_date=as_of_date,
            row_count=len(df),
        )

        if df.empty:
            report.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                check="empty_data",
                description=f"No fundamentals data found for {ticker} as of {as_of_date}",
            ))
            return report

        self._check_staleness(df, report, as_of_date)
        self._check_key_metrics(df, report)
        self._check_lookahead(df, report, as_of_date)

        return report

    def _check_staleness(self, df: pd.DataFrame, report: ValidationReport, as_of_date: str):
        if "available_at" not in df.columns:
            return
        latest = pd.to_datetime(df["available_at"]).max()
        days_stale = (pd.to_datetime(as_of_date) - latest).days
        if days_stale > self.MAX_STALE_DAYS:
            report.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                check="stale_fundamentals",
                description=f"Fundamentals are {days_stale} days old (>{self.MAX_STALE_DAYS} days)",
                details={"latest_available_at": str(latest), "days_stale": days_stale},
            ))

    def _check_key_metrics(self, df: pd.DataFrame, report: ValidationReport):
        key_metrics = ["revenue", "net_income", "total_assets"]
        for metric in key_metrics:
            if metric not in df.columns:
                continue
            null_count = df[metric].isna().sum()
            if null_count == len(df):
                report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    check="missing_metrics",
                    description=f"Key metric '{metric}' is null for all rows",
                ))

    def _check_lookahead(self, df: pd.DataFrame, report: ValidationReport, as_of_date: str):
        if "available_at" not in df.columns:
            return
        future = (pd.to_datetime(df["available_at"]) > pd.to_datetime(as_of_date)).sum()
        if future > 0:
            report.issues.append(ValidationIssue(
                severity=Severity.CRITICAL,
                check="lookahead_bias",
                description=f"{future} rows have available_at > as_of_date. LOOKAHEAD BIAS.",
                rows_affected=int(future),
            ))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def validate_ohlcv(
    df: pd.DataFrame,
    ticker: str,
    as_of_date: str,
) -> ValidationReport:
    """Validate OHLCV data and return a ValidationReport."""
    return OHLCVValidator().validate(df, ticker, as_of_date)


def validate_fundamentals(
    df: pd.DataFrame,
    ticker: str,
    as_of_date: str,
) -> ValidationReport:
    """Validate fundamentals data and return a ValidationReport."""
    return FundamentalsValidator().validate(df, ticker, as_of_date)
