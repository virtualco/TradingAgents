"""Walk-forward validation engine for TradingAgents.

Walk-forward validation (WFV) is the gold standard for evaluating trading
strategies. Unlike simple train/test splits, WFV simulates real-world
deployment by:

1. Training on an expanding window of historical data
2. Testing on the next out-of-sample period
3. Rolling forward and repeating

This prevents lookahead bias and gives a realistic estimate of live
performance. The engine supports both:
- Quick Mode: rule-based signals (no LLM calls, fast)
- Deep Mode: LLM-powered signals (realistic but slow/expensive)

Walk-forward structure (expanding window):
    |------ train_1 ------|-- test_1 --|
    |-------- train_2 --------|-- test_2 --|
    |---------- train_3 ----------|-- test_3 --|
    ...
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .strategy_rules import (
    MultiRoleSignals,
    SignalStrength,
    compute_multi_role_signals,
)
from .signal_registry import SignalDirection, SignalRecord, SignalRegistry, SignalStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward validation run."""
    ticker: str
    start_date: str          # First date of the full evaluation period
    end_date: str            # Last date of the full evaluation period
    test_window_days: int = 21        # ~1 month of trading days
    min_train_days: int = 252         # ~1 year minimum training period
    step_days: int = 21               # Roll forward by this many days each fold
    signal_horizon_days: int = 45     # How many days to hold a position
    active_roles: Optional[List[str]] = None  # None = all roles
    mode: str = "quick"              # "quick" (rule-based) or "deep" (LLM)
    pipeline_version: str = "0.2.4"


@dataclass
class WalkForwardFold:
    """A single fold in the walk-forward validation."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    signals: List[SignalRecord] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Results from a complete walk-forward validation run."""
    ticker: str
    config: WalkForwardConfig
    folds: List[WalkForwardFold] = field(default_factory=list)
    aggregate_metrics: Dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    completed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def summary(self) -> str:
        m = self.aggregate_metrics
        if not m:
            return f"Walk-forward result for {self.ticker}: no metrics computed"
        return (
            f"Walk-Forward Validation: {self.ticker}\n"
            f"  Folds: {len(self.folds)}\n"
            f"  Total signals: {m.get('total_signals', 0)}\n"
            f"  Win rate: {m.get('win_rate', 0):.1%}\n"
            f"  Avg return per signal: {m.get('avg_return', 0):.2%}\n"
            f"  Sharpe proxy: {m.get('sharpe_proxy', 0):.2f}\n"
            f"  Max drawdown: {m.get('max_drawdown', 0):.2%}"
        )


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """Runs walk-forward validation using rule-based or LLM signals.

    Usage (Quick Mode — no LLM calls):
        engine = WalkForwardEngine(data_lake=lake)
        config = WalkForwardConfig(
            ticker="AAPL",
            start_date="2024-01-01",
            end_date="2026-04-28",
            mode="quick",
        )
        result = engine.run(config)
        print(result.summary())
    """

    def __init__(
        self,
        data_lake=None,
        signal_registry: Optional[SignalRegistry] = None,
        llm_runner: Optional[Callable] = None,
    ):
        """
        Args:
            data_lake: DataLake instance for fetching price data.
            signal_registry: SignalRegistry for persisting results.
            llm_runner: Callable for Deep Mode LLM signal generation.
                        Signature: (ticker, trade_date, ohlcv, fundamentals, news) -> AgentThesis
        """
        self.data_lake = data_lake
        self.registry = signal_registry
        self.llm_runner = llm_runner

    def run(self, config: WalkForwardConfig) -> WalkForwardResult:
        """Execute the full walk-forward validation."""
        logger.info(f"Starting walk-forward validation for {config.ticker} "
                    f"({config.start_date} to {config.end_date}, mode={config.mode})")

        result = WalkForwardResult(ticker=config.ticker, config=config)

        # Generate fold dates
        folds = self._generate_folds(config)
        if not folds:
            logger.warning("No folds generated — check date range and window sizes")
            return result

        logger.info(f"Generated {len(folds)} folds")

        # Fetch full price history once (time-safe reads happen per fold)
        all_ohlcv = self._fetch_ohlcv(config.ticker, config.start_date, config.end_date)
        if all_ohlcv.empty:
            logger.error(f"No price data for {config.ticker}")
            return result

        # Run each fold
        for fold in folds:
            logger.info(f"Processing fold {fold.fold_id}: test {fold.test_start} to {fold.test_end}")
            fold = self._run_fold(fold, config, all_ohlcv)
            result.folds.append(fold)

        # Compute aggregate metrics
        result.aggregate_metrics = self._aggregate_metrics(result.folds)

        logger.info(f"Walk-forward complete: {result.summary()}")
        return result

    def _generate_folds(self, config: WalkForwardConfig) -> List[WalkForwardFold]:
        """Generate fold date ranges."""
        folds = []
        start = pd.to_datetime(config.start_date)
        end = pd.to_datetime(config.end_date)
        min_train = timedelta(days=config.min_train_days)
        test_window = timedelta(days=config.test_window_days)
        step = timedelta(days=config.step_days)

        fold_id = 0
        test_start = start + min_train

        while test_start + test_window <= end:
            test_end = test_start + test_window
            folds.append(WalkForwardFold(
                fold_id=fold_id,
                train_start=config.start_date,
                train_end=(test_start - timedelta(days=1)).strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=min(test_end, end).strftime("%Y-%m-%d"),
            ))
            test_start += step
            fold_id += 1

        return folds

    def _run_fold(
        self,
        fold: WalkForwardFold,
        config: WalkForwardConfig,
        all_ohlcv: pd.DataFrame,
    ) -> WalkForwardFold:
        """Run a single fold: generate signals on test dates, then score outcomes."""
        # Get test period trading dates
        test_dates = self._get_trading_dates(all_ohlcv, fold.test_start, fold.test_end)

        for trade_date in test_dates:
            # Get data available as of trade_date (time-safe)
            ohlcv_as_of = all_ohlcv[
                (all_ohlcv["event_time"].astype(str) <= trade_date) &
                (all_ohlcv["available_at"].astype(str) <= trade_date + " 23:59:59")
            ].copy()

            if len(ohlcv_as_of) < 20:
                continue

            # Generate signal
            if config.mode == "quick":
                signal = self._generate_quick_signal(
                    config.ticker, trade_date, ohlcv_as_of, config
                )
            else:
                signal = self._generate_deep_signal(
                    config.ticker, trade_date, ohlcv_as_of, config
                )

            if signal is not None:
                fold.signals.append(signal)

        # Score outcomes: look up actual prices after horizon
        fold.signals = self._score_outcomes(fold.signals, all_ohlcv, config.signal_horizon_days)

        # Compute fold metrics
        fold.metrics = self._compute_fold_metrics(fold.signals)

        # Persist to registry if available
        if self.registry:
            for sig in fold.signals:
                self.registry.save(sig)

        return fold

    def _generate_quick_signal(
        self,
        ticker: str,
        trade_date: str,
        ohlcv: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> Optional[SignalRecord]:
        """Generate a rule-based signal (no LLM calls)."""
        multi_signals = compute_multi_role_signals(
            ticker=ticker,
            trade_date=trade_date,
            ohlcv=ohlcv,
            fundamentals=pd.DataFrame(),  # No fundamentals in quick mode
            news=pd.DataFrame(),
            active_roles=config.active_roles or ["TechnicalAnalyst", "SentimentAnalyst"],
        )

        score = multi_signals.ensemble_score
        signal_strength = multi_signals.ensemble_signal

        # Only generate a signal if conviction is sufficient
        if abs(score) < 0.15:
            return None

        direction = SignalDirection.LONG if score > 0 else SignalDirection.SHORT
        entry_price = ohlcv["close"].iloc[-1]

        # Simple stop/target based on ATR proxy
        recent_closes = ohlcv["close"].tail(14)
        atr_proxy = recent_closes.diff().abs().mean()
        stop_distance = max(atr_proxy * 2, entry_price * 0.03)
        target_distance = stop_distance * 2  # 2:1 reward/risk

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + target_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - target_distance

        role_scores = {}
        if multi_signals.technical:
            role_scores["TechnicalAnalyst"] = multi_signals.technical.composite_score
        if multi_signals.fundamental:
            role_scores["FundamentalAnalyst"] = multi_signals.fundamental.composite_score
        if multi_signals.sentiment:
            role_scores["SentimentAnalyst"] = multi_signals.sentiment.composite_score

        return SignalRecord(
            signal_id=str(uuid.uuid4()),
            ticker=ticker,
            trade_date=trade_date,
            pipeline_version=config.pipeline_version,
            direction=direction,
            conviction=abs(score),
            target_horizon_days=config.signal_horizon_days,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            active_roles=config.active_roles or ["TechnicalAnalyst", "SentimentAnalyst"],
            role_scores=role_scores,
            ensemble_score=score,
            executive_summary=f"Quick mode signal: {signal_strength.value} (score: {score:+.2f})",
        )

    def _generate_deep_signal(
        self,
        ticker: str,
        trade_date: str,
        ohlcv: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> Optional[SignalRecord]:
        """Generate an LLM-powered signal (Deep Mode)."""
        if self.llm_runner is None:
            logger.warning("Deep mode requested but no llm_runner provided. Falling back to quick mode.")
            return self._generate_quick_signal(ticker, trade_date, ohlcv, config)

        try:
            thesis = self.llm_runner(ticker, trade_date, ohlcv, pd.DataFrame(), pd.DataFrame())
            if thesis is None:
                return None

            direction = SignalDirection.LONG if thesis.signal.direction.value == "long" else SignalDirection.SHORT

            return SignalRecord(
                signal_id=str(uuid.uuid4()),
                ticker=ticker,
                trade_date=trade_date,
                pipeline_version=config.pipeline_version,
                direction=direction,
                conviction=thesis.confidence,
                target_horizon_days=thesis.signal.target_horizon_days or config.signal_horizon_days,
                entry_price=thesis.signal.entry_price,
                stop_loss=thesis.signal.stop_loss,
                take_profit=thesis.signal.take_profit,
                active_roles=config.active_roles or [],
                ensemble_score=thesis.confidence if direction == SignalDirection.LONG else -thesis.confidence,
                executive_summary=thesis.executive_summary,
                investment_thesis=thesis.investment_thesis,
                invalidation_condition=thesis.invalidation_condition,
                debate_winner=thesis.debate_winner,
            )
        except Exception as e:
            logger.error(f"Deep mode signal generation failed for {ticker} on {trade_date}: {e}")
            return None

    def _score_outcomes(
        self,
        signals: List[SignalRecord],
        all_ohlcv: pd.DataFrame,
        horizon_days: int,
    ) -> List[SignalRecord]:
        """Score signal outcomes using actual price data."""
        for signal in signals:
            if signal.entry_price is None:
                continue

            # Find the exit price (close price at horizon)
            horizon_date = (
                pd.to_datetime(signal.trade_date) + timedelta(days=horizon_days)
            ).strftime("%Y-%m-%d")

            future_prices = all_ohlcv[
                all_ohlcv["event_time"].astype(str) > signal.trade_date
            ].sort_values("event_time")

            if future_prices.empty:
                continue

            # Check for stop/target hit first
            hit_stop = False
            hit_target = False
            exit_price = None
            exit_date = None

            for _, row in future_prices.iterrows():
                row_date = str(row["event_time"])
                if row_date > horizon_date:
                    break

                low = row.get("low", row["close"])
                high = row.get("high", row["close"])

                if signal.direction == SignalDirection.LONG:
                    if signal.stop_loss and low <= signal.stop_loss:
                        hit_stop = True
                        exit_price = signal.stop_loss
                        exit_date = row_date
                        break
                    if signal.take_profit and high >= signal.take_profit:
                        hit_target = True
                        exit_price = signal.take_profit
                        exit_date = row_date
                        break
                else:  # SHORT
                    if signal.stop_loss and high >= signal.stop_loss:
                        hit_stop = True
                        exit_price = signal.stop_loss
                        exit_date = row_date
                        break
                    if signal.take_profit and low <= signal.take_profit:
                        hit_target = True
                        exit_price = signal.take_profit
                        exit_date = row_date
                        break

            # If no stop/target hit, use horizon close
            if exit_price is None:
                horizon_rows = future_prices[future_prices["event_time"].astype(str) <= horizon_date]
                if not horizon_rows.empty:
                    last_row = horizon_rows.iloc[-1]
                    exit_price = last_row["close"]
                    exit_date = str(last_row["event_time"])

            if exit_price is not None and signal.entry_price > 0:
                actual_return = (exit_price - signal.entry_price) / signal.entry_price
                if signal.direction == SignalDirection.SHORT:
                    actual_return = -actual_return

                signal.status = SignalStatus.CLOSED
                signal.exit_price = round(exit_price, 2)
                signal.exit_date = exit_date
                signal.actual_return = round(actual_return, 4)
                signal.hit_target = hit_target
                signal.hit_stop = hit_stop

        return signals

    def _compute_fold_metrics(self, signals: List[SignalRecord]) -> Dict:
        """Compute metrics for a single fold."""
        closed = [s for s in signals if s.status == SignalStatus.CLOSED and s.actual_return is not None]
        if not closed:
            return {"total_signals": len(signals), "closed": 0}

        returns = pd.Series([s.actual_return for s in closed])
        wins = (returns > 0).sum()

        return {
            "total_signals": len(signals),
            "closed": len(closed),
            "win_rate": round(wins / len(closed), 3),
            "avg_return": round(returns.mean(), 4),
            "std_return": round(returns.std(), 4),
            "sharpe_proxy": round(returns.mean() / returns.std(), 3) if returns.std() > 0 else 0,
        }

    def _aggregate_metrics(self, folds: List[WalkForwardFold]) -> Dict:
        """Aggregate metrics across all folds."""
        all_signals = [s for fold in folds for s in fold.signals]
        closed = [s for s in all_signals if s.status == SignalStatus.CLOSED and s.actual_return is not None]

        if not closed:
            return {"total_signals": len(all_signals), "closed": 0}

        returns = pd.Series([s.actual_return for s in closed])
        wins = (returns > 0).sum()
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max

        return {
            "total_signals": len(all_signals),
            "closed": len(closed),
            "folds": len(folds),
            "win_rate": round(wins / len(closed), 3),
            "avg_return": round(returns.mean(), 4),
            "median_return": round(returns.median(), 4),
            "std_return": round(returns.std(), 4),
            "sharpe_proxy": round(returns.mean() / returns.std(), 3) if returns.std() > 0 else 0,
            "max_drawdown": round(drawdowns.min(), 4),
            "total_return": round(cum_returns.iloc[-1] - 1, 4),
            "hit_target_rate": round(sum(1 for s in closed if s.hit_target) / len(closed), 3),
            "hit_stop_rate": round(sum(1 for s in closed if s.hit_stop) / len(closed), 3),
        }

    def _fetch_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from the data lake or yfinance."""
        if self.data_lake is not None:
            df = self.data_lake.read_ohlcv(ticker, end_date, start_date=start_date, lookback_days=9999)
            if not df.empty:
                return df

        # Fallback: fetch directly from yfinance
        try:
            import yfinance as yf
            from tradingagents.dataflows.pit_schema import DataVendor, normalize_ohlcv
            ticker_obj = yf.Ticker(ticker.upper())
            raw = ticker_obj.history(start=start_date, end=end_date)
            if raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)
            return normalize_ohlcv(raw, ticker, DataVendor.YFINANCE)
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    def _get_trading_dates(
        self,
        ohlcv: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """Get actual trading dates within a range from the OHLCV data."""
        if ohlcv.empty:
            return []
        dates = ohlcv[
            (ohlcv["event_time"].astype(str) >= start_date) &
            (ohlcv["event_time"].astype(str) <= end_date)
        ]["event_time"].astype(str).tolist()
        return sorted(set(dates))
