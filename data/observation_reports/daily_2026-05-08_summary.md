# TradingAgents Daily Observation Report — 2026-05-08

**Run Status:** SUCCESS (Exit Code: 0)
**Timestamp:** 2026-05-08T13:38:50 UTC
**Observation ID:** cc807561-fab4-4567-aeb5-2ee3d2666c03

---

## Portfolio Snapshot

| Metric | Value |
|---|---|
| NAV | $99,992.22 |
| Cash | $96,915.72 |
| Gross Long Exposure | $3,076.50 |
| Gross Short Exposure | $0.00 |
| Daily P&L | **-$7.78** |
| Total P&L | -$7.78 |
| Drawdown | 0.01% |
| Open Positions | 1 |

---

## Signal Generation — 10-Ticker Universe

| Ticker | Signal | Score / Conviction |
|---|---|---|
| AAPL | FLAT | score = -0.03 |
| MSFT | FLAT | score = -0.08 |
| NVDA | FLAT | score = +0.02 |
| GOOGL | FLAT | score = -0.05 |
| TSLA | FLAT | score = +0.14 |
| AMZN | FLAT | score = -0.03 |
| **META** | **LONG** | conviction = 0.62 (tech = 0.25) |
| JPM | FLAT | score = -0.02 |
| V | FLAT | score = +0.07 |
| UNH | FLAT | score = -0.03 |

**Actionable signals generated:** 1 (META — LONG)

---

## Order Execution

| Field | Value |
|---|---|
| Signals Received | 1 |
| Orders Submitted | 1 |
| Orders Approved | 0 |
| Orders Rejected | 1 |
| Orders Filled | 0 |
| Rejection Reason | Duplicate signal for META within 24h cooldown |

> **Note:** The META LONG signal was rejected on the second run due to the 24-hour cooldown deduplication guard. The initial fill from the first run (5 shares @ $616.24, notional $3,081.20, commission $3.08) remains the active position.

---

## Open Positions

| Ticker | Qty | Avg Cost | Last Price | Unrealized P&L |
|---|---|---|---|---|
| META | 5 | $616.24 | $615.30 | **-$4.70** |

---

## Risk & Safety Checks

| Check | Status |
|---|---|
| Kill Switch | OK — not triggered |
| Circuit Breaker | OK — not triggered |
| Reconciliation | CLEAN (0 breaks) |

---

## Observation Period Summary (Cumulative)

| Metric | Value |
|---|---|
| Period | 2026-05-08 → 2026-05-08 (2 days) |
| Initial NAV | $100,000.00 |
| Final NAV | $99,992.22 |
| Total Return | -0.01% |
| Annualized Return | -0.78% |
| Max Drawdown | 0.01% |
| Sharpe Ratio | -96.83 |
| Total Signals | 2 |
| Total Trades | 1 |
| Win Rate | 0.0% |
| Kill Switch Days | 0 |
| Circuit Breakers | 0 |
| **Ready for Live** | **NO** |

### Readiness Notes
- Observation period too short: 2/90 days elapsed
- Sharpe ratio (-96.83) below 0.5 threshold — insufficient trade history for meaningful calculation
- Only 1 trade executed; statistical significance requires more data

---

## Non-Fatal Warnings

- `SignalRegistry.save_signal` attribute missing — signal persistence to registry skipped (non-fatal; signals still processed via DailyObserver pipeline)

---

## Raw Report Path
`/home/ubuntu/TradingAgents/data/observation_reports/daily_2026-05-08.json`
