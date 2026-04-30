# TradingAgents AutoResearch Program

## Goal

Maximise the **Sharpe ratio** of the paper trading system over a 90-day backfill
while keeping **max drawdown below 15%** and **total trades ≥ 20**.

Primary metric: `sharpe_ratio` (from observation period summary)
Secondary constraints: `max_drawdown < 0.15`, `total_trades >= 20`

## Eval Command

```bash
cd /home/ubuntu/TradingAgents && \
  rm -f data/paper_trading.db data/kill_switch_state.json && \
  export TRADINGAGENTS_DB=/home/ubuntu/TradingAgents/data/paper_trading.db && \
  export TRADINGAGENTS_CAPITAL=100000 && \
  python3 scripts/backfill_observations.py --days 89 --threshold 0.20 2>/dev/null && \
  python3 scripts/run_daily.py --summary 2>&1 | grep -E "Sharpe|Drawdown|Total Trades|Total Return"
```

Parse the output for:
- `Sharpe Ratio:` → primary metric (higher is better)
- `Max Drawdown:` → must stay below 15%
- `Total Trades:` → must be ≥ 20

## Baseline (as of 2026-04-29)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 2.863 |
| Max Drawdown | 1.54% |
| Total Trades | 20 |
| Total Return | +12.39% |

## Target Files (may be modified)

- `tradingagents/research/strategy_rules.py` — signal weights, RSI/MACD/BB thresholds
- `tradingagents/execution/observer.py` — `ObservationConfig` defaults
- `tradingagents/execution/kill_switch.py` — `KillSwitchConfig` defaults
- `scripts/run_daily.py` — signal score threshold (currently 0.20), conviction mapping

## Read-Only Files (must NOT be modified)

- `scripts/backfill_observations.py` — evaluation harness
- `tradingagents/execution/order_manager.py` — order execution logic
- `tradingagents/execution/reconciliation.py` — reconciliation engine
- `tests/` — test suite (run as sanity check before full eval)

## Fast Sanity Check (run before full eval)

```bash
cd /home/ubuntu/TradingAgents && python3 -m pytest tests/test_execution_engine.py -x -q 2>&1 | tail -5
```

If pytest fails → revert immediately, do not run full eval.

## Research Directions (priority order)

1. **Signal weight tuning** — adjust the composite score weights in `TechnicalStrategyRules.compute()`
   (currently equal-weighted across SMA crossover, RSI, MACD, Bollinger Band, volume, momentum).
   Try momentum-heavy (0.35 momentum, 0.25 RSI, 0.20 MACD, 0.10 SMA, 0.10 BB).

2. **RSI thresholds** — current oversold=30, overbought=70. Try 35/65 for earlier signals.

3. **MACD signal sensitivity** — adjust fast/slow/signal EMA periods (default 12/26/9).
   Try 8/21/5 for faster response.

4. **Conviction-to-position mapping** — current: `conviction = 0.5 + score * 0.5`.
   Try linear scaling with floor: `conviction = max(0.55, score * 0.8)`.

5. **Score threshold** — current: 0.20. Try 0.15 (more signals) or 0.25 (fewer, higher quality).

6. **Position sizing** — current: 5% max. Try Kelly-fraction sizing:
   `size = min(0.05, conviction * 0.08)`.

## Constraints

- Never set `max_position_size_pct > 0.10` (10% hard cap)
- Never disable the kill switch or circuit breaker
- Never modify the eval harness (`backfill_observations.py`)
- Keep changes to one logical unit per iteration (one file, one concept)
- Prefer simpler code — removing lines and getting equal results is a win

## Results Tracking

Results are logged to `autoresearch/results.tsv` by the ratchet script.
