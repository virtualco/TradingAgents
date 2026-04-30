#!/usr/bin/env bash
# AutoResearch evaluation harness for TradingAgents.
# Outputs a single line: SHARPE=<value> DRAWDOWN=<value> TRADES=<value>
# Exit 0 = success, Exit 1 = failure (treat as metric=0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TRADINGAGENTS_DB="$REPO_ROOT/data/ar_eval_paper_trading.db"
export TRADINGAGENTS_CAPITAL=100000

# Clean slate for this eval run
rm -f "$TRADINGAGENTS_DB" "$REPO_ROOT/data/kill_switch_state.json"

# Fast sanity check first
cd "$REPO_ROOT"
if ! python3 -m pytest tests/test_execution_engine.py -x -q 2>&1 | grep -q "passed"; then
  echo "SANITY_FAIL"
  exit 1
fi

# Run backfill (89 days) + today's cycle
python3 scripts/backfill_observations.py --days 89 --threshold 0.20 \
  --db "$TRADINGAGENTS_DB" 2>/dev/null

SUMMARY=$(python3 scripts/run_daily.py --summary \
  --db "$TRADINGAGENTS_DB" \
  --reports "$REPO_ROOT/data/ar_eval_reports" 2>&1)

SHARPE=$(echo "$SUMMARY" | grep "Sharpe Ratio:" | grep -oP '[\d.]+' | head -1)
DRAWDOWN=$(echo "$SUMMARY" | grep "Max Drawdown:" | grep -oP '[\d.]+' | head -1)
TRADES=$(echo "$SUMMARY" | grep "Total Trades:" | grep -oP '\d+' | head -1)

SHARPE=${SHARPE:-0}
DRAWDOWN=${DRAWDOWN:-100}
TRADES=${TRADES:-0}

echo "SHARPE=$SHARPE DRAWDOWN=$DRAWDOWN TRADES=$TRADES"

# Fail if constraints violated (drawdown > 15% or trades < 20)
if (( $(echo "$DRAWDOWN > 15" | bc -l) )) || (( TRADES < 20 )); then
  exit 1
fi
