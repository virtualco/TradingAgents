#!/usr/bin/env bash
# ============================================================
# TradingAgents AutoResearch Composite Evaluator
# Outputs: SCORE: <float>
# ============================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== TradingAgents AutoResearch Eval ==="
echo ""

# ── Step 1: Test pass rate (weight 40%) ──────────────────────────────────
echo "--- Test Suite ---"
TEST_OUTPUT=$(python3 -m pytest tests/test_research_factory.py tests/test_backtest_engine.py tests/test_execution_engine.py --tb=no -q 2>&1)
echo "$TEST_OUTPUT" | tail -5

# Parse pass count from "N passed" line
PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' | tail -1 || echo "0")
TOTAL_TESTS=92   # Expected: 45 research + 47 backtest = 92 core tests
TEST_SCORE=$(python3 -c "print(min(float('${PASSED}') / ${TOTAL_TESTS} * 100, 100))")
echo "  Test score: ${PASSED}/${TOTAL_TESTS} = ${TEST_SCORE}"
echo ""

# ── Step 2: Signal quality (weight 30%) ──────────────────────────────────
echo "--- Signal Quality ---"
SQ_OUTPUT=$(python3 scripts/eval_signal_quality.py 2>&1)
echo "$SQ_OUTPUT"
SQ_SCORE=$(echo "$SQ_OUTPUT" | grep -oP 'SIGNAL_QUALITY_SCORE: \K[\d.]+' | tail -1 || echo "0")
echo ""

# ── Step 3: Backtest quality (weight 20%) ─────────────────────────────────
echo "--- Backtest Quality ---"
BQ_OUTPUT=$(python3 scripts/eval_backtest_quality.py 2>&1)
echo "$BQ_OUTPUT"
BQ_SCORE=$(echo "$BQ_OUTPUT" | grep -oP 'BACKTEST_QUALITY_SCORE: \K[\d.]+' | tail -1 || echo "0")
echo ""

# ── Step 4: Code simplicity (weight 10%) ─────────────────────────────────
echo "--- Code Simplicity ---"
TARGET_FILES=(
    "tradingagents/research/strategy_rules.py"
    "tradingagents/research/walk_forward.py"
    "tradingagents/backtest/engine.py"
    "tradingagents/backtest/analytics.py"
    "tradingagents/execution/order_manager.py"
    "tradingagents/execution/kill_switch.py"
)
TOTAL_LINES=0
for f in "${TARGET_FILES[@]}"; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f")
        TOTAL_LINES=$((TOTAL_LINES + LINES))
    fi
done
# Baseline is 3236 lines. Score: 100 if lines <= 3000, decreases above that
SIMPLICITY_SCORE=$(python3 -c "
baseline = 3236
lines = ${TOTAL_LINES}
score = max(0, min(100, 100 - (lines - 3000) / 30))
print(f'{score:.1f}')
")
echo "  Total lines in target files: ${TOTAL_LINES} → simplicity score: ${SIMPLICITY_SCORE}"
echo ""

# ── Composite Score ───────────────────────────────────────────────────────
COMPOSITE=$(python3 -c "
test_score = float('${TEST_SCORE}')
sq_score   = float('${SQ_SCORE}')
bq_score   = float('${BQ_SCORE}')
simp_score = float('${SIMPLICITY_SCORE}')

composite = (
    0.40 * test_score +
    0.30 * sq_score   +
    0.20 * bq_score   +
    0.10 * simp_score
)
print(f'{composite:.4f}')
")

echo "=== COMPOSITE SCORES ==="
echo "  Test pass rate (40%):    ${TEST_SCORE}"
echo "  Signal quality (30%):    ${SQ_SCORE}"
echo "  Backtest quality (20%):  ${BQ_SCORE}"
echo "  Code simplicity (10%):   ${SIMPLICITY_SCORE}"
echo ""
echo "SCORE: ${COMPOSITE}"
