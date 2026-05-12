# AutoResearch Program — TradingAgents Signal Quality

This experiment autonomously improves the signal generation, backtesting, and
decision-making quality of the TradingAgents research stack.

## Setup

1. **Run tag**: `apr29`  (branch: `autoresearch/apr29`)
2. **Init**: `bash /home/ubuntu/skills/autoresearch/scripts/init_loop.sh /home/ubuntu/TradingAgents apr29`
3. **Read all target files** before starting.
4. **Baseline**: Run eval command once to establish baseline score.
5. **Confirm and go.**

---

## Target Files

**What you CAN modify** (these files may be edited freely):

- `tradingagents/research/strategy_rules.py`   — Technical/Fundamental/Sentiment rule logic
- `tradingagents/research/walk_forward.py`      — Walk-forward validation engine
- `tradingagents/research/signal_registry.py`   — Signal persistence and scoring
- `tradingagents/backtest/engine.py`            — Event-driven backtest engine
- `tradingagents/backtest/analytics.py`         — Performance metrics (Sharpe, Sortino, etc.)
- `tradingagents/backtest/optimizer.py`         — Portfolio optimizer (mean-variance, risk parity)
- `tradingagents/backtest/risk_model.py`        — Factor risk model
- `tradingagents/backtest/stress.py`            — Stress testing scenarios
- `tradingagents/execution/order_manager.py`    — Pre-trade risk checks
- `tradingagents/execution/kill_switch.py`      — Circuit breakers
- `scripts/run_daily.py`                        — Daily runner pipeline

**What you CANNOT modify** (read-only — evaluation harness and interfaces):

- `tests/test_research_factory.py`    — Evaluation test suite (DO NOT TOUCH)
- `tests/test_backtest_engine.py`     — Evaluation test suite (DO NOT TOUCH)
- `tests/test_execution_engine.py`    — Evaluation test suite (DO NOT TOUCH)
- `tests/test_thesis_schema.py`       — Schema tests (DO NOT TOUCH)
- `tests/test_data_foundation.py`     — Data layer tests (DO NOT TOUCH)
- `tradingagents/agents/thesis_schema.py`       — Core schema (DO NOT TOUCH)
- `tradingagents/dataflows/pit_schema.py`       — Data schema (DO NOT TOUCH)
- `tradingagents/execution/observer.py`         — Observer interface (DO NOT TOUCH)
- `pyproject.toml`                              — No new dependencies without justification
- `drizzle/`                                    — Not applicable (Python project)

---

## Goal

**The goal is: maximise the composite autoresearch score.**

The score is a weighted combination of:

| Component | Weight | What it measures |
|---|---|---|
| Test pass rate | 40% | All 292 tests pass → 292 pts |
| Signal quality score | 30% | Ensemble signal sharpness, conviction calibration, coverage |
| Backtest quality score | 20% | Sharpe proxy, win rate, drawdown control in synthetic tests |
| Code simplicity score | 10% | Inverse of total lines in target files (simpler = better) |

**Optimization direction: maximize**

Baseline metric: **292.0** (all tests pass, signal/backtest scores at initial level)

The eval script outputs: `SCORE: <float>`

---

## Eval Command

```bash
bash /home/ubuntu/skills/autoresearch/scripts/run_eval.sh /home/ubuntu/TradingAgents
```

The eval script (`eval.sh` in repo root) runs:
1. `python3 -m pytest tests/test_research_factory.py tests/test_backtest_engine.py tests/test_execution_engine.py --tb=no -q` → test pass count
2. `python3 scripts/eval_signal_quality.py` → signal quality score
3. `python3 scripts/eval_backtest_quality.py` → backtest quality score
4. Combines into composite `SCORE: <value>`

---

## Research Directions

Explore these areas in rough priority order:

### Priority 1 — Signal Quality (highest impact on decision-making)
- **Better RSI thresholds**: The current RSI uses fixed 30/70 thresholds. Try adaptive thresholds based on rolling volatility (e.g., 25/75 in low-vol, 35/65 in high-vol regimes).
- **MACD parameter tuning**: Default (12, 26, 9) may not be optimal. Try (8, 21, 5) for faster signals or (19, 39, 9) for fewer false positives.
- **Bollinger Band width filter**: Only trade Bollinger signals when band width > 1.5× its 20-day average (avoid choppy markets).
- **Volume confirmation**: Require volume > 1.2× 20-day average for any BUY signal (institutional confirmation).
- **Signal combination logic**: Current ensemble uses simple average. Try conviction-weighted harmonic mean to penalise disagreement between roles.
- **Regime detection**: Add a simple market regime filter (trending vs ranging via ADX > 25) and only trade trend-following signals in trending regimes.

### Priority 2 — Backtest Quality (validates signal reliability)
- **Slippage model**: Current slippage is fixed. Try market-impact slippage = 0.1% × sqrt(order_size / ADV).
- **Position sizing**: Try Kelly criterion sizing (f* = edge / odds) capped at 20% max position.
- **Stop-loss optimisation**: Current stop is fixed. Try ATR-based stops (2× ATR) for volatility-adaptive risk management.
- **Holding period analysis**: Add per-signal holding period tracking to identify optimal exit timing.

### Priority 3 — Performance Analytics (better decision-making outputs)
- **Calmar ratio**: Ensure Calmar = annualised return / max drawdown is computed correctly for all time windows.
- **Rolling Sharpe**: Add 30-day and 90-day rolling Sharpe to analytics output for regime-aware performance.
- **Drawdown duration**: Track average and maximum drawdown duration in days (not just magnitude).
- **Tail risk metrics**: Add Cornish-Fisher VaR adjustment for non-normal return distributions.

### Priority 4 — Code Quality (simplicity and robustness)
- **Remove dead code**: Any unused helper functions or commented-out blocks in target files.
- **Consolidate duplicate logic**: The RSI calculation appears in both strategy_rules.py and analytics.py — extract to a shared utility.
- **Edge case handling**: Ensure all signal computations handle NaN, infinite, and zero-length series gracefully without silent failures.

---

## Constraints

- **Do not break any existing tests.** The test suite is the ground truth — if tests fail, revert.
- **Do not add new pip dependencies** unless absolutely necessary (and document why).
- **Do not change function signatures** of public APIs used by tests (e.g., `TechnicalStrategyRules.compute()`, `BacktestEngine.run()`).
- **Do not modify read-only files** listed above.
- **Keep each change atomic**: one hypothesis per commit. Do not bundle multiple changes.
- **Simplicity criterion**: All else being equal, simpler is better. Removing code and getting equal or better results is a great outcome.
- **Time budget**: Each eval run must complete within 5 minutes. If it exceeds 10 minutes, kill it and treat as failure.
- **No LLM API calls in eval**: The eval scripts must run without OpenAI/Anthropic keys (Quick Mode only).

---

## The Experiment Loop

LOOP FOREVER:

1. Read this program.md for current priorities
2. Examine the target files and recent results in results.tsv
3. Propose a hypothesis — a specific, testable improvement
4. Implement the change in the allowed target files
5. `git add -A && git commit -m "autoresearch: <description>"`
6. Run: `bash /home/ubuntu/skills/autoresearch/scripts/run_eval.sh /home/ubuntu/TradingAgents`
7. Parse the `SCORE: <value>` from output
8. Run: `python3 /home/ubuntu/skills/autoresearch/scripts/ratchet.py /home/ubuntu/TradingAgents <score> "<description>"`
9. The ratchet keeps the commit if score improved, reverts if not
10. Go to step 1

**NEVER STOP.** Run until manually interrupted.
**Timeout**: Kill eval if it exceeds 10 minutes, treat as failure.
**Crashes**: Fix typos and re-run. If fundamentally broken, revert and move on.
