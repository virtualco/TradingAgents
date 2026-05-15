# AutoResearch Program: Per-Asset Crypto Strategy Optimisation (v2)

## Goal
Maximise the **portfolio-level OOS Sharpe ratio** across 4 years of hourly crypto data (2022–2026).
The portfolio trades BTCUSDT and ETHUSDT with per-asset strategy routing.

**Primary metric**: Portfolio Sharpe Ratio (annualised, 0.1% transaction costs)
**Target**: Sharpe >= 1.5
**Current**: Sharpe 1.14 (grid-search optimised 2026-05-15)

**Current breakdown**:
- BTC ATR Expansion: Sharpe 1.09, +140.5%, 17.0% DD, 43.9% WR, 640 trades, PF 1.25
- ETH Donchian Momentum: Sharpe 0.70, +118.2%, 31.1% DD, 27.9% WR, 534 trades, PF 1.18
- Portfolio: Sharpe 1.14, +150.4%, 17.4% DD, 36.6% WR, 1174 trades

**Walk-Forward Validation** (12mo train / 6mo test, 6 windows):
- Avg Test Sharpe: 0.68 ± 1.27
- Positive windows: 4/6
- Overfit ratio: 44.5% (target: <30%)

## AutoResearch v2 Architecture

### Modes
```bash
# Phase 1: LLM proposes config changes as JSON (no full-file rewrites)
python3 scripts/autoresearch_v2.py --mode llm --iterations 10

# Phase 2: Bayesian parameter tuning with Optuna TPE
python3 scripts/autoresearch_v2.py --mode bayesian --trials 200

# Grid search (systematic, exhaustive)
python3 scripts/autoresearch_v2.py --mode grid

# Walk-forward validation (anti-overfitting check)
python3 scripts/autoresearch_v2.py --mode validate

# Full pipeline: Phase 1 → Phase 2 → Validation
python3 scripts/autoresearch_v2.py --mode full --iterations 5 --trials 200
```

### Key Improvements over v1
1. **Config-only JSON output** — LLM outputs `{"btc_config": {...}, "eth_config": {...}}` instead of full file rewrites. Eliminates syntax errors, reduces token usage by 80%.
2. **Bayesian optimisation** — Optuna TPE sampler explores parameter space 10x more efficiently than LLM-driven iteration. Found BTC Sharpe 1.28 in 200 trials (7 seconds).
3. **Walk-forward validation** — Rolling train/test windows detect overfitting. Reports overfit ratio and average OOS Sharpe.
4. **Research direction rotation** — 6 focused prompts rotate each iteration to ensure diversity (entry_filters, risk_management, breakout_sensitivity, eth_improvement, btc_improvement, combined_rr).
5. **Temperature cycling** — LLM temperature varies 0.8/0.9/1.0 across iterations for exploration diversity.
6. **Two-phase architecture** — LLM proposes structural changes (Phase 1), Bayesian tunes parameters (Phase 2).

## Target Files
- `tradingagents/research/per_asset_router.py`

## What You Can Modify
The eval harness imports `BTC_CONFIG` and `ETH_CONFIG` dicts from per_asset_router.py.

**BTC_CONFIG parameters**: `atr_period`, `expansion_mult`, `vol_mult`, `max_hold_bars`, `stop_mult`, `tp_mult`
**ETH_CONFIG parameters**: `donchian_period`, `adx_min`, `adx_trend`, `vol_mult`, `hurst_min`, `vol_atr_max`, `max_hold_bars`, `stop_mult`, `tp_mult`, `atr_donchian_factor`

Special values:
- `vol_atr_max: None` → disables ATR volatility filter
- `atr_donchian_factor: 0.5` → enables adaptive Donchian periods

## Read-Only Files
- `scripts/eval_per_asset_oos.py` (evaluation harness)
- `scripts/optimise_bayesian.py` (Bayesian optimiser)
- `scripts/walk_forward.py` (walk-forward validator)
- `data/historical/*.parquet`

## Eval Command
```bash
python3 scripts/eval_per_asset_oos.py
```

## Metric Direction
maximize

## Metric Extraction
The eval script prints: `METRIC: <float>` on the last line.

## Research Directions (ordered by expected impact)

1. **Reduce overfit ratio** (currently 44.5%, target <30%):
   - Parameters that are robust across windows vs window-specific
   - Simpler parameter sets may generalise better
   - Consider reducing number of active filters

2. **ETH strategy improvement** (Sharpe 0.70 — still the weak link):
   - Current best: dp=28, adx_min=12, adx_trend=24, vol=1.8, hurst=0.42, atr_max=0.04, hold=60, stop=1.2, tp=6.0
   - Win rate 27.9% is low — consider partial TP mechanism
   - Try adaptive Donchian with atr_donchian_factor

3. **BTC fine-tuning** (Sharpe 1.09 — strong but can improve):
   - Bayesian found 1.28 with atr=17, exp=3.8, vol=1.6, hold=12, stop=1.9, tp=8.0
   - Validate these params with walk-forward before adopting

4. **Weight optimisation**:
   - 60/40 BTC/ETH gives Sharpe 1.22 vs 1.14 at 50/50
   - 70/30 gives 1.27 but concentration risk

## Constraints
- Module must export: `PerAssetRouter` class with `generate_signals(df, symbol)` method
- Must export `BTC_CONFIG` and `ETH_CONFIG` dicts at module level
- `generate_signals` returns a dict with key `signal` (one of: 'LONG', 'SHORT', 'FLAT')
- Must use pandas/numpy only (no external TA libs)
- No look-ahead bias — signals at bar[i] use only data[:i+1]
- Transaction costs applied by eval harness (0.1% per trade)
- Keep code clean and well-documented
- Max 10 tuneable params per asset
