# AutoResearch Program: Per-Asset Crypto Strategy Optimisation

## Goal
Maximise the **portfolio-level OOS Sharpe ratio** across 4 years of hourly crypto data (2022–2026).
The portfolio trades BTCUSDT and ETHUSDT with per-asset strategy routing.

**Primary metric**: Portfolio Sharpe Ratio (annualised, 0.1% transaction costs)
**Target**: Sharpe >= 1.5 (currently 0.76)

**Current breakdown**:
- BTC ATR Expansion: Sharpe 0.82, +94.4%, 23.8% DD, 46.5% WR, 623 trades, PF 1.19
- ETH Donchian Momentum: Sharpe 0.39, +36.0%, 29.6% DD, 44.5% WR, 339 trades, PF 1.11
- Portfolio: Sharpe 0.76, +76.2%, 16.1% DD, 45.8% WR, 962 trades

## Target Files
- `tradingagents/research/per_asset_router.py`

## What You Can Modify
The eval harness imports `BTC_CONFIG` and `ETH_CONFIG` dicts from per_asset_router.py.
It uses these parameters to generate signals and manage positions.

**Key parameters that affect the eval metric:**
- BTC_CONFIG: `expansion_mult`, `vol_mult`, `max_hold_bars`, `stop_mult`, `tp_mult`, `atr_period`
- ETH_CONFIG: `donchian_period`, `adx_min`, `adx_trend`, `vol_mult`, `hurst_min`, `vol_atr_max`, `max_hold_bars`, `stop_mult`, `tp_mult`, `atr_donchian_factor`

**If you add `atr_donchian_factor` to ETH_CONFIG**, the eval will use adaptive Donchian periods.
**If you set `vol_atr_max` to None**, the eval will disable the ATR volatility filter.

## Read-Only Files
- `scripts/eval_per_asset_oos.py` (evaluation harness — DO NOT MODIFY)
- `data/historical/BTC_USD_1h_2022-01-01_2026-01-01.parquet`
- `data/historical/ETH_USD_1h_2022-01-01_2026-01-01.parquet`

## Eval Command
```bash
python3 scripts/eval_per_asset_oos.py
```

## Metric Direction
maximize

## Metric Extraction
The eval script prints: `METRIC: <float>` on the last line.
Parse this float as the primary metric for the ratchet.

## Research Directions (ordered by expected impact)

1. **ETH strategy improvement** (Sharpe 0.39 — biggest opportunity):
   - Remove or relax `vol_atr_max` filter (set to None or increase to 0.08)
   - Reduce `hurst_min` below 0.48 (try 0.40-0.45)
   - Reduce `adx_trend` below 22 (try 18-20)
   - Reduce `vol_mult` below 2.0 (try 1.3-1.7)
   - Add `atr_donchian_factor` (try 0.3-0.7) for adaptive period
   - Shorten `donchian_period` (try 18-22)
   - Adjust `stop_mult`/`tp_mult` ratio (try 2.0/6.0 or 1.5/4.0)
   - Reduce `max_hold_bars` (try 24-36)

2. **BTC strategy improvement** (Sharpe 0.82 — already strong):
   - Reduce `expansion_mult` (try 2.5-2.8)
   - Adjust `stop_mult`/`tp_mult` (try 1.5/3.0 or 2.5/5.0)
   - Increase `max_hold_bars` (try 18-24)
   - Reduce `vol_mult` (try 1.2-1.4)

3. **Combined improvements**:
   - Both assets benefit from tighter stops + wider TPs (higher R:R)
   - Both benefit from more trades (relaxing filters)

## Constraints
- Module must export: `PerAssetRouter` class with `generate_signals(df, symbol)` method
- Must export `BTC_CONFIG` and `ETH_CONFIG` dicts at module level
- `generate_signals` returns a dict with key `signal` (one of: 'LONG', 'SHORT', 'FLAT')
- Must use pandas/numpy only (no external TA libs)
- No look-ahead bias — signals at bar[i] use only data[:i+1]
- Transaction costs are applied by the eval harness (0.1% per trade)
- Must handle both BTC and ETH with per-asset logic
- Keep the code clean and well-documented
- Avoid over-parameterisation (max 10 tuneable params per asset)
