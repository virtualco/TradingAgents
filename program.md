# AutoResearch Program: Per-Asset Crypto Strategy Optimisation

## Goal

Maximise the **portfolio-level OOS Sharpe ratio** across 4 years of hourly crypto data (2022–2026).
The portfolio trades BTCUSDT and ETHUSDT with per-asset strategy routing.

**Primary metric**: Portfolio Sharpe Ratio (annualised, 0.1% transaction costs)
**Target**: Sharpe >= 1.5 (currently 1.03)
**Secondary metrics** (printed but not used for ratchet):
  - Portfolio total return (currently +75.1%)
  - Max drawdown (target <= 25%, currently 34.3%)
  - Win rate (target >= 45%)
  - Profit factor (target >= 1.5)

## Target Files

- `tradingagents/research/per_asset_router.py`

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

## Research Directions

1. **BTC strategy improvement** (highest leverage): BTC currently has Sharpe 0.52. Explore:
   - Adaptive ATR multiplier based on volatility regime
   - Keltner Channel breakout instead of raw ATR expansion
   - Volume profile analysis (VWAP deviation)
   - Multi-timeframe confirmation (4h trend + 1h entry)
   - Breakout fade filter (reject breakouts into strong resistance)

2. **ETH strategy refinement** (already strong at 1.55): Explore:
   - Donchian period adaptation based on ADX strength
   - Partial profit-taking at 1.5x ATR, trailing remainder
   - Re-entry logic after stop-out in same trend
   - Hurst-based position sizing (higher conviction when H > 0.6)

3. **Portfolio-level improvements**:
   - Correlation-aware position sizing (reduce when BTC/ETH correlated)
   - Dynamic capital allocation (shift weight to higher-Sharpe asset)
   - Regime-based portfolio heat (reduce total exposure in TRANSITION)

4. **Risk management**:
   - Tighter stop-losses during low-conviction signals
   - Time-based exit (close after N bars if not profitable)
   - Volatility scaling (reduce size when ATR/price > threshold)

## Constraints

- Module must export: `PerAssetRouter` class with `generate_signals(symbol, df)` method
- `generate_signals` returns a dict with keys: signal, regime, strategy, conviction, diagnostics
- signal must be one of: 'LONG', 'SHORT', 'FLAT'
- Must use pandas/numpy only (no external TA libs)
- No look-ahead bias — signals at bar[i] use only data[:i+1]
- Transaction costs are applied by the eval harness (0.1% per trade)
- Must handle both BTC and ETH with per-asset logic
- Keep the code clean and well-documented
- Avoid over-parameterisation (max 8 tuneable params per asset)
