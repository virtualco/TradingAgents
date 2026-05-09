# AutoResearch Program: Crypto Day Trading Algorithm

## Goal
Develop a high-frequency crypto day-trading algorithm targeting 30% weekly returns.
Uses intraday 1-hour candles for BTC-USD and ETH-USD over 60 days.
Metric to maximize: composite score (0-100) = 40% weekly return + 40% Sharpe + 20% win rate - drawdown penalty.

## Target Files
- `tradingagents/research/crypto_strategy.py`

## Read-Only Files
- `scripts/eval_crypto_strategy.py`
- `tradingagents/execution/observer.py`

## Eval Command
```bash
python3 scripts/eval_crypto_strategy.py
```

## Metric Direction
maximize

## Constraints
- Class must be CryptoDayTradingStrategy with generate_signals(df) returning pd.Series of +1/-1/0
- Must use pandas/numpy only (no external TA libs unless pre-installed)
- Include transaction costs in strategy design
- No look-ahead bias
