#!/usr/bin/env python3
"""Debug why Sharpe is negative on uptrending data."""
import numpy as np, pandas as pd, sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradingagents.backtest.engine import BacktestEngine, BacktestConfig

rng = np.random.default_rng(1)
n = 252
trend, vol = 0.0008, 0.012
returns = rng.normal(trend, vol, n)
prices = 100.0 * np.cumprod(1 + returns)
dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(n)]
df = pd.DataFrame({'date': dates, 'close': prices, 'open': prices*0.999,
                   'high': prices*1.01, 'low': prices*0.99,
                   'volume': rng.integers(1_000_000, 5_000_000, n).astype(float)})

signals = pd.DataFrame([{'signal_id': f'sig-{i}', 'ticker': 'TEST',
    'trade_date': str(df.iloc[i]['date']), 'direction': 'long',
    'conviction': 0.75, 'stop_loss': None, 'take_profit': None}
    for i in range(0, len(df), 15)])

ohlcv = df.rename(columns={'date': 'event_time'})
ohlcv['event_time'] = pd.to_datetime(ohlcv['event_time'])

config = BacktestConfig(initial_capital=100_000, commission_pct=0.001,
    slippage_pct=0.001, max_position_pct=0.50, max_open_positions=2, max_hold_days=20)
engine = BacktestEngine(config=config)
result = engine.run(signals=signals, price_data={'TEST': ohlcv})

print(f'Trades executed: {len(result.trades)}')
total_pnl = 0
for t in result.trades:
    pnl = t.net_pnl or 0
    total_pnl += pnl
    print(f'  fill_date={t.fill_date} entry={t.fill_price:.2f} exit={t.exit_price} '
          f'exit_date={t.exit_date} hold={t.holding_days}d pnl={pnl:.2f}')

print(f'\nTotal net PnL: {total_pnl:.2f}')
eq = pd.Series([p.portfolio_value for p in result.equity_curve])
print(f'Start: {eq.iloc[0]:.0f}  End: {eq.iloc[-1]:.0f}  Return: {(eq.iloc[-1]/eq.iloc[0]-1)*100:.1f}%')
dr = eq.pct_change().dropna()
sharpe = dr.mean() / dr.std() * (252**0.5)
print(f'Sharpe (0% rfr): {sharpe:.3f}')
print(f'Daily returns: mean={dr.mean()*100:.4f}%  std={dr.std()*100:.4f}%')
print(f'\nEquity curve (first 30 days):')
for p in result.equity_curve[:30]:
    print(f'  {p.date}: {p.portfolio_value:.0f}')
