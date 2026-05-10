"""
Dual-Regime Strategy Diagnostic — Deep failure mode analysis
Uses get_diagnostics() to expose all intermediate signals and identify
root causes of underperformance across 4 years of OOS data (2022–2026).
"""
import sys
sys.path.insert(0, '/home/ubuntu/TradingAgents')

import pandas as pd
import numpy as np
import json

from tradingagents.research.dual_regime_strategy import DualRegimeStrategy

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data(symbol: str) -> pd.DataFrame:
    path = f'/home/ubuntu/TradingAgents/data/historical/{symbol}_USD_1h_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df

# ── Vectorised Backtest ───────────────────────────────────────────────────────
def backtest(df: pd.DataFrame, signals: pd.Series, fee: float = 0.001) -> dict:
    """Fast vectorised backtest on pre-computed signals."""
    close = df['close'].values
    sig = signals.values

    equity = 1.0
    trades = []
    position = 0
    entry_price = 0.0
    entry_bar = 0

    for i in range(1, len(close)):
        if position == 0 and sig[i] != 0:
            position = sig[i]
            entry_price = close[i] * (1 + fee)
            entry_bar = i
        elif position != 0:
            # Exit on reversal signal or after max 48h hold
            if sig[i] == -position or (sig[i] == 0 and i - entry_bar > 48):
                exit_price = close[i] * (1 - fee)
                pnl = (exit_price / entry_price - 1) * position
                equity *= (1 + pnl)
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'hold_hours': i - entry_bar,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl': pnl,
                    'year': pd.Timestamp(df.index[entry_bar]).year,
                })
                position = 0

    if not trades:
        return {'total_return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0}

    trade_df = pd.DataFrame(trades)
    wins = (trade_df['pnl'] > 0).sum()
    win_rate = wins / len(trade_df)

    eq_curve = [1.0]
    for t in trades:
        eq_curve.append(eq_curve[-1] * (1 + t['pnl']))
    eq_arr = np.array(eq_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = abs(dd.min())

    pnls = trade_df['pnl'].values
    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(len(pnls)) if pnls.std() > 0 else 0

    return {
        'total_return': round((equity - 1) * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': len(trades),
        'win_rate': round(win_rate * 100, 1),
        'trades_by_year': trade_df.groupby('year').apply(
            lambda x: {'n': len(x), 'wr': round((x['pnl']>0).mean()*100,1), 'pnl': round(x['pnl'].sum()*100,2)}
        ).to_dict(),
    }

# ── Main Diagnostic ───────────────────────────────────────────────────────────
def main():
    findings = {}

    for symbol in ['BTC', 'ETH']:
        print(f"\n{'='*65}")
        print(f"  DIAGNOSING {symbol}USDT — 4yr OOS (2022–2026)")
        print(f"{'='*65}")

        df = load_data(symbol)
        strat = DualRegimeStrategy()
        diag = strat.get_diagnostics(df)
        signals = diag['combined_sig']

        # ── 1. Regime Distribution ──────────────────────────────────────────
        regime_counts = diag['regime'].value_counts()
        total = len(diag)
        print(f"\n[1] REGIME DISTRIBUTION ({total} bars)")
        for r, c in regime_counts.items():
            print(f"    {r:12s}: {c:6d} bars ({c/total*100:.1f}%)")

        # ── 2. Signal Generation Rate ───────────────────────────────────────
        mom_signals = (diag['momentum_sig'] != 0).sum()
        rev_signals = (diag['revert_sig'] != 0).sum()
        combined_signals = (diag['combined_sig'] != 0).sum()
        print(f"\n[2] SIGNAL GENERATION RATE")
        print(f"    Momentum sub-strategy:     {mom_signals} signals ({mom_signals/total*100:.2f}% of bars)")
        print(f"    Mean-reversion sub-strategy:{rev_signals} signals ({rev_signals/total*100:.2f}% of bars)")
        print(f"    Combined (after regime mask):{combined_signals} signals ({combined_signals/total*100:.2f}% of bars)")

        # ── 3. Regime Predictive Power ──────────────────────────────────────
        close = df['close']
        fwd_1h = close.pct_change(1).shift(-1)
        fwd_4h = close.pct_change(4).shift(-4)
        fwd_24h = close.pct_change(24).shift(-24)

        print(f"\n[3] REGIME PREDICTIVE POWER (avg forward return in regime)")
        print(f"    {'Regime':12s}  {'1h fwd':>10s}  {'4h fwd':>10s}  {'24h fwd':>10s}  {'Sharpe 4h':>10s}")
        for r in ['TRENDING', 'RANGING', 'TRANSITION']:
            mask = diag['regime'] == r
            f1 = fwd_1h[mask].dropna()
            f4 = fwd_4h[mask].dropna()
            f24 = fwd_24h[mask].dropna()
            sharpe_4h = f4.mean() / f4.std() * np.sqrt(252*24/4) if f4.std() > 0 else 0
            print(f"    {r:12s}  {f1.mean()*100:>10.4f}%  {f4.mean()*100:>10.4f}%  {f24.mean()*100:>10.4f}%  {sharpe_4h:>10.3f}")

        # ── 4. Hurst Distribution ───────────────────────────────────────────
        hurst = diag['hurst'].dropna()
        print(f"\n[4] HURST EXPONENT DISTRIBUTION")
        print(f"    Mean: {hurst.mean():.4f}  Std: {hurst.std():.4f}  Min: {hurst.min():.4f}  Max: {hurst.max():.4f}")
        print(f"    >0.55 (TRENDING):  {(hurst>0.55).mean()*100:.1f}%")
        print(f"    <0.45 (RANGING):   {(hurst<0.45).mean()*100:.1f}%")
        print(f"    0.45-0.55 (TRANS): {((hurst>=0.45)&(hurst<=0.55)).mean()*100:.1f}%")

        # ── 5. ADX Distribution ─────────────────────────────────────────────
        adx = diag['adx'].dropna()
        print(f"\n[5] ADX DISTRIBUTION")
        print(f"    Mean: {adx.mean():.2f}  Std: {adx.std():.2f}  >25: {(adx>25).mean()*100:.1f}%  <20: {(adx<20).mean()*100:.1f}%")

        # ── 6. Full Backtest ────────────────────────────────────────────────
        bt = backtest(df, signals, fee=0.001)
        print(f"\n[6] FULL 4YR BACKTEST (0.1% fees)")
        print(f"    Return: {bt['total_return']}%  Sharpe: {bt['sharpe']}  MaxDD: {bt['max_dd']}%")
        print(f"    Trades: {bt['n_trades']}  Win Rate: {bt['win_rate']}%")
        print(f"    Year-by-year:")
        for yr, stats in bt.get('trades_by_year', {}).items():
            print(f"      {yr}: {stats['n']} trades, WR={stats['wr']}%, PnL={stats['pnl']}%")

        # ── 7. Momentum-only backtest (ignore regime) ───────────────────────
        bt_mom = backtest(df, diag['momentum_sig'], fee=0.001)
        print(f"\n[7] MOMENTUM-ONLY (no regime filter)")
        print(f"    Return: {bt_mom['total_return']}%  Sharpe: {bt_mom['sharpe']}  Trades: {bt_mom['n_trades']}")

        # ── 8. Mean-reversion-only backtest ────────────────────────────────
        bt_rev = backtest(df, diag['revert_sig'], fee=0.001)
        print(f"\n[8] MEAN-REVERSION-ONLY (no regime filter)")
        print(f"    Return: {bt_rev['total_return']}%  Sharpe: {bt_rev['sharpe']}  Trades: {bt_rev['n_trades']}")

        # ── 9. Entry timing analysis ────────────────────────────────────────
        long_entries = diag[diag['combined_sig'] == 1].index
        short_entries = diag[diag['combined_sig'] == -1].index

        if len(long_entries) > 0:
            fwd_4h_long = fwd_4h.reindex(long_entries).dropna()
            print(f"\n[9] ENTRY TIMING QUALITY")
            print(f"    LONG entries ({len(long_entries)}): avg 4h fwd return = {fwd_4h_long.mean()*100:.4f}%  WR = {(fwd_4h_long>0).mean()*100:.1f}%")
        if len(short_entries) > 0:
            fwd_4h_short = (-fwd_4h).reindex(short_entries).dropna()
            print(f"    SHORT entries ({len(short_entries)}): avg 4h fwd return = {fwd_4h_short.mean()*100:.4f}%  WR = {(fwd_4h_short>0).mean()*100:.1f}%")

        findings[symbol] = {
            'regime_dist': {r: int(c) for r, c in regime_counts.items()},
            'signal_rate_pct': round(combined_signals/total*100, 3),
            'backtest': bt,
            'momentum_only': bt_mom,
            'meanrev_only': bt_rev,
            'hurst_mean': round(float(hurst.mean()), 4),
            'adx_mean': round(float(adx.mean()), 2),
        }

    # ── Summary of root causes ──────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  ROOT CAUSE SUMMARY")
    print(f"{'='*65}")
    for sym, f in findings.items():
        print(f"\n{sym}:")
        rd = f['regime_dist']
        total = sum(rd.values())
        trending_pct = rd.get('TRENDING', 0) / total * 100
        ranging_pct = rd.get('RANGING', 0) / total * 100
        trans_pct = rd.get('TRANSITION', 0) / total * 100
        print(f"  Regime: TRENDING={trending_pct:.1f}%  RANGING={ranging_pct:.1f}%  TRANSITION={trans_pct:.1f}%")
        print(f"  Signal rate: {f['signal_rate_pct']}% of bars")
        print(f"  Combined: {f['backtest']['total_return']}% return, Sharpe {f['backtest']['sharpe']}")
        print(f"  Mom-only: {f['momentum_only']['total_return']}% return, Sharpe {f['momentum_only']['sharpe']}")
        print(f"  Rev-only: {f['meanrev_only']['total_return']}% return, Sharpe {f['meanrev_only']['sharpe']}")

        if trans_pct > 70:
            print(f"  ⚠ CRITICAL: {trans_pct:.0f}% of bars are TRANSITION — strategy is mostly flat, starved of trades")
        if f['signal_rate_pct'] < 1.0:
            print(f"  ⚠ CRITICAL: Only {f['signal_rate_pct']}% signal rate — entry conditions too restrictive")
        if f['momentum_only']['sharpe'] > f['backtest']['sharpe']:
            print(f"  ⚠ Regime filter is HURTING momentum performance — filter is miscalibrated")
        if f['meanrev_only']['sharpe'] > 0.5:
            print(f"  ✓ Mean-reversion sub-strategy has positive Sharpe standalone")

    with open('/tmp/diagnostic_results.json', 'w') as fh:
        json.dump(findings, fh, indent=2, default=str)
    print(f"\nFull results → /tmp/diagnostic_results.json")

if __name__ == '__main__':
    main()
