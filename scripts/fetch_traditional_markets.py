#!/usr/bin/env python3
"""
Fetch historical daily OHLCV data for traditional markets from Yahoo Finance.
Uses the Manus Data API (YahooFinance/get_stock_chart).

Daily data is reliable from Yahoo (unlike hourly which has gaps).
We use daily bars for traditional markets — the strategies adapt to the timeframe.

Saves to data/historical/{SYMBOL}_1d_2022-01-01_2026-01-01.parquet

Usage:
    python3 scripts/fetch_traditional_markets.py
"""
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

import pandas as pd
import numpy as np
from pathlib import Path
import time

SYMBOLS = [
    ('SPY', 'SPY'),    # S&P 500 ETF
    ('QQQ', 'QQQ'),    # Nasdaq 100 ETF
    ('GLD', 'GLD'),    # Gold ETF
]

OUT_DIR = Path('data/historical')


def fetch_yahoo_daily(symbol: str) -> pd.DataFrame:
    """Fetch daily data from Yahoo Finance via Manus API."""
    client = ApiClient()

    print(f"  Fetching {symbol} daily data (4 years)...")
    try:
        response = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': symbol,
            'region': 'US',
            'interval': '1d',
            'range': '5y',
            'includeAdjustedClose': True,
        })

        if response and 'chart' in response and 'result' in response['chart']:
            result = response['chart']['result'][0]
            timestamps = result.get('timestamp', [])
            quotes = result['indicators']['quote'][0]

            if timestamps:
                df = pd.DataFrame({
                    'open': quotes.get('open', []),
                    'high': quotes.get('high', []),
                    'low': quotes.get('low', []),
                    'close': quotes.get('close', []),
                    'volume': quotes.get('volume', []),
                }, index=pd.to_datetime(timestamps, unit='s', utc=True))

                df = df.dropna(subset=['close'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Filter to 2022-2026
                df = df[(df.index >= '2022-01-01') & (df.index < '2026-01-01')]
                print(f"    Got {len(df)} daily bars")
                return df

        print(f"    No data in response")
    except Exception as e:
        print(f"    Error: {e}")

    return pd.DataFrame()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, label in SYMBOLS:
        out_path = OUT_DIR / f'{label}_1d_2022-01-01_2026-01-01.parquet'

        if out_path.exists():
            df = pd.read_parquet(out_path)
            print(f"{label}: cached ({len(df)} rows)")
            continue

        print(f"\nFetching {label} ({symbol})...")
        df = fetch_yahoo_daily(symbol)

        if len(df) > 0:
            df.to_parquet(out_path)
            print(f"  Saved {len(df)} rows → {out_path}")
        else:
            print(f"  FAILED: No data for {symbol}")

        time.sleep(1)

    # Summary
    print(f"\n{'='*60}")
    print("TRADITIONAL MARKET DATA SUMMARY")
    print(f"{'='*60}")
    for _, label in SYMBOLS:
        path = OUT_DIR / f'{label}_1d_2022-01-01_2026-01-01.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  {label:6s}: {len(df):,} daily bars | "
                  f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"  {label:6s}: MISSING")


if __name__ == '__main__':
    main()
