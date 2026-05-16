#!/usr/bin/env python3
"""
Fetch 4 years of daily OHLCV data for Forex pairs from Yahoo Finance.
Yahoo Finance uses symbols like EURUSD=X, GBPUSD=X, USDJPY=X
"""
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

PAIRS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'JPY=X',  # Yahoo uses JPY=X for USD/JPY
}

def fetch_forex_pair(client, pair_name, yahoo_symbol):
    """Fetch daily data for a forex pair."""
    print(f"\nFetching {pair_name} ({yahoo_symbol})...")
    
    try:
        response = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': yahoo_symbol,
            'region': 'US',
            'interval': '1d',
            'range': '5y',
            'includeAdjustedClose': True,
        })
        
        if not response or 'chart' not in response:
            print(f"  ERROR: No chart data in response")
            return None
            
        result = response['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': quotes['open'],
            'high': quotes['high'],
            'low': quotes['low'],
            'close': quotes['close'],
            'volume': quotes.get('volume', [0] * len(timestamps)),
        })
        
        # Drop rows with NaN prices
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Save to parquet
        out_path = os.path.join(DATA_DIR, f'{pair_name}_daily.parquet')
        df.to_parquet(out_path, index=False)
        
        print(f"  Saved {len(df)} bars to {out_path}")
        print(f"  Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        print(f"  Latest close: {df['close'].iloc[-1]:.5f}")
        
        return df
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    client = ApiClient()
    print("=== Forex Data Fetch ===")
    
    results = {}
    for pair_name, yahoo_symbol in PAIRS.items():
        df = fetch_forex_pair(client, pair_name, yahoo_symbol)
        if df is not None:
            results[pair_name] = df
    
    print(f"\n=== Summary ===")
    print(f"Successfully fetched: {list(results.keys())}")
    for name, df in results.items():
        print(f"  {name}: {len(df)} bars, {df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()}")

if __name__ == '__main__':
    main()
