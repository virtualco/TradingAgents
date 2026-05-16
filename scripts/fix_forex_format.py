#!/usr/bin/env python3
"""Reformat forex parquet files to match the expected format for eval_multi_asset."""
import pandas as pd
import os

DATA_DIR = '/home/ubuntu/TradingAgents/data/historical'

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']

for pair in PAIRS:
    path = f'{DATA_DIR}/{pair}_1d_2022-01-01_2026-01-01.parquet'
    df = pd.read_parquet(path)
    
    # Convert to datetime index matching SPY format
    df.index = pd.to_datetime(df['datetime'])
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Filter to 2022+ to match other assets
    df = df[df.index >= '2022-01-01']
    
    # Save back
    df.to_parquet(path)
    print(f"{pair}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    print(f"  Columns: {list(df.columns)}, Index: {df.index.name}")
