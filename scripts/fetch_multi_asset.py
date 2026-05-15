"""
Fetch historical 1-hour OHLCV data from Binance for multiple crypto symbols.
Saves to data/historical/{SYMBOL}_USD_1h_2022-01-01_2026-01-01.parquet

Usage:
    python3 scripts/fetch_multi_asset.py
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("multi_fetch")

BASE_URL = "https://api.binance.com/api/v3/klines"

# All symbols to fetch (Binance pair → output label)
SYMBOLS = [
    ("SOLUSDT",  "SOL_USD"),
    ("AVAXUSDT", "AVAX_USD"),
    ("DOGEUSDT", "DOGE_USD"),
    ("BNBUSDT",  "BNB_USD"),
    ("XRPUSDT",  "XRP_USD"),
    ("LINKUSDT", "LINK_USD"),
]

START_DATE = "2022-01-01"
END_DATE   = "2026-01-01"
INTERVAL   = "1h"
LIMIT      = 1000


def dt_to_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch all klines in [start_ms, end_ms] with pagination."""
    all_klines = []
    current_ms = start_ms
    while current_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current_ms,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        for attempt in range(1, 4):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                klines = resp.json()
                break
            except Exception as e:
                log.warning(f"  {symbol} attempt {attempt} failed: {e}")
                time.sleep(3 * attempt)
                klines = []

        if not klines:
            break
        all_klines.extend(klines)
        current_ms = klines[-1][0] + 3_600_000
        if len(klines) < LIMIT:
            break
        time.sleep(0.25)

    return all_klines


def klines_to_df(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_one(symbol: str, label: str, out_dir: Path, start_ms: int, end_ms: int):
    out_path = out_dir / f"{label}_1h_{START_DATE}_{END_DATE}.parquet"
    if out_path.exists():
        df_existing = pd.read_parquet(out_path)
        log.info(f"{label}: cached ({len(df_existing)} rows)")
        return label, len(df_existing), True

    log.info(f"Fetching {label} ({symbol})...")
    klines = fetch_klines(symbol, start_ms, end_ms)
    if not klines:
        log.error(f"  No data for {symbol}")
        return label, 0, False

    df = klines_to_df(klines)
    df.to_parquet(out_path)
    log.info(f"  {label}: saved {len(df)} rows")
    return label, len(df), True


def main():
    out_dir = Path("data/historical")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ms = dt_to_ms(START_DATE)
    end_ms = dt_to_ms(END_DATE)

    # Fetch sequentially to respect Binance rate limits
    results = []
    for symbol, label in SYMBOLS:
        result = fetch_one(symbol, label, out_dir, start_ms, end_ms)
        results.append(result)

    print("\n" + "=" * 60)
    print("DATA ACQUISITION SUMMARY")
    print("=" * 60)
    for label, rows, success in results:
        status = f"{rows:,} rows" if success else "FAILED"
        print(f"  {label:12s}: {status}")
    print("=" * 60)

    # Verify all files
    all_ok = all(s for _, _, s in results)
    if all_ok:
        print(f"\nAll {len(results)} symbols fetched successfully!")
    else:
        failed = [l for l, _, s in results if not s]
        print(f"\nFailed: {failed}")


if __name__ == "__main__":
    main()
