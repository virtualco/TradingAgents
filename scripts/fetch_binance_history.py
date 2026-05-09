"""
Fetch historical 1-hour OHLCV data from Binance public API (no API key required).
Saves to data/historical/{symbol}_1h_{start}_{end}.parquet

Usage:
    python3 scripts/fetch_binance_history.py
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] binance_fetch: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("binance_fetch")

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = [("BTCUSDT", "BTC-USD"), ("ETHUSDT", "ETH-USD")]
START_DATE = "2022-01-01"
END_DATE   = "2026-01-01"
INTERVAL   = "1h"
LIMIT      = 1000  # Binance max per request


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
                log.warning(f"  Attempt {attempt} failed: {e}")
                time.sleep(3 * attempt)
                klines = []

        if not klines:
            break
        all_klines.extend(klines)
        # Next batch starts after the last candle's open time
        current_ms = klines[-1][0] + 3_600_000  # +1 hour in ms
        if len(klines) < LIMIT:
            break
        time.sleep(0.2)  # Rate limit courtesy

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


def main():
    out_dir = Path("data/historical")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ms = dt_to_ms(START_DATE)
    end_ms   = dt_to_ms(END_DATE)

    for symbol, label in SYMBOLS:
        out_path = out_dir / f"{label.replace('-','_')}_1h_{START_DATE}_{END_DATE}.parquet"
        if out_path.exists():
            df_existing = pd.read_parquet(out_path)
            log.info(f"{label}: cached ({len(df_existing)} rows) → {out_path}")
            continue

        log.info(f"Fetching {label} ({symbol}) from {START_DATE} to {END_DATE}...")
        klines = fetch_klines(symbol, start_ms, end_ms)
        if not klines:
            log.error(f"  No data for {symbol}")
            continue

        df = klines_to_df(klines)
        df.to_parquet(out_path)
        log.info(f"  Saved {len(df)} rows → {out_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
