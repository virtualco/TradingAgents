#!/usr/bin/env python3
"""
ML Rolling Retrain Pipeline — TradingAgents v9.0
=================================================
Automated monthly retraining of the GBM regime detection model.

Features:
  - Fetches latest 6 months of hourly data from CCXT (Bybit)
  - Recomputes features and labels using the same pipeline as train_regime_model.py
  - Walk-forward validation: only promotes model if accuracy > threshold
  - Model versioning: saves with timestamp, keeps last 3 versions
  - Logs metrics: accuracy, feature importance drift, regime distribution shift
  - Designed to run as a systemd timer or cron job on GCE

Usage:
    python3 scripts/retrain_regime_model.py [--symbols BTC,ETH,SOL] [--threshold 0.85]

Environment:
    BYBIT_API_KEY, BYBIT_API_SECRET (for data fetching)
    MODEL_DIR (default: data/models/)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("retrain.log")]
)
log = logging.getLogger("retrain_pipeline")

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "AVAX", "LINK", "XRP", "BNB", "DOGE"]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "data/models"))
MAX_VERSIONS = 3
DEFAULT_THRESHOLD = 0.85
LOOKBACK_DAYS = 180  # 6 months


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_ohlcv_ccxt(symbol: str, days: int = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """Fetch hourly OHLCV data from Bybit via CCXT."""
    try:
        import ccxt
    except ImportError:
        log.error("ccxt not installed")
        return None

    exchange = ccxt.bybit({"enableRateLimit": True})
    ccxt_symbol = f"{symbol}/USDT:USDT"

    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    limit = 1000

    log.info(f"Fetching {symbol} OHLCV ({days} days)...")

    while True:
        try:
            candles = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1h", since=since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if len(candles) < limit:
                break
            time.sleep(0.5)  # Rate limit
        except Exception as e:
            log.warning(f"Error fetching {symbol}: {e}")
            break

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    log.info(f"  {symbol}: {len(df)} candles fetched")
    return df


def fetch_all_symbols(symbols: List[str], days: int = LOOKBACK_DAYS) -> Dict[str, pd.DataFrame]:
    """Fetch data for all symbols."""
    data = {}
    for sym in symbols:
        df = fetch_ohlcv_ccxt(sym, days)
        if df is not None and len(df) > 100:
            data[sym] = df
        else:
            log.warning(f"Insufficient data for {sym}, skipping")
    return data


# ── Feature Engineering (mirrors ml_regime_detector.py) ───────────────────────

def compute_features(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Compute the same 7 features used by the regime detector."""
    features = pd.DataFrame(index=df.index)

    returns = df["close"].pct_change()
    log_returns = np.log(df["close"] / df["close"].shift(1))

    # 1. Hurst exponent (simplified R/S method)
    def rolling_hurst(series, w):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(w, len(series)):
            window_data = series.iloc[i - w:i].dropna()
            if len(window_data) < 20:
                result.iloc[i] = 0.5
                continue
            mean_val = window_data.mean()
            deviations = (window_data - mean_val).cumsum()
            R = deviations.max() - deviations.min()
            S = window_data.std()
            if S > 0 and R > 0:
                result.iloc[i] = np.log(R / S) / np.log(len(window_data))
            else:
                result.iloc[i] = 0.5
        return result

    features["hurst"] = rolling_hurst(returns, window)

    # 2. ADX
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = (high - high.shift()).clip(lower=0)
    minus_dm = (low.shift() - low).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    features["adx"] = dx.rolling(14).mean()

    # 3. RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi"] = 100 - (100 / (1 + rs))

    # 4. Volatility ratio (short/long)
    features["vol_ratio"] = returns.rolling(10).std() / (returns.rolling(50).std() + 1e-10)

    # 5. Volume trend
    features["volume_trend"] = df["volume"].rolling(10).mean() / (df["volume"].rolling(50).mean() + 1e-10)

    # 6. Price momentum
    features["momentum"] = df["close"].pct_change(20)

    # 7. ATR percentile
    atr_pct = atr / df["close"]
    features["atr_percentile"] = atr_pct.rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    return features.dropna()


def label_regimes(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Generate regime labels using statistical thresholds."""
    labels = pd.Series(index=features.index, dtype=str)

    for i, idx in enumerate(features.index):
        hurst = features.loc[idx, "hurst"]
        adx = features.loc[idx, "adx"]
        vol_ratio = features.loc[idx, "vol_ratio"]

        if hurst > 0.55 and adx > 25:
            labels.iloc[i] = "TRENDING"
        elif hurst < 0.45 and adx < 20:
            labels.iloc[i] = "RANGING"
        else:
            labels.iloc[i] = "TRANSITION"

    return labels


# ── Training Pipeline ─────────────────────────────────────────────────────────

def train_model(data: Dict[str, pd.DataFrame], threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, dict]:
    """
    Train a new regime model with walk-forward validation.

    Returns:
        (success, metrics_dict)
    """
    try:
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return False, {"error": str(e)}

    # Build training dataset
    all_features = []
    all_labels = []

    for sym, df in data.items():
        features = compute_features(df)
        labels = label_regimes(df, features)

        # Align
        common_idx = features.index.intersection(labels.index)
        all_features.append(features.loc[common_idx])
        all_labels.append(labels.loc[common_idx])

    if not all_features:
        return False, {"error": "No training data"}

    X = pd.concat(all_features)
    y = pd.concat(all_labels)

    log.info(f"Training dataset: {len(X)} samples, {X.shape[1]} features")
    log.info(f"Regime distribution: {y.value_counts().to_dict()}")

    # Walk-forward cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)
        log.info(f"  Fold {fold + 1}: accuracy = {acc:.4f}")

    mean_accuracy = np.mean(fold_scores)
    log.info(f"Mean CV accuracy: {mean_accuracy:.4f} (threshold: {threshold})")

    # Check if model passes threshold
    if mean_accuracy < threshold:
        log.warning(f"Model accuracy {mean_accuracy:.4f} below threshold {threshold}, NOT promoting")
        return False, {
            "mean_accuracy": mean_accuracy,
            "fold_scores": fold_scores,
            "regime_distribution": y.value_counts().to_dict(),
            "promoted": False,
            "reason": "below_threshold",
        }

    # Train final model on all data
    final_model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X, y)

    # Feature importance
    importance = dict(zip(X.columns, final_model.feature_importances_.tolist()))

    # Save model with versioning
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"regime_gbm_{timestamp}.joblib"
    meta_path = MODEL_DIR / f"regime_gbm_{timestamp}.json"

    joblib.dump(final_model, model_path)

    metadata = {
        "version": timestamp,
        "trained_at": datetime.utcnow().isoformat(),
        "symbols": list(data.keys()),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "feature_importance": importance,
        "mean_accuracy": mean_accuracy,
        "fold_scores": fold_scores,
        "regime_distribution": y.value_counts().to_dict(),
        "threshold": threshold,
        "promoted": True,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Update symlink to latest
    latest_model = MODEL_DIR / "regime_gbm_latest.joblib"
    latest_meta = MODEL_DIR / "regime_gbm_latest.json"
    if latest_model.exists():
        latest_model.unlink()
    if latest_meta.exists():
        latest_meta.unlink()
    latest_model.symlink_to(model_path.name)
    latest_meta.symlink_to(meta_path.name)

    log.info(f"Model saved: {model_path}")
    log.info(f"Symlink updated: regime_gbm_latest.joblib → {model_path.name}")

    # Cleanup old versions (keep last MAX_VERSIONS)
    _cleanup_old_models()

    return True, metadata


def _cleanup_old_models():
    """Keep only the last MAX_VERSIONS model files."""
    model_files = sorted(MODEL_DIR.glob("regime_gbm_2*.joblib"), reverse=True)
    meta_files = sorted(MODEL_DIR.glob("regime_gbm_2*.json"), reverse=True)

    for f in model_files[MAX_VERSIONS:]:
        log.info(f"Removing old model: {f.name}")
        f.unlink()

    for f in meta_files[MAX_VERSIONS:]:
        f.unlink()


# ── Drift Detection ──────────────────────────────────────────────────────────

def detect_drift(new_metadata: dict) -> dict:
    """Compare new model metrics against previous version to detect drift."""
    latest_meta = MODEL_DIR / "regime_gbm_v1.json"  # Original v1 model
    if not latest_meta.exists():
        return {"drift_detected": False, "reason": "no_previous_model"}

    with open(latest_meta) as f:
        prev = json.load(f)

    drift_report = {
        "accuracy_delta": new_metadata["mean_accuracy"] - prev.get("accuracy", prev.get("mean_accuracy", 0)),
        "regime_shift": {},
        "importance_drift": {},
    }

    # Regime distribution shift
    prev_dist = prev.get("regime_distribution", {})
    new_dist = new_metadata.get("regime_distribution", {})
    for regime in ["TRENDING", "RANGING", "TRANSITION"]:
        prev_pct = prev_dist.get(regime, 0) / max(sum(prev_dist.values()), 1)
        new_pct = new_dist.get(regime, 0) / max(sum(new_dist.values()), 1)
        drift_report["regime_shift"][regime] = round(new_pct - prev_pct, 4)

    # Feature importance drift
    prev_imp = prev.get("feature_importance", {})
    new_imp = new_metadata.get("feature_importance", {})
    for feat in new_imp:
        if feat in prev_imp:
            drift_report["importance_drift"][feat] = round(
                new_imp[feat] - prev_imp[feat], 4
            )

    # Flag significant drift
    max_regime_shift = max(abs(v) for v in drift_report["regime_shift"].values()) if drift_report["regime_shift"] else 0
    drift_report["drift_detected"] = max_regime_shift > 0.15 or abs(drift_report["accuracy_delta"]) > 0.05

    return drift_report


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML Rolling Retrain Pipeline")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                       help="Comma-separated list of symbols")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help="Minimum accuracy threshold for model promotion")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS,
                       help="Lookback period in days")
    parser.add_argument("--skip-fetch", action="store_true",
                       help="Skip data fetching, use existing parquet files")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    log.info(f"=== ML Rolling Retrain Pipeline ===")
    log.info(f"Symbols: {symbols}")
    log.info(f"Threshold: {args.threshold}")
    log.info(f"Lookback: {args.days} days")

    # Fetch data
    if args.skip_fetch:
        log.info("Skipping fetch, loading from parquet files...")
        data = {}
        data_dir = Path("data")
        for sym in symbols:
            for pattern in [f"{sym}_USD_1h.parquet", f"historical/{sym}_USD_1h.parquet"]:
                path = data_dir / pattern
                if path.exists():
                    data[sym] = pd.read_parquet(path)
                    # Use last N days
                    if "timestamp" in data[sym].columns:
                        data[sym]["timestamp"] = pd.to_datetime(data[sym]["timestamp"])
                        data[sym].set_index("timestamp", inplace=True)
                    data[sym] = data[sym].tail(args.days * 24)
                    log.info(f"  Loaded {sym}: {len(data[sym])} rows")
                    break
    else:
        data = fetch_all_symbols(symbols, args.days)

    if not data:
        log.error("No data available for training")
        sys.exit(1)

    log.info(f"Training on {len(data)} symbols: {list(data.keys())}")

    # Train
    success, metrics = train_model(data, threshold=args.threshold)

    if success:
        log.info("✓ Model promoted successfully!")

        # Drift detection
        drift = detect_drift(metrics)
        if drift["drift_detected"]:
            log.warning(f"⚠ Drift detected: {json.dumps(drift, indent=2)}")
        else:
            log.info("No significant drift detected")

        # Save retrain report
        report_path = MODEL_DIR / f"retrain_report_{metrics['version']}.json"
        report = {**metrics, "drift": drift}
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Report saved: {report_path}")
    else:
        log.warning(f"✗ Model not promoted: {metrics.get('reason', 'unknown')}")
        sys.exit(2)


if __name__ == "__main__":
    main()
