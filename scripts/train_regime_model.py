"""
Train ML Regime Detection Model — LightGBM
============================================
Trains a gradient-boosted classifier on historical OHLCV data with
engineered features to predict market regime (TRENDING/RANGING/TRANSITION).

Usage:
    python3 scripts/train_regime_model.py [--symbols BTCUSDT ETHUSDT ...] [--output path]

Label generation:
    - TRENDING: ADX > 25 AND Hurst > 0.52 (sustained for 3+ bars)
    - RANGING:  ADX < 20 AND Hurst < 0.45 (sustained for 3+ bars)
    - TRANSITION: everything else

This creates training labels from the statistical classifier, then trains
a GBM to learn the non-linear decision boundary with better generalisation.
The model can detect regime transitions earlier than the threshold-based approach.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tradingagents.research.ml_regime_detector import compute_features, _hurst, _adx_full

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

MODEL_DIR = ROOT / 'data' / 'models'
DATA_DIR = ROOT / 'data'


def generate_labels(df: pd.DataFrame, lookback: int = 3) -> np.ndarray:
    """
    Generate regime labels from price data using sustained threshold approach.
    
    Labels:
        0 = TRENDING (ADX > 25, Hurst > 0.52, sustained 3+ bars)
        1 = RANGING  (ADX < 20, Hurst < 0.45, sustained 3+ bars)
        2 = TRANSITION (everything else)
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    n = len(close)
    
    labels = np.full(n, 2)  # Default: TRANSITION
    
    # Compute rolling indicators
    from tradingagents.research.per_asset_router import _adx as _adx_array, _hurst_fast
    
    adx_arr = _adx_array(high, low, close)[0]  # Returns (adx, di_p, di_m)
    hurst_arr = _hurst_fast(close, 96)
    
    # Assign labels with sustained confirmation
    for i in range(lookback, n):
        trending_count = sum(
            1 for j in range(i - lookback + 1, i + 1)
            if adx_arr[j] > 25 and hurst_arr[j] > 0.52
        )
        ranging_count = sum(
            1 for j in range(i - lookback + 1, i + 1)
            if adx_arr[j] < 20 and hurst_arr[j] < 0.45
        )
        
        if trending_count >= lookback:
            labels[i] = 0  # TRENDING
        elif ranging_count >= lookback:
            labels[i] = 1  # RANGING
        # else: remains TRANSITION (2)
    
    return labels


def compute_features_batch(df: pd.DataFrame, min_bars: int = 100) -> np.ndarray:
    """Compute features for all bars in the DataFrame (vectorised where possible)."""
    n = len(df)
    features = np.zeros((n, 8))
    
    for i in range(min_bars, n):
        window_df = df.iloc[max(0, i - 200):i + 1]
        features[i] = compute_features(window_df)
    
    return features


def load_data(symbols: list) -> dict:
    """Load parquet data files for given symbols."""
    from tradingagents.research.per_asset_router import DATA_FILE_MAP
    
    datasets = {}
    for sym in symbols:
        if sym not in DATA_FILE_MAP:
            log.warning(f"No data file mapping for {sym}, skipping")
            continue
        
        prefix, timeframe = DATA_FILE_MAP[sym]
        # Try common file patterns
        candidates = [
            DATA_DIR / f'{prefix}_{timeframe}.parquet',
            DATA_DIR / f'{prefix}.parquet',
            DATA_DIR / 'parquet' / f'{prefix}_{timeframe}.parquet',
            DATA_DIR / 'parquet' / f'{prefix}.parquet',
        ]
        
        found = None
        for path in candidates:
            if path.exists():
                found = path
                break
        
        if found:
            df = pd.read_parquet(found)
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            if 'close' in df.columns:
                datasets[sym] = df
                log.info(f"Loaded {sym}: {len(df)} bars from {found}")
            else:
                log.warning(f"No 'close' column in {found}")
        else:
            log.warning(f"No data file found for {sym}")
    
    return datasets


def train_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """Train LightGBM classifier with cross-validation."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("LightGBM not installed. Run: pip install lightgbm")
        sys.exit(1)
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    log.info(f"Training set: {len(X_train)} samples")
    log.info(f"Test set: {len(X_test)} samples")
    log.info(f"Class distribution (train): {np.bincount(y_train.astype(int))}")
    
    # LightGBM parameters optimised for regime detection
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    log.info(f"\nTest Accuracy: {accuracy:.4f}")
    log.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['TRENDING', 'RANGING', 'TRANSITION'])}")
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = ['hurst', 'adx', 'rsi', 'vol_ratio', 'vol_trend', 'momentum', 'atr_pct', 'di_balance']
    log.info("\nFeature Importance (gain):")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        log.info(f"  {name}: {imp:.1f}")
    
    return model, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train ML Regime Detection Model')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'BNBUSDT', 'XRPUSDT', 'LINKUSDT', 'DOGEUSDT'])
    parser.add_argument('--output', type=str, default=str(MODEL_DIR / 'regime_gbm_v1.joblib'))
    parser.add_argument('--min-bars', type=int, default=100)
    args = parser.parse_args()
    
    log.info(f"Training regime model for symbols: {args.symbols}")
    
    # Load data
    datasets = load_data(args.symbols)
    if not datasets:
        log.error("No data loaded. Ensure parquet files exist in data/ directory.")
        sys.exit(1)
    
    # Generate features and labels for all symbols
    all_X = []
    all_y = []
    
    for sym, df in datasets.items():
        log.info(f"Processing {sym} ({len(df)} bars)...")
        
        # Generate labels
        labels = generate_labels(df)
        
        # Compute features
        features = compute_features_batch(df, args.min_bars)
        
        # Only use bars with enough history
        valid_mask = np.arange(len(df)) >= args.min_bars
        X = features[valid_mask]
        y = labels[valid_mask]
        
        all_X.append(X)
        all_y.append(y)
        
        log.info(f"  {sym}: {len(X)} valid samples, class dist: {np.bincount(y.astype(int), minlength=3)}")
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    log.info(f"\nTotal training samples: {len(X)}")
    log.info(f"Total class distribution: TRENDING={np.sum(y==0)}, RANGING={np.sum(y==1)}, TRANSITION={np.sum(y==2)}")
    
    # Train
    model, accuracy = train_model(X, y)
    
    # Save model
    import joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    joblib.dump(model, output_path)
    log.info(f"\nModel saved to: {output_path}")
    log.info(f"Model accuracy: {accuracy:.4f}")
    
    # Save metadata
    meta = {
        'version': 'regime_gbm_v1',
        'accuracy': round(accuracy, 4),
        'symbols_trained': list(datasets.keys()),
        'total_samples': int(len(X)),
        'feature_count': 8,
        'feature_names': ['hurst', 'adx', 'rsi', 'vol_ratio', 'vol_trend', 'momentum', 'atr_pct', 'di_balance'],
        'classes': ['TRENDING', 'RANGING', 'TRANSITION'],
    }
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved to: {meta_path}")


if __name__ == '__main__':
    main()
