"""
HMM Hybrid Regime Layer
=========================
Implements a 3-state Gaussian Hidden Markov Model for regime detection
with temporal transition dynamics, combined with the existing GBM classifier
for point-in-time confidence confirmation.

Architecture (Hybrid):
  1. HMM Layer — Models regime transitions as a Markov chain with:
     - 3 latent states: LOW_VOL (trending calm), HIGH_VOL (trending volatile), RANGING
     - Gaussian emissions on log-returns + realised volatility
     - Filtered probabilities for smooth risk adjustment
     - Regime persistence modelling (expected duration per state)
  
  2. GBM Confirmation — Uses existing ml_regime_detector for:
     - Point-in-time feature-rich classification
     - Confidence calibration
     - Fallback when HMM has insufficient data

  3. Hybrid Output — Combines both models:
     - HMM provides transition dynamics and filtered probabilities
     - GBM provides feature-rich confirmation
     - Disagreement → reduce confidence (regime transition likely)

Key Advantages over GBM-only:
  - Temporal dynamics: models HOW regimes transition (not just current state)
  - Smooth probabilities: avoids whipsaw from bar-to-bar classification
  - Duration modelling: expected time remaining in current regime
  - Path-dependent: considers sequence of observations, not just latest bar

References:
  - Preprints.org (2026): HMM for Bitcoin regime detection (BIC 11,910 for 3-state)
  - Bucci et al. (2021): Realized covariance + VLSTAR
  - Cube Exchange (2026): Market Regime Detection with HMMs
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed — HMM regime detection unavailable")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HMMConfig:
    """Configuration for the HMM regime detector."""
    n_states: int = 3                    # Number of hidden states
    n_features: int = 3                  # Observation dimensions
    covariance_type: str = "full"        # full/diag/spherical/tied
    n_iter: int = 100                    # EM iterations for fitting
    min_train_bars: int = 500            # Minimum bars to fit HMM
    vol_window: int = 20                 # Realised volatility window
    vol_long_window: int = 60            # Long-term vol for ratio
    retrain_interval_bars: int = 500     # Re-fit HMM every N new bars
    
    # State labels (assigned post-fit based on emission means)
    state_names: List[str] = field(default_factory=lambda: [
        "LOW_VOL", "HIGH_VOL", "RANGING"
    ])


@dataclass
class HMMRegimeState:
    """Current regime state from HMM."""
    regime: str                          # LOW_VOL / HIGH_VOL / RANGING
    regime_id: int                       # 0, 1, 2
    confidence: float                    # Probability of current state
    filtered_probabilities: Dict[str, float]  # All state probabilities
    transition_matrix: List[List[float]]  # Current transition probabilities
    expected_duration: float             # Expected bars remaining in state
    persistence: float                   # Self-transition probability
    method: str                          # 'hmm' or 'hmm_gbm_hybrid'


@dataclass
class HybridRegimeResult:
    """Combined HMM + GBM regime classification."""
    # Primary output
    regime: str                          # Final consensus regime
    confidence: float                    # Hybrid confidence (0-1)
    
    # HMM layer
    hmm_regime: str
    hmm_confidence: float
    hmm_probabilities: Dict[str, float]
    hmm_expected_duration: float
    hmm_persistence: float
    
    # GBM layer
    gbm_regime: str
    gbm_confidence: float
    gbm_probabilities: Dict[str, float]
    
    # Hybrid metadata
    agreement: bool                      # Whether HMM and GBM agree
    transition_risk: float               # 0-1: likelihood of regime change
    method: str = "hmm_gbm_hybrid"
    
    def to_dict(self) -> Dict:
        return {
            'regime': self.regime,
            'confidence': self.confidence,
            'hmm_regime': self.hmm_regime,
            'hmm_confidence': self.hmm_confidence,
            'hmm_probabilities': self.hmm_probabilities,
            'hmm_expected_duration': self.hmm_expected_duration,
            'hmm_persistence': self.hmm_persistence,
            'gbm_regime': self.gbm_regime,
            'gbm_confidence': self.gbm_confidence,
            'gbm_probabilities': self.gbm_probabilities,
            'agreement': self.agreement,
            'transition_risk': self.transition_risk,
            'method': self.method,
        }


# ---------------------------------------------------------------------------
# HMM Regime Detector
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """
    3-state Gaussian HMM for market regime detection.
    
    Fits a Hidden Markov Model on observation features derived from OHLCV:
      - Log returns
      - Realised volatility (rolling std of returns)
      - Volatility ratio (short/long)
    
    The HMM learns:
      - Emission distributions (what each regime "looks like")
      - Transition matrix (how regimes switch)
      - Initial state distribution
    
    Usage:
        detector = HMMRegimeDetector()
        detector.fit(ohlcv_df)  # Fit on historical data
        state = detector.predict(ohlcv_df)  # Classify current regime
    """
    
    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self._model: Optional[GaussianHMM] = None
        self._state_mapping: Dict[int, str] = {}  # Maps HMM state → label
        self._fitted = False
        self._bars_since_fit = 0
        self._fit_log_likelihood = 0.0
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted and self._model is not None
    
    def fit(self, df: pd.DataFrame) -> bool:
        """
        Fit the HMM on historical OHLCV data.
        
        Args:
            df: DataFrame with columns [close, high, low, volume] (minimum)
        
        Returns:
            True if fitting succeeded, False otherwise
        """
        if not HMM_AVAILABLE:
            logger.error("hmmlearn not available — cannot fit HMM")
            return False
        
        if len(df) < self.config.min_train_bars:
            logger.warning(f"Insufficient data ({len(df)} bars, need {self.config.min_train_bars})")
            return False
        
        try:
            # Compute observation features
            X = self._compute_observations(df)
            if X is None or len(X) < self.config.min_train_bars:
                return False
            
            # Fit Gaussian HMM
            model = GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=42,
            )
            model.fit(X)
            
            self._model = model
            self._fit_log_likelihood = model.score(X)
            
            # Assign state labels based on emission means
            self._assign_state_labels()
            
            self._fitted = True
            self._bars_since_fit = 0
            
            logger.info(
                f"HMM fitted: {len(X)} observations, "
                f"log-likelihood={self._fit_log_likelihood:.2f}, "
                f"states={self._state_mapping}"
            )
            return True
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Optional[HMMRegimeState]:
        """
        Predict current regime using the fitted HMM.
        
        Uses forward algorithm to compute filtered probabilities
        (P(state_t | observations_1:t)) for smooth regime estimation.
        
        Args:
            df: Recent OHLCV data (at least vol_long_window + 1 bars)
        
        Returns:
            HMMRegimeState or None if prediction fails
        """
        if not self.is_fitted:
            logger.warning("HMM not fitted — cannot predict")
            return None
        
        try:
            X = self._compute_observations(df)
            if X is None or len(X) < 10:
                return None
            
            # Get filtered probabilities using forward algorithm
            # predict_proba gives P(state | all observations)
            proba = self._model.predict_proba(X)
            
            # Current state = last observation's probabilities
            current_proba = proba[-1]
            state_id = int(np.argmax(current_proba))
            confidence = float(current_proba[state_id])
            
            # Map to regime label
            regime = self._state_mapping.get(state_id, "UNKNOWN")
            
            # Build probability dict
            filtered_probs = {}
            for sid, label in self._state_mapping.items():
                filtered_probs[label] = round(float(current_proba[sid]), 4)
            
            # Get transition matrix
            trans_matrix = self._model.transmat_.tolist()
            
            # Expected duration = 1 / (1 - self-transition probability)
            self_trans = trans_matrix[state_id][state_id]
            expected_duration = 1.0 / (1.0 - self_trans) if self_trans < 1.0 else float('inf')
            
            self._bars_since_fit += 1
            
            return HMMRegimeState(
                regime=regime,
                regime_id=state_id,
                confidence=round(confidence, 4),
                filtered_probabilities=filtered_probs,
                transition_matrix=[[round(p, 4) for p in row] for row in trans_matrix],
                expected_duration=round(expected_duration, 1),
                persistence=round(self_trans, 4),
                method='hmm',
            )
            
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return None
    
    def needs_refit(self) -> bool:
        """Check if the model should be re-fitted."""
        return self._bars_since_fit >= self.config.retrain_interval_bars
    
    def _compute_observations(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Compute observation features for HMM:
          1. Log returns
          2. Realised volatility (rolling std of returns)
          3. Volatility ratio (short/long)
        """
        close = df['close'].values.astype(float)
        n = len(close)
        
        if n < self.config.vol_long_window + 2:
            return None
        
        # 1. Log returns
        log_returns = np.diff(np.log(close + 1e-10))
        
        # 2. Realised volatility (rolling std)
        vol_window = self.config.vol_window
        realised_vol = np.zeros(len(log_returns))
        for i in range(vol_window, len(log_returns)):
            realised_vol[i] = np.std(log_returns[i - vol_window:i])
        
        # 3. Volatility ratio (short/long)
        long_window = self.config.vol_long_window
        vol_ratio = np.ones(len(log_returns))
        for i in range(long_window, len(log_returns)):
            short_vol = np.std(log_returns[i - vol_window:i])
            long_vol = np.std(log_returns[i - long_window:i])
            if long_vol > 1e-10:
                vol_ratio[i] = short_vol / long_vol
        
        # Trim to valid range (after long_window)
        start = long_window
        X = np.column_stack([
            log_returns[start:],
            realised_vol[start:],
            vol_ratio[start:],
        ])
        
        # Remove any NaN/inf
        mask = np.all(np.isfinite(X), axis=1)
        X = X[mask]
        
        return X if len(X) > 0 else None
    
    def _assign_state_labels(self):
        """
        Assign semantic labels to HMM states based on emission means.
        
        Strategy:
          - State with lowest volatility mean → LOW_VOL
          - State with highest volatility mean → HIGH_VOL
          - Remaining state → RANGING
        """
        if self._model is None:
            return
        
        means = self._model.means_  # Shape: (n_states, n_features)
        # Feature index 1 = realised volatility
        vol_means = means[:, 1]
        
        sorted_indices = np.argsort(vol_means)
        
        self._state_mapping = {
            int(sorted_indices[0]): "LOW_VOL",      # Lowest vol
            int(sorted_indices[-1]): "HIGH_VOL",    # Highest vol
            int(sorted_indices[1]): "RANGING",      # Middle vol
        }


# ---------------------------------------------------------------------------
# Hybrid Regime Classifier
# ---------------------------------------------------------------------------

class HybridRegimeClassifier:
    """
    Combines HMM temporal dynamics with GBM point-in-time classification.
    
    The hybrid approach:
    1. HMM provides: transition probabilities, filtered state, expected duration
    2. GBM provides: feature-rich confidence, point-in-time classification
    3. Hybrid logic:
       - If both agree → high confidence in consensus regime
       - If they disagree → likely in transition, reduce confidence
       - HMM persistence informs position sizing (longer expected duration → larger size)
    
    Usage:
        classifier = HybridRegimeClassifier()
        classifier.fit(historical_ohlcv)
        result = classifier.classify(recent_ohlcv)
        print(f"Regime: {result.regime}, Confidence: {result.confidence}")
        print(f"Transition risk: {result.transition_risk}")
    """
    
    def __init__(self, hmm_config: Optional[HMMConfig] = None):
        self.hmm = HMMRegimeDetector(config=hmm_config)
        self._regime_label_map = {
            # Map GBM labels to HMM labels for comparison
            'TRENDING': 'LOW_VOL',      # Trending ≈ low-vol directional
            'RANGING': 'RANGING',
            'TRANSITION': 'HIGH_VOL',   # Transition ≈ high-vol uncertainty
        }
        self._hmm_to_unified = {
            'LOW_VOL': 'TRENDING',
            'HIGH_VOL': 'TRANSITION',
            'RANGING': 'RANGING',
        }
    
    def fit(self, df: pd.DataFrame) -> bool:
        """Fit the HMM on historical data."""
        return self.hmm.fit(df)
    
    def classify(self, df: pd.DataFrame, gbm_result: Optional[Dict] = None) -> HybridRegimeResult:
        """
        Classify regime using hybrid HMM + GBM approach.
        
        Args:
            df: Recent OHLCV data
            gbm_result: Optional pre-computed GBM result from ml_regime_detector.classify_regime()
                       If None, only HMM is used.
        
        Returns:
            HybridRegimeResult with consensus regime and transition risk
        """
        # Get HMM prediction
        hmm_state = self.hmm.predict(df)
        
        # Get GBM prediction (use provided or compute)
        if gbm_result is None:
            try:
                from .ml_regime_detector import classify_regime
                gbm_result = classify_regime(df)
            except Exception as e:
                logger.warning(f"GBM classification failed: {e}")
                gbm_result = None
        
        # If HMM failed, fall back to GBM only
        if hmm_state is None:
            if gbm_result:
                return HybridRegimeResult(
                    regime=gbm_result['regime'],
                    confidence=gbm_result['confidence'],
                    hmm_regime="UNKNOWN",
                    hmm_confidence=0.0,
                    hmm_probabilities={},
                    hmm_expected_duration=0.0,
                    hmm_persistence=0.0,
                    gbm_regime=gbm_result['regime'],
                    gbm_confidence=gbm_result['confidence'],
                    gbm_probabilities=gbm_result.get('probabilities', {}),
                    agreement=True,
                    transition_risk=0.5,
                    method='gbm_only',
                )
            else:
                return HybridRegimeResult(
                    regime="RANGING",
                    confidence=0.3,
                    hmm_regime="UNKNOWN",
                    hmm_confidence=0.0,
                    hmm_probabilities={},
                    hmm_expected_duration=0.0,
                    hmm_persistence=0.0,
                    gbm_regime="UNKNOWN",
                    gbm_confidence=0.0,
                    gbm_probabilities={},
                    agreement=False,
                    transition_risk=0.8,
                    method='fallback',
                )
        
        # Map HMM regime to unified labels
        hmm_unified = self._hmm_to_unified.get(hmm_state.regime, hmm_state.regime)
        
        # Get GBM info
        gbm_regime = gbm_result['regime'] if gbm_result else "UNKNOWN"
        gbm_confidence = gbm_result['confidence'] if gbm_result else 0.0
        gbm_probabilities = gbm_result.get('probabilities', {}) if gbm_result else {}
        
        # Determine agreement
        agreement = (hmm_unified == gbm_regime)
        
        # Compute hybrid confidence
        if agreement:
            # Both agree → boost confidence
            hybrid_confidence = min(1.0, (hmm_state.confidence + gbm_confidence) / 2 * 1.2)
            final_regime = gbm_regime  # Use GBM label (more granular)
        else:
            # Disagreement → likely transition, reduce confidence
            hybrid_confidence = min(hmm_state.confidence, gbm_confidence) * 0.7
            # Prefer GBM if its confidence is much higher, else use HMM
            if gbm_confidence > hmm_state.confidence + 0.2:
                final_regime = gbm_regime
            else:
                final_regime = hmm_unified
        
        # Transition risk: based on HMM persistence and agreement
        # Low persistence + disagreement = high transition risk
        transition_risk = 1.0 - hmm_state.persistence
        if not agreement:
            transition_risk = min(1.0, transition_risk * 1.5)
        
        return HybridRegimeResult(
            regime=final_regime,
            confidence=round(hybrid_confidence, 4),
            hmm_regime=hmm_state.regime,
            hmm_confidence=hmm_state.confidence,
            hmm_probabilities=hmm_state.filtered_probabilities,
            hmm_expected_duration=hmm_state.expected_duration,
            hmm_persistence=hmm_state.persistence,
            gbm_regime=gbm_regime,
            gbm_confidence=gbm_confidence,
            gbm_probabilities=gbm_probabilities,
            agreement=agreement,
            transition_risk=round(transition_risk, 4),
            method='hmm_gbm_hybrid',
        )
