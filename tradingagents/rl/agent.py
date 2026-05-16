"""RL Sizing Agent — wraps Stable-Baselines3 PPO/SAC for position sizing.

Provides a clean interface for training, inference, and model persistence.
Supports both PPO (on-policy, stable) and SAC (off-policy, sample-efficient).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports to avoid heavy torch load at module level
_SB3_AVAILABLE = None


def _check_sb3():
    global _SB3_AVAILABLE
    if _SB3_AVAILABLE is None:
        try:
            from stable_baselines3 import PPO, SAC  # noqa: F401
            _SB3_AVAILABLE = True
        except ImportError:
            _SB3_AVAILABLE = False
    return _SB3_AVAILABLE


class RLSizingAgent:
    """RL agent for dynamic position sizing.
    
    Wraps Stable-Baselines3 PPO or SAC with sensible defaults for
    the trading sizing problem.
    """
    
    DEFAULT_PPO_KWARGS = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 0,
    }
    
    DEFAULT_SAC_KWARGS = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "verbose": 0,
    }
    
    def __init__(
        self,
        algorithm: Literal["PPO", "SAC"] = "PPO",
        model_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            algorithm: Which SB3 algorithm to use.
            model_path: Path to a pre-trained model to load.
            **kwargs: Override default hyperparameters.
        """
        if not _check_sb3():
            raise ImportError("stable-baselines3 is required: pip install stable-baselines3")
        
        self.algorithm = algorithm
        self.model_path = model_path
        self.model = None
        self._custom_kwargs = kwargs
        self._is_trained = False
    
    def build(self, env) -> "RLSizingAgent":
        """Build or load the model for the given environment."""
        from stable_baselines3 import PPO, SAC
        
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading pre-trained {self.algorithm} model from {self.model_path}")
            cls = PPO if self.algorithm == "PPO" else SAC
            self.model = cls.load(self.model_path, env=env)
            self._is_trained = True
        else:
            logger.info(f"Building new {self.algorithm} model")
            if self.algorithm == "PPO":
                params = {**self.DEFAULT_PPO_KWARGS, **self._custom_kwargs}
                self.model = PPO("MlpPolicy", env, **params)
            else:
                params = {**self.DEFAULT_SAC_KWARGS, **self._custom_kwargs}
                self.model = SAC("MlpPolicy", env, **params)
        
        return self
    
    def train(self, total_timesteps: int = 100_000, progress_bar: bool = False) -> dict:
        """Train the agent.
        
        Returns:
            Training summary with final metrics.
        """
        if self.model is None:
            raise RuntimeError("Call build(env) before train()")
        
        logger.info(f"Training {self.algorithm} for {total_timesteps:,} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        self._is_trained = True
        
        return {"algorithm": self.algorithm, "timesteps": total_timesteps, "status": "trained"}
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> float:
        """Predict sizing multiplier for a single observation.
        
        Args:
            observation: State vector (12 features).
            deterministic: If True, use mean action (no exploration noise).
            
        Returns:
            Position size multiplier [0, 2].
        """
        if self.model is None:
            raise RuntimeError("Model not built or loaded")
        
        if not self._is_trained:
            # Return neutral sizing if not trained
            return 1.0
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return float(np.clip(action[0], 0.0, 2.0))
    
    def save(self, path: str) -> str:
        """Save model to disk.
        
        Returns:
            Actual save path.
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        return path
    
    def load(self, path: str, env=None) -> "RLSizingAgent":
        """Load a pre-trained model."""
        from stable_baselines3 import PPO, SAC
        
        cls = PPO if self.algorithm == "PPO" else SAC
        self.model = cls.load(path, env=env)
        self._is_trained = True
        self.model_path = path
        logger.info(f"Model loaded from {path}")
        return self
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    def get_sizing_multiplier(
        self,
        signal: dict,
        portfolio_state: dict,
        deterministic: bool = True,
    ) -> float:
        """High-level interface for live trading integration.
        
        Constructs observation from signal + portfolio state and returns
        the sizing multiplier.
        
        Args:
            signal: Output from PerAssetRouter.generate_signals()
            portfolio_state: Dict with portfolio_heat, drawdown, correlation_avg, etc.
            deterministic: Use mean action (True for live trading).
            
        Returns:
            Sizing multiplier [0, 2].
        """
        if not self._is_trained:
            return 1.0  # neutral fallback
        
        obs = np.array([
            signal.get("regime_trending_prob", 0.5),
            signal.get("regime_ranging_prob", 0.3),
            signal.get("volatility_percentile", 0.5),
            min(signal.get("atr_ratio", 1.0), 3.0),
            signal.get("conviction", 0.5),
            signal.get("rsi", 50.0) / 100.0,
            signal.get("adx", 25.0) / 100.0,
            min(portfolio_state.get("portfolio_heat", 0.0), 1.0),
            min(portfolio_state.get("drawdown", 0.0), 1.0),
            np.clip(portfolio_state.get("correlation_avg", 0.0), -1.0, 1.0),
            np.clip(signal.get("sentiment_score", 0.0), -1.0, 1.0),
            min(portfolio_state.get("bars_since_last_trade", 0.0), 1.0),
        ], dtype=np.float32)
        
        return self.predict(obs, deterministic=deterministic)
