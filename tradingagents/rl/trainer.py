"""RL Trainer — Walk-forward training pipeline for the sizing agent.

Implements proper temporal train/validation splits to prevent look-ahead bias.
Trains the RL agent on rolling windows and evaluates on out-of-sample periods.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .agent import RLSizingAgent
from .env import TradingSizingEnv

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for RL training pipeline."""
    algorithm: str = "PPO"                    # PPO or SAC
    total_timesteps: int = 100_000            # Steps per training window
    train_window_bars: int = 4000             # ~6 months of hourly data
    val_window_bars: int = 1000               # ~6 weeks validation
    step_size_bars: int = 1000                # Roll forward by this many bars
    base_qty: float = 1.0
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    max_drawdown_penalty: float = 2.0
    turnover_penalty: float = 0.1
    max_position_pct: float = 0.15
    model_dir: str = "data/models/rl"
    min_sharpe_threshold: float = 0.5         # Minimum Sharpe to accept model


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    train_sharpe: float
    val_sharpe: float
    val_max_dd: float
    val_return: float
    avg_multiplier: float
    trade_count: int
    wfe: float  # Walk-Forward Efficiency (val_sharpe / train_sharpe)
    model_path: Optional[str] = None


@dataclass
class TrainResult:
    """Aggregate result from walk-forward training."""
    folds: List[FoldResult] = field(default_factory=list)
    best_fold_idx: int = 0
    best_model_path: str = ""
    avg_val_sharpe: float = 0.0
    avg_wfe: float = 0.0
    total_training_time: float = 0.0
    
    @property
    def passed_gate(self) -> bool:
        """Whether the trained model passes deployment criteria."""
        return self.avg_wfe >= 0.5 and self.avg_val_sharpe > 0.0


class RLTrainer:
    """Walk-forward RL training pipeline.
    
    Trains the RL sizing agent on rolling windows of historical data,
    evaluating on out-of-sample periods to prevent overfitting.
    """
    
    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
    
    def train_walk_forward(
        self,
        signals: list[dict],
        prices: np.ndarray,
        progress_callback=None,
    ) -> TrainResult:
        """Run walk-forward training across multiple folds.
        
        Args:
            signals: Full history of signal dicts (one per bar).
            prices: Aligned close prices array.
            progress_callback: Optional callable(fold_idx, total_folds, fold_result).
            
        Returns:
            TrainResult with all fold results and best model path.
        """
        cfg = self.config
        n_bars = len(signals)
        
        if n_bars < cfg.train_window_bars + cfg.val_window_bars:
            raise ValueError(
                f"Insufficient data: {n_bars} bars, need at least "
                f"{cfg.train_window_bars + cfg.val_window_bars}"
            )
        
        # Calculate fold boundaries
        folds = []
        start = 0
        while start + cfg.train_window_bars + cfg.val_window_bars <= n_bars:
            train_start = start
            train_end = start + cfg.train_window_bars
            val_start = train_end
            val_end = min(train_end + cfg.val_window_bars, n_bars)
            folds.append((train_start, train_end, val_start, val_end))
            start += cfg.step_size_bars
        
        if not folds:
            raise ValueError("No valid folds could be created with current config")
        
        logger.info(f"Walk-forward training: {len(folds)} folds, "
                    f"train={cfg.train_window_bars} bars, val={cfg.val_window_bars} bars")
        
        results = []
        best_val_sharpe = -np.inf
        best_model_path = ""
        t0 = time.time()
        
        for fold_idx, (ts, te, vs, ve) in enumerate(folds):
            logger.info(f"Fold {fold_idx + 1}/{len(folds)}: "
                        f"train[{ts}:{te}] val[{vs}:{ve}]")
            
            # Create training environment
            train_env = TradingSizingEnv(
                signals=signals[ts:te],
                prices=prices[ts:te],
                base_qty=cfg.base_qty,
                initial_capital=cfg.initial_capital,
                commission_pct=cfg.commission_pct,
                slippage_pct=cfg.slippage_pct,
                max_drawdown_penalty=cfg.max_drawdown_penalty,
                turnover_penalty=cfg.turnover_penalty,
                max_position_pct=cfg.max_position_pct,
            )
            
            # Build and train agent
            agent = RLSizingAgent(algorithm=cfg.algorithm)
            agent.build(train_env)
            agent.train(total_timesteps=cfg.total_timesteps)
            
            # Evaluate on training set (for WFE calculation)
            train_stats = self._evaluate(agent, signals[ts:te], prices[ts:te], cfg)
            
            # Evaluate on validation set
            val_stats = self._evaluate(agent, signals[vs:ve], prices[vs:ve], cfg)
            
            # Calculate WFE
            wfe = val_stats["sharpe"] / train_stats["sharpe"] if train_stats["sharpe"] > 0 else 0.0
            
            # Save model
            model_path = str(Path(cfg.model_dir) / f"rl_sizing_fold{fold_idx}.zip")
            agent.save(model_path)
            
            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start=ts,
                train_end=te,
                val_start=vs,
                val_end=ve,
                train_sharpe=train_stats["sharpe"],
                val_sharpe=val_stats["sharpe"],
                val_max_dd=val_stats["max_drawdown"],
                val_return=val_stats["total_return"],
                avg_multiplier=val_stats["avg_multiplier"],
                trade_count=val_stats["trade_count"],
                wfe=wfe,
                model_path=model_path,
            )
            results.append(fold_result)
            
            if val_stats["sharpe"] > best_val_sharpe:
                best_val_sharpe = val_stats["sharpe"]
                best_model_path = model_path
                best_fold_idx = fold_idx
            
            if progress_callback:
                progress_callback(fold_idx, len(folds), fold_result)
            
            logger.info(f"  Train Sharpe: {train_stats['sharpe']:.3f}, "
                        f"Val Sharpe: {val_stats['sharpe']:.3f}, WFE: {wfe:.2f}")
        
        total_time = time.time() - t0
        
        # Copy best model to canonical path
        best_canonical = str(Path(cfg.model_dir) / "rl_sizing_best.zip")
        if best_model_path:
            import shutil
            Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_model_path, best_canonical)
        
        train_result = TrainResult(
            folds=results,
            best_fold_idx=best_fold_idx if results else 0,
            best_model_path=best_canonical,
            avg_val_sharpe=float(np.mean([r.val_sharpe for r in results])),
            avg_wfe=float(np.mean([r.wfe for r in results])),
            total_training_time=total_time,
        )
        
        logger.info(f"Training complete in {total_time:.1f}s. "
                    f"Avg Val Sharpe: {train_result.avg_val_sharpe:.3f}, "
                    f"Avg WFE: {train_result.avg_wfe:.2f}, "
                    f"Gate: {'PASS' if train_result.passed_gate else 'FAIL'}")
        
        return train_result
    
    def _evaluate(
        self,
        agent: RLSizingAgent,
        signals: list[dict],
        prices: np.ndarray,
        cfg: TrainConfig,
    ) -> dict:
        """Evaluate agent on a data segment."""
        env = TradingSizingEnv(
            signals=signals,
            prices=prices,
            base_qty=cfg.base_qty,
            initial_capital=cfg.initial_capital,
            commission_pct=cfg.commission_pct,
            slippage_pct=cfg.slippage_pct,
            max_drawdown_penalty=cfg.max_drawdown_penalty,
            turnover_penalty=cfg.turnover_penalty,
            max_position_pct=cfg.max_position_pct,
        )
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            multiplier = agent.predict(obs, deterministic=True)
            action = np.array([multiplier], dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        return env.episode_stats
