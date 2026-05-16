"""RL Environment for Position Sizing.

Wraps the existing backtest engine as a Gymnasium environment.
The agent observes market state features and outputs a position size
multiplier. Reward is risk-adjusted PnL with drawdown and turnover penalties.

State Space (12 features):
  - regime_trending_prob: HMM/GBM probability of trending regime [0, 1]
  - regime_ranging_prob: probability of ranging regime [0, 1]
  - volatility_percentile: current vol vs 90-day distribution [0, 1]
  - atr_ratio: current ATR / 20-period ATR mean [0, 3]
  - conviction: signal conviction from quant strategy [0, 1]
  - rsi_normalised: RSI / 100 [0, 1]
  - adx_normalised: ADX / 100 [0, 1]
  - portfolio_heat: current exposure / max exposure [0, 1]
  - drawdown: current drawdown from peak [0, 1]
  - correlation_avg: average pairwise correlation of portfolio [−1, 1]
  - sentiment_score: LLM sentiment [-1, 1]
  - bars_since_last_trade: normalised time since last trade [0, 1]

Action Space:
  - Continuous [0, 2]: position size multiplier
    0.0 = skip trade entirely
    1.0 = base size (unchanged)
    2.0 = double position size
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class TradingSizingEnv(gym.Env):
    """Gymnasium environment for RL-based position sizing.
    
    The environment replays historical data bar-by-bar. At each step where
    the quant strategy generates a signal (LONG/SHORT), the RL agent decides
    the position size multiplier. Steps without signals are fast-forwarded.
    """
    
    metadata = {"render_modes": ["human"]}
    
    # State dimension
    STATE_DIM = 12
    
    def __init__(
        self,
        signals: list[dict],
        prices: np.ndarray,
        base_qty: float = 1.0,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        max_drawdown_penalty: float = 2.0,
        turnover_penalty: float = 0.1,
        max_position_pct: float = 0.15,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            signals: List of signal dicts from PerAssetRouter, one per bar.
                     Each must have: signal, conviction, regime features.
            prices: Array of close prices aligned with signals.
            base_qty: Base position quantity (before RL multiplier).
            initial_capital: Starting capital.
            commission_pct: Commission per trade (fraction).
            slippage_pct: Slippage per trade (fraction).
            max_drawdown_penalty: Penalty coefficient for drawdown.
            turnover_penalty: Penalty coefficient for excessive trading.
            max_position_pct: Max single position as fraction of equity.
        """
        super().__init__()
        
        self.signals = signals
        self.prices = prices
        self.base_qty = base_qty
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_drawdown_penalty = max_drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.max_position_pct = max_position_pct
        self.render_mode = render_mode
        
        # Action: continuous multiplier [0, 2]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        
        # Observation: 12 normalised features
        self.observation_space = spaces.Box(
            low=-1.0,
            high=3.0,
            shape=(self.STATE_DIM,),
            dtype=np.float32,
        )
        
        # Episode state
        self._step_idx = 0
        self._equity = initial_capital
        self._peak_equity = initial_capital
        self._position = 0.0  # current position qty (signed)
        self._entry_price = 0.0
        self._last_trade_bar = 0
        self._total_turnover = 0.0
        self._trade_count = 0
        self._equity_curve: list[float] = []
        self._sizing_history: list[float] = []
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to start of episode."""
        super().reset(seed=seed)
        
        self._step_idx = 0
        self._equity = self.initial_capital
        self._peak_equity = self.initial_capital
        self._position = 0.0
        self._entry_price = 0.0
        self._last_trade_bar = 0
        self._total_turnover = 0.0
        self._trade_count = 0
        self._equity_curve = [self.initial_capital]
        self._sizing_history = []
        
        obs = self._get_observation()
        info = {"equity": self._equity, "step": 0}
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step: apply sizing multiplier and compute reward."""
        multiplier = float(np.clip(action[0], 0.0, 2.0))
        self._sizing_history.append(multiplier)
        
        signal = self.signals[self._step_idx]
        price = self.prices[self._step_idx]
        direction = signal.get("signal", "FLAT")
        conviction = signal.get("conviction", 0.0)
        
        reward = 0.0
        
        # Close existing position if direction changed or FLAT
        if self._position != 0.0:
            if direction == "FLAT" or (direction == "LONG" and self._position < 0) or \
               (direction == "SHORT" and self._position > 0):
                reward += self._close_position(price)
        
        # Open new position if signal is active and multiplier > 0
        if direction in ("LONG", "SHORT") and multiplier > 0.05 and self._position == 0.0:
            # Calculate position size
            sized_qty = self.base_qty * multiplier * conviction
            
            # Enforce max position constraint
            max_qty = (self._equity * self.max_position_pct) / price
            sized_qty = min(sized_qty, max_qty)
            
            if sized_qty > 0.001:
                # Debit the full notional + costs from cash (margin model)
                notional = price * sized_qty
                cost = notional * (self.commission_pct + self.slippage_pct)
                
                if notional + cost > self._equity:
                    # Not enough capital — reduce size
                    sized_qty = (self._equity * 0.95) / (price * (1 + self.commission_pct + self.slippage_pct))
                    notional = price * sized_qty
                    cost = notional * (self.commission_pct + self.slippage_pct)
                
                if direction == "LONG":
                    self._position = sized_qty
                else:
                    self._position = -sized_qty
                self._entry_price = price
                self._equity -= (notional + cost)  # cash goes down by full notional + fees
                self._total_turnover += notional
                self._trade_count += 1
                self._last_trade_bar = self._step_idx
        
        # Mark-to-market
        if self._position != 0.0:
            unrealised = self._position * (price - self._entry_price)
            current_equity = self._equity + unrealised
        else:
            current_equity = self._equity
        
        self._peak_equity = max(self._peak_equity, current_equity)
        self._equity_curve.append(current_equity)
        
        # Compute reward components
        # 1. Risk-adjusted return (step PnL / initial capital)
        step_return = (current_equity - self._equity_curve[-2]) / self.initial_capital if len(self._equity_curve) > 1 else 0
        
        # 2. Drawdown penalty
        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else 0
        dd_penalty = self.max_drawdown_penalty * drawdown ** 2  # quadratic penalty
        
        # 3. Turnover penalty (discourage excessive trading)
        turnover_ratio = self._total_turnover / self.initial_capital
        turn_penalty = self.turnover_penalty * (turnover_ratio / max(self._step_idx + 1, 1))
        
        reward += step_return - dd_penalty - turn_penalty
        
        # Advance step
        self._step_idx += 1
        terminated = self._step_idx >= len(self.signals)
        truncated = current_equity <= self.initial_capital * 0.5  # 50% loss = game over
        
        if truncated and self._position != 0.0:
            self._close_position(price)
        
        obs = self._get_observation() if not terminated else np.zeros(self.STATE_DIM, dtype=np.float32)
        
        info = {
            "equity": current_equity,
            "drawdown": drawdown,
            "trade_count": self._trade_count,
            "multiplier": multiplier,
            "step": self._step_idx,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _close_position(self, price: float) -> float:
        """Close current position and return realised PnL."""
        if self._position == 0.0:
            return 0.0
        
        # Calculate PnL
        gross_pnl = self._position * (price - self._entry_price)
        notional_exit = abs(self._position) * price
        cost = notional_exit * (self.commission_pct + self.slippage_pct)
        net_pnl = gross_pnl - cost
        
        # Return the original notional (margin release) + net PnL
        notional_entry = abs(self._position) * self._entry_price
        self._equity += notional_entry + net_pnl
        self._total_turnover += notional_exit
        self._position = 0.0
        self._entry_price = 0.0
        
        return net_pnl / self.initial_capital  # normalised PnL
    
    def _get_observation(self) -> np.ndarray:
        """Extract normalised state vector from current signal."""
        if self._step_idx >= len(self.signals):
            return np.zeros(self.STATE_DIM, dtype=np.float32)
        
        sig = self.signals[self._step_idx]
        
        # Extract features with safe defaults
        regime_trending = sig.get("regime_trending_prob", 0.5)
        regime_ranging = sig.get("regime_ranging_prob", 0.3)
        vol_pct = sig.get("volatility_percentile", 0.5)
        atr_ratio = sig.get("atr_ratio", 1.0)
        conviction = sig.get("conviction", 0.0)
        rsi = sig.get("rsi", 50.0) / 100.0
        adx = sig.get("adx", 25.0) / 100.0
        
        # Portfolio state
        current_equity = self._equity_curve[-1] if self._equity_curve else self.initial_capital
        portfolio_heat = abs(self._position * self._entry_price) / max(current_equity, 1.0)
        drawdown = (self._peak_equity - current_equity) / max(self._peak_equity, 1.0)
        
        # Correlation and sentiment (from signal metadata)
        corr_avg = sig.get("correlation_avg", 0.0)
        sentiment = sig.get("sentiment_score", 0.0)
        
        # Time since last trade (normalised by episode length)
        bars_since = (self._step_idx - self._last_trade_bar) / max(len(self.signals), 1)
        
        state = np.array([
            regime_trending,
            regime_ranging,
            vol_pct,
            min(atr_ratio, 3.0),
            conviction,
            rsi,
            adx,
            min(portfolio_heat, 1.0),
            min(drawdown, 1.0),
            np.clip(corr_avg, -1.0, 1.0),
            np.clip(sentiment, -1.0, 1.0),
            min(bars_since, 1.0),
        ], dtype=np.float32)
        
        return state
    
    @property
    def equity_curve(self) -> list[float]:
        """Return the equity curve for analysis."""
        return self._equity_curve
    
    @property
    def sizing_history(self) -> list[float]:
        """Return the history of sizing multipliers."""
        return self._sizing_history
    
    @property
    def episode_stats(self) -> dict:
        """Return summary statistics for the episode."""
        curve = np.array(self._equity_curve)
        returns = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([0.0])
        
        final_equity = curve[-1] if len(curve) > 0 else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Sharpe (annualised, assuming hourly bars → 8760 bars/year)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760)
        else:
            sharpe = 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trade_count": self._trade_count,
            "avg_multiplier": float(np.mean(self._sizing_history)) if self._sizing_history else 1.0,
            "final_equity": final_equity,
        }
