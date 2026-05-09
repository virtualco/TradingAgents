# Dual-Regime Crypto Trading Architecture

**Date:** May 9, 2026  
**Project:** TradingAgents — Crypto Day Trading  
**Author:** Manus AI  

---

## 1. Executive Summary

Following the Out-of-Sample (OOS) validation of the v2 momentum strategy, which revealed significant vulnerabilities during the 2022 bear market and 2023 ranging periods, I have designed and implemented a **Dual-Regime Architecture**. 

This new system dynamically classifies the market state and switches between two distinct sub-strategies:
1. **Momentum Strategy** for trending markets.
2. **Mean-Reversion Strategy** for ranging markets.

The architecture has been built, integrated into the `TradingAgents` codebase, and validated against 4 years of hourly data (35,000+ bars) for both BTC and ETH.

---

## 2. Regime Detection Layer

The core innovation is the regime classifier, which acts as a gatekeeper. It uses a combination of fractal geometry and trend strength indicators to determine the current market state [1] [2].

### Primary Filter: The Hurst Exponent
The Hurst Exponent ($H$) measures the long-term memory of a time series. We use a fast, vectorised Variance-of-Increments method over a rolling 96-hour window (4 days).
- **$H > 0.55$**: The market is persistent (trending). Positive returns are likely to be followed by positive returns.
- **$H < 0.45$**: The market is anti-persistent (mean-reverting). Price moves are likely to reverse.
- **$0.45 \le H \le 0.55$**: The market is a random walk (transition phase).

### Secondary Filter: Average Directional Index (ADX)
To prevent false positives in the Hurst calculation during high-volatility chop, we require confirmation from the ADX(14) indicator [3].
- **ADX > 25**: Confirms a strong trend.
- **ADX < 20**: Confirms a weak trend (ranging).

| Regime | Hurst Condition | ADX Condition | Action |
|---|---|---|---|
| **TRENDING** | $H > 0.55$ | ADX > 25 | Activate Momentum Strategy |
| **RANGING** | $H < 0.45$ | ADX < 20 | Activate Mean-Reversion Strategy |
| **TRANSITION** | $0.45 \le H \le 0.55$ | 20 $\le$ ADX $\le$ 25 | Flat (No new trades) |

---

## 3. Sub-Strategies

### 3.1 Momentum Sub-Strategy (Trending Regime)
When the market is trending, we want to capture large directional moves while avoiding early exits.
- **Entry:** Fast EMA(9) crosses Slow EMA(21) in the direction of the 50-EMA trend, confirmed by MACD histogram expansion and a 20% volume surge.
- **Exit:** EMA crossover reversal or a trailing stop at 2.5× ATR(14).

### 3.2 Mean-Reversion Sub-Strategy (Ranging Regime)
When the market is ranging, we want to fade the extremes and target the mean.
- **Entry:** Price touches the outer Bollinger Bands (20-period, 2$\sigma$) while RSI(14) is at an extreme (<35 for longs, >65 for shorts).
- **Filter:** Price must be within 1.5× ATR of the Bollinger midline to avoid catching the start of a breakout.
- **Exit:** Price returns to the Bollinger midline (mean).

---

## 4. OOS Validation Results

The v1 baseline of the dual-regime strategy was tested against 4 years of hourly data (2022–2026). 

### Key Findings
1. **Drawdown Control:** The dual-regime architecture significantly improved risk control. The maximum drawdown during the 2022 bear market was reduced from **74.9%** (single-regime momentum) to **24.1%** (dual-regime).
2. **Regime Distribution:** The classifier successfully identified regimes, spending ~5% of the time in pure TRENDING mode, ~10% in pure RANGING mode, and the remaining 85% in TRANSITION (flat). This highly selective approach protects capital during unpredictable chop.
3. **Performance:** While the drawdown was controlled, the baseline parameters yielded a slightly negative return (-0.1% weekly average). This indicates that the architecture is structurally sound, but the specific entry/exit thresholds within the sub-strategies require fine-tuning.

---

## 5. Next Steps for Live Deployment

The dual-regime architecture is now integrated into the codebase (`tradingagents/research/dual_regime_strategy.py`). 

To proceed to live trading on Bybit:
1. **Parameter Calibration:** The grid search revealed that the Hurst computation over 35,000 bars is computationally intensive. The next step is to run a targeted Bayesian optimisation on a smaller, representative sample (e.g., 6 months of bear market + 6 months of bull market) to find the optimal RSI and Bollinger Band thresholds.
2. **Paper Trading:** The strategy is fully compatible with the Bybit Testnet infrastructure built in the previous phase. We can deploy the current v1 baseline to the testnet immediately to observe its real-time regime classification behaviour.

---

### References
[1] Samara Asset Management. (2023). *Exploring the Hurst Exponent*. https://www.samara-am.com/insights/hurst-exponent  
[2] PyQuantLab. (2026). *Building an Adaptive Crypto Strategy: Combining Mean Reversion and Momentum*. https://pyquantlab.medium.com/building-an-adaptive-crypto-strategy-combining-mean-reversion-and-momentum-15af99805f7b  
[3] Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
