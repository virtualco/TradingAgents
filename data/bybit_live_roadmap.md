# Pre-Live Validation & Bybit Deployment Roadmap
**Strategy:** CryptoDayTradingStrategy v2 (Regime-Aware Confluence)
**Target Exchange:** Bybit (Linear Perpetual / Spot)

---

## Executive Summary

The `CryptoDayTradingStrategy v2` has demonstrated exceptional performance (+41.6% weekly return, Sharpe 9.39) on a 60-day in-sample backtest. However, deploying a high-frequency day-trading algorithm directly to live capital based solely on in-sample data carries significant risk of overfitting and regime collapse.

To safely transition this strategy to live trading on Bybit, we must execute a rigorous 4-stage validation and deployment pipeline. This roadmap outlines the exact steps required to validate robustness, build the exchange infrastructure, and transition to live capital.

---

## Stage 1: Out-of-Sample & Walk-Forward Validation

Before writing any exchange integration code, we must prove the strategy's edge is robust across different market regimes (bull, bear, and sideways).

### 1.1 Extended Historical Backtesting
- **Data Acquisition:** Fetch 1-hour OHLCV data for BTC-USD and ETH-USD spanning **2022-01-01 to 2026-01-01** (capturing the 2022 bear market and 2023-2024 recovery).
- **Out-of-Sample Run:** Execute the exact v2 strategy on this unseen data without modifying any parameters.
- **Success Criteria:** Positive expectancy, Sharpe > 1.5, Max Drawdown < 25%.

### 1.2 Walk-Forward Analysis (WFA)
- **Methodology:** Implement a rolling window optimisation (e.g., 90 days training, 30 days testing) to ensure the strategy parameters (RSI thresholds, EMA periods) adapt dynamically to changing volatility.
- **Objective:** Confirm that the strategy does not rely on curve-fitted static parameters.

### 1.3 Slippage & Fee Stress Testing
- **Simulation:** Increase transaction costs from 0.1% to 0.25% to simulate worst-case slippage during high-volatility breakouts.
- **Success Criteria:** Strategy must remain profitable under stressed execution conditions.

---

## Stage 2: Bybit Infrastructure & Testnet Integration

Once the strategy's edge is validated, we build the execution layer connecting the TradingAgents framework to Bybit.

### 2.1 Exchange Connector Architecture
- **Library:** Implement `pybit` (official Bybit SDK) or `ccxt` for standardised exchange interaction.
- **Module:** Create `tradingagents/execution/bybit_connector.py` to handle authentication, order routing, and position management.
- **Account Type:** Target Bybit Unified Trading Account (UTA) using Linear Perpetual contracts (USDT-margined) to allow seamless long/short execution.

### 2.2 Testnet Paper Trading Environment
- **Setup:** Configure API keys for the Bybit Testnet.
- **Execution Loop:** Modify the existing `observer.py` to route approved orders to the Bybit Testnet rather than the internal SQLite simulation.
- **Data Feed:** Transition from `yfinance` daily fetches to Bybit's REST API for real-time 1-hour candle construction.

---

## Stage 3: Production Risk Management Framework

Live trading requires robust safeguards that sit *outside* the strategy logic to protect capital from black-swan events or API failures.

### 3.1 Dynamic Position Sizing
- **Current State:** Fixed percentage of NAV.
- **Upgrade:** Implement Kelly Criterion or Volatility-Targeted sizing based on the 14-period ATR. High volatility = smaller position size.

### 3.2 Hardened Kill Switches
- **Exchange-Level Stops:** Every order sent to Bybit MUST include a hard Stop Loss (SL) order attached at the exchange level, ensuring protection even if the TradingAgents server crashes.
- **Global Drawdown Circuit Breaker:** If the account NAV drops by >15% from its peak, the system automatically closes all positions, cancels all open orders, and halts trading until manual intervention.

### 3.3 Reconciliation Engine
- **State Sync:** Build a module that compares the TradingAgents internal SQLite state against the actual Bybit account state every hour.
- **Orphan Order Cleanup:** Automatically cancel any orders on Bybit that do not exist in the internal database.

---

## Stage 4: Live Deployment & Monitoring

The final transition to live capital.

### 4.1 Testnet Incubation (14 Days)
- Run the system autonomously on the Bybit Testnet for 14 consecutive days.
- **Success Criteria:** Zero API errors, zero reconciliation breaks, and execution prices matching the backtest assumptions within a 0.05% margin of error.

### 4.2 Live Deployment (Fractional Capital)
- **Initial Allocation:** Deploy with 5% of the target capital.
- **Monitoring:** Set up Slack/Discord webhooks for every trade execution, error, and daily summary.
- **Review:** Conduct weekly performance reviews comparing live execution against the theoretical backtest.

### 4.3 Full Scale
- Gradually scale up capital allocation in 20% increments every two weeks, provided the live Sharpe ratio remains > 1.5.

---

## Immediate Next Actions

To begin this process, I propose we start with **Stage 1: Out-of-Sample Validation**. 

I will build a script to fetch historical data from 2022-2024 and run the v2 strategy against it to confirm its robustness before we write any Bybit integration code.
