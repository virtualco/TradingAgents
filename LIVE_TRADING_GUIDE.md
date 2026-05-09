# TradingAgents — Live Trading Deployment Guide

**Branch:** `autoresearch/crypto-daytrading-v1`  
**Date:** May 9, 2026  

---

## Architecture Overview

The system is composed of four tightly integrated layers:

```
┌─────────────────────────────────────────────────────────┐
│                    live_trader.py                        │
│              (Production Orchestrator)                   │
├──────────────┬──────────────────────┬───────────────────┤
│ DualRegime   │  BybitConnector      │  RiskManager      │
│ Strategy     │  (V5 Unified API)    │  (Circuit Breaker │
│              │                      │   Kill Switch     │
│ Hurst + ADX  │  Testnet / Mainnet   │   ATR Sizing)     │
│ Momentum     │  Market Orders       │                   │
│ Mean-Rev     │  Stop-Loss / TP      │                   │
├──────────────┴──────────────────────┴───────────────────┤
│              SQLite State DB (live_trading.db)           │
├─────────────────────────────────────────────────────────┤
│         Metrics HTTP Server (:8765/metrics)              │
│         Dashboard (dashboard/index.html)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Get Bybit API Keys

### Testnet (Recommended First)
1. Go to [https://testnet.bybit.com](https://testnet.bybit.com)
2. Create an account and navigate to **API Management**
3. Create a new key with **Read + Trade** permissions
4. Copy the API Key and Secret

### Mainnet (Live Money — when ready)
1. Go to [https://www.bybit.com/app/user/api-management](https://www.bybit.com/app/user/api-management)
2. Create a key with **Unified Trading** permissions
3. **Restrict to your server IP** for security

---

## Step 2: Configure Environment

```bash
cd /home/ubuntu/TradingAgents
cp .env.bybit.example .env.bybit
nano .env.bybit
```

Fill in your keys and adjust parameters:

| Variable | Default | Description |
|---|---|---|
| `BYBIT_API_KEY` | — | Your Bybit API key |
| `BYBIT_API_SECRET` | — | Your Bybit API secret |
| `BYBIT_TESTNET` | `true` | `false` for mainnet |
| `TRADING_SYMBOLS` | `BTCUSDT,ETHUSDT` | Symbols to trade |
| `TRADING_INTERVAL` | `60` | Candle interval (minutes) |
| `TRADING_CAPITAL_USDT` | `1000` | Capital per symbol |
| `TRADING_MAX_POSITIONS` | `2` | Max concurrent positions |
| `TRADING_MAX_DD_PCT` | `5.0` | Daily drawdown circuit breaker |
| `TRADING_DRY_RUN` | `true` | `false` to place real orders |

---

## Step 3: Start Trading

### Option A — Quick Start (Testnet, Dry Run)
```bash
./deploy/start_trading.sh testnet dry
```

### Option B — Testnet with Real Testnet Orders
```bash
./deploy/start_trading.sh testnet live
```

### Option C — Mainnet Live Trading (REAL MONEY)
```bash
# Only when you are confident in the strategy
./deploy/start_trading.sh mainnet live
```

### Option D — Direct Python
```bash
source .env.bybit
python3 scripts/live_trader.py
```

---

## Step 4: Monitor

### Dashboard
Open `dashboard/index.html` in your browser. Set the endpoint to:
```
http://localhost:8765/metrics
```

The dashboard auto-refreshes every 10 seconds and shows:
- **Regime Detection:** Hurst exponent, ADX, current regime per symbol
- **Open Positions:** Side, quantity, entry price, stop-loss
- **Recent Trades:** P&L per trade, regime at entry
- **Risk Manager:** Daily drawdown, circuit breaker status, kill switch
- **Signal Log:** Full signal diagnostics per cycle

### Logs
```bash
tail -f data/live_trader.log
```

### Metrics API
```bash
curl http://localhost:8765/metrics | python3 -m json.tool
```

---

## Step 5: Production Deployment (systemd)

For 24/7 operation on a Linux server:

```bash
# Copy service file
sudo cp deploy/tradingagents.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable tradingagents
sudo systemctl start tradingagents

# Check status
sudo systemctl status tradingagents
sudo journalctl -u tradingagents -f
```

---

## Risk Controls

The system has three layers of protection:

| Control | Trigger | Action |
|---|---|---|
| **Stop-Loss** | Price hits 2× ATR below entry | Close position immediately |
| **Circuit Breaker** | Daily drawdown exceeds `TRADING_MAX_DD_PCT` | Halt all new trades |
| **Kill Switch** | `TRADING_KILL_SWITCH=true` env var | Emergency halt all trading |

To activate the kill switch immediately:
```bash
export TRADING_KILL_SWITCH=true
# or create the file:
touch data/kill_switch.flag
```

---

## Recommended Phased Rollout

| Phase | Duration | Mode | Capital |
|---|---|---|---|
| 1. Testnet Dry Run | 1 week | Testnet + Dry Run | $0 |
| 2. Testnet Live Orders | 2 weeks | Testnet + Live Orders | Testnet USDT |
| 3. Mainnet Small | 2 weeks | Mainnet + Live | $100–$500 |
| 4. Mainnet Full | Ongoing | Mainnet + Live | Full allocation |

**Do not skip phases.** The testnet period is critical for validating the execution pipeline before risking real capital.
