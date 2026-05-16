# TradingAgents Cloud Computer Deployment

## Prerequisites

- Manus Cloud Computer with Ubuntu 22.04
- Python 3.11+
- Bybit Testnet API credentials

## Quick Start

1. Clone the repository
2. Copy the env template and fill in credentials
3. Run: chmod +x deploy/setup.sh then ./deploy/setup.sh

## Services

trading-agents: Live trader v2 for 7 crypto symbols (always-on)
trading-agents-reoptimise: Monthly Bayesian re-optimisation (1st of month 02:00 UTC)

## Dashboard Integration

Point the dashboard metrics endpoint to http://<cloud-computer-ip>:8765/metrics
The dashboard will automatically switch from DEMO mode to LIVE mode.

## Architecture

Cloud Computer (always-on):
- trading-agents.service runs live_trader_v2.py (7 crypto symbols)
  - Bybit Testnet API for order execution
  - Port 8765 for JSON metrics endpoint
- trading-agents-reoptimise runs monthly Bayesian re-optimisation
  - Optuna (500 trials) for parameter search
  - Walk-forward validation for overfit detection
  - POST /api/scheduled/reoptimise to push results to dashboard
