#!/bin/bash
# ============================================================
# TradingAgents — Quick Start Script
# Usage: ./deploy/start_trading.sh [testnet|mainnet] [dry|live]
# ============================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MODE="${1:-testnet}"
TRADE_MODE="${2:-dry}"

echo "=============================================="
echo "  TradingAgents Live Trader"
echo "  Mode:  $MODE"
echo "  Trade: $TRADE_MODE"
echo "=============================================="

# Load env file
if [ -f ".env.bybit" ]; then
  export $(grep -v '^#' .env.bybit | xargs)
  echo "✓ Loaded .env.bybit"
else
  echo "ERROR: .env.bybit not found. Copy .env.bybit.example and fill in your API keys."
  exit 1
fi

# Override based on args
if [ "$MODE" = "mainnet" ]; then
  export BYBIT_TESTNET=false
  echo "⚠️  MAINNET MODE — REAL MONEY"
else
  export BYBIT_TESTNET=true
  echo "✓ Testnet mode (paper trading)"
fi

if [ "$TRADE_MODE" = "live" ]; then
  export TRADING_DRY_RUN=false
  echo "⚠️  LIVE ORDER PLACEMENT ENABLED"
else
  export TRADING_DRY_RUN=true
  echo "✓ Dry run (signals only, no orders)"
fi

# Check dependencies
python3 -c "import pybit; print('✓ pybit OK')"
python3 -c "from tradingagents.research.dual_regime_strategy import DualRegimeStrategy; print('✓ DualRegimeStrategy OK')"
python3 -c "from tradingagents.execution.bybit_connector import BybitConnector; print('✓ BybitConnector OK')"
python3 -c "from tradingagents.execution.risk_manager import RiskManager; print('✓ RiskManager OK')"

echo ""
echo "Starting live trader..."
echo "Dashboard: open dashboard/index.html in your browser"
echo "Metrics:   http://localhost:${TRADING_METRICS_PORT:-8765}/metrics"
echo "Logs:      tail -f data/live_trader.log"
echo ""

python3 scripts/live_trader.py
