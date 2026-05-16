#!/bin/bash
set -euo pipefail

echo "=== TradingAgents Cloud Computer Setup ==="

# Install Python dependencies
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt
pip install optuna pyarrow

# Create log directory
echo "[2/5] Creating log directory..."
sudo mkdir -p /var/log/trading-agents
sudo chown ubuntu:ubuntu /var/log/trading-agents

# Check env file
echo "[3/5] Checking environment file..."
if [ ! -f deploy/.env ]; then
  echo "ERROR: deploy/.env not found. Copy deploy/.env.template to deploy/.env and fill in credentials."
  exit 1
fi

# Install systemd services
echo "[4/5] Installing systemd services..."
sudo cp deploy/trading-agents.service /etc/systemd/system/
sudo cp deploy/trading-agents-reoptimise.service /etc/systemd/system/
sudo cp deploy/trading-agents-reoptimise.timer /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start
echo "[5/5] Enabling and starting services..."
sudo systemctl enable --now trading-agents
sudo systemctl enable --now trading-agents-reoptimise.timer

echo ""
echo "=== Setup Complete ==="
echo "Live trader: sudo systemctl status trading-agents"
echo "Re-optimise timer: sudo systemctl list-timers trading-agents-reoptimise.timer"
echo "Metrics endpoint: http://localhost:8765/metrics"
echo ""
echo "Point your dashboard to: http://$(hostname -I | awk '{print $1}'):8765/metrics"
