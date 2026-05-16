#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# TradingAgents — Google Cloud Compute Engine Setup Script
# ═══════════════════════════════════════════════════════════════════════════════
# Usage:
#   1. SSH into your GCE instance
#   2. Clone the repo: git clone https://github.com/virtualco/TradingAgents.git /opt/trading-agents
#   3. Run: sudo bash /opt/trading-agents/deploy/gce-startup.sh
#   4. Configure: sudo nano /opt/trading-agents/deploy/.env
#   5. Start: sudo systemctl start trading-agents
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

INSTALL_DIR="/opt/trading-agents"
SERVICE_USER="ubuntu"
LOG_DIR="/var/log/trading-agents"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TradingAgents GCE Setup — v6.1                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl jq ufw

# ── 2. Firewall ───────────────────────────────────────────────────────────────
echo "[2/6] Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 8765/tcp  # Metrics endpoint
ufw --force enable

# ── 3. Python environment ────────────────────────────────────────────────────
echo "[3/6] Setting up Python virtual environment..."
cd "$INSTALL_DIR"
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install optuna pyarrow lightgbm scikit-learn joblib

# ── 4. Log directory ─────────────────────────────────────────────────────────
echo "[4/6] Creating log directory..."
mkdir -p "$LOG_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
mkdir -p "$INSTALL_DIR/data/models"
mkdir -p "$INSTALL_DIR/data/live_reports"

# ── 5. Environment file ──────────────────────────────────────────────────────
echo "[5/6] Preparing environment file..."
ENV_FILE="$INSTALL_DIR/deploy/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$INSTALL_DIR/deploy/.env.template" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
fi

# ── 6. Systemd services (installed but NOT started) ──────────────────────────
echo "[6/6] Installing systemd services..."

cat > /etc/systemd/system/trading-agents.service << 'EOF'
[Unit]
Description=TradingAgents Live Trader v2 — Multi-Asset Crypto Trading
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/trading-agents
EnvironmentFile=/opt/trading-agents/deploy/.env
ExecStart=/opt/trading-agents/venv/bin/python3 -u scripts/live_trader_v2.py
Restart=always
RestartSec=30
StandardOutput=append:/var/log/trading-agents/trader.log
StandardError=append:/var/log/trading-agents/trader.log

# Resource limits
MemoryMax=2G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/trading-agents-reoptimise.service << 'EOF'
[Unit]
Description=TradingAgents Monthly Bayesian Re-optimisation
After=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/opt/trading-agents
EnvironmentFile=/opt/trading-agents/deploy/.env
ExecStart=/opt/trading-agents/venv/bin/python3 scripts/run_reoptimise.py
StandardOutput=append:/var/log/trading-agents/reoptimise.log
StandardError=append:/var/log/trading-agents/reoptimise.log
TimeoutStartSec=3600
EOF

cat > /etc/systemd/system/trading-agents-reoptimise.timer << 'EOF'
[Unit]
Description=Monthly re-optimisation timer (1st of month, 02:00 UTC)

[Timer]
OnCalendar=*-*-01 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable trading-agents
systemctl enable trading-agents-reoptimise.timer

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ SETUP COMPLETE — Services installed but NOT started      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  NEXT STEPS:                                               ║"
echo "║  1. Edit credentials:                                      ║"
echo "║     sudo nano /opt/trading-agents/deploy/.env              ║"
echo "║                                                            ║"
echo "║  2. Start the trader:                                      ║"
echo "║     sudo systemctl start trading-agents                    ║"
echo "║     sudo systemctl start trading-agents-reoptimise.timer   ║"
echo "║                                                            ║"
echo "║  3. Verify:                                                ║"
echo "║     curl http://localhost:8765/metrics | jq .              ║"
echo "║                                                            ║"
echo "║  4. Point dashboard to:                                    ║"
echo "║     http://EXTERNAL_IP:8765/metrics                        ║"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
