#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# TradingAgents — Google Cloud Compute Engine Startup Script
# ═══════════════════════════════════════════════════════════════════════════════
# Usage: Run this on a fresh GCE instance (Ubuntu 22.04 LTS) or pass as
#        --metadata-from-file startup-script=deploy/gce-startup.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_URL="https://github.com/virtualco/TradingAgents.git"
INSTALL_DIR="/opt/trading-agents"
SERVICE_USER="trading"
LOG_DIR="/var/log/trading-agents"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TradingAgents GCE Deployment — v6.1                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl jq ufw

# ── 2. Firewall ───────────────────────────────────────────────────────────────
echo "[2/8] Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 8765/tcp  # Metrics endpoint
ufw --force enable

# ── 3. Create service user ────────────────────────────────────────────────────
echo "[3/8] Creating service user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --home-dir "$INSTALL_DIR" --shell /bin/bash "$SERVICE_USER"
fi

# ── 4. Clone repository ──────────────────────────────────────────────────────
echo "[4/8] Cloning repository..."
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR" && git pull --ff-only
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# ── 5. Python environment ────────────────────────────────────────────────────
echo "[5/8] Setting up Python virtual environment..."
cd "$INSTALL_DIR"
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install optuna pyarrow lightgbm scikit-learn joblib

# ── 6. Log directory ─────────────────────────────────────────────────────────
echo "[6/8] Creating log directory..."
mkdir -p "$LOG_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"

# ── 7. Environment file ──────────────────────────────────────────────────────
echo "[7/8] Checking environment file..."
ENV_FILE="$INSTALL_DIR/deploy/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$INSTALL_DIR/deploy/.env.template" "$ENV_FILE"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ACTION REQUIRED: Edit $ENV_FILE                            ║"
    echo "║  Fill in your Bybit API keys and dashboard URL              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
fi
chown "$SERVICE_USER:$SERVICE_USER" "$ENV_FILE"
chmod 600 "$ENV_FILE"

# ── 8. Systemd services ──────────────────────────────────────────────────────
echo "[8/8] Installing systemd services..."

cat > /etc/systemd/system/trading-agents.service << 'EOF'
[Unit]
Description=TradingAgents Live Trader v2
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/trading-agents
EnvironmentFile=/opt/trading-agents/deploy/.env
ExecStart=/opt/trading-agents/venv/bin/python3 scripts/live_trader_v2.py
Restart=always
RestartSec=30
StandardOutput=append:/var/log/trading-agents/trader.log
StandardError=append:/var/log/trading-agents/trader.log

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/opt/trading-agents/data /var/log/trading-agents

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/trading-agents-reoptimise.service << 'EOF'
[Unit]
Description=TradingAgents Monthly Bayesian Re-optimisation
After=network-online.target

[Service]
Type=oneshot
User=trading
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
systemctl enable --now trading-agents
systemctl enable --now trading-agents-reoptimise.timer

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ DEPLOYMENT COMPLETE                                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Live Trader:  systemctl status trading-agents              ║"
echo "║  Re-optimise:  systemctl list-timers                        ║"
echo "║  Metrics:      http://$(curl -s ifconfig.me):8765/metrics   ║"
echo "║  Logs:         journalctl -u trading-agents -f              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Point your dashboard Settings → Metrics Endpoint to:"
echo "  http://$(curl -s ifconfig.me):8765/metrics"
