# TradingAgents — Google Cloud Compute Engine Deployment Guide

## Overview

This guide deploys the TradingAgents live trading system to a Google Cloud Compute Engine (GCE) instance. The system runs 24/7, executing trades on Bybit Testnet across 8 crypto symbols with regime-aware signal generation and a JSON metrics endpoint for the dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  GCE Instance (e2-small, Ubuntu 22.04)                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  trading-agents.service (systemd, always-on)        │    │
│  │  ├── live_trader_v2.py                              │    │
│  │  │   ├── PerAssetRouter (3 strategies)              │    │
│  │  │   ├── ML Regime Detector (LightGBM)              │    │
│  │  │   ├── BybitConnector (Testnet API)               │    │
│  │  │   └── Metrics HTTP Server (:8765)                │    │
│  │  └── SQLite state persistence                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  trading-agents-reoptimise.timer (monthly)          │    │
│  │  └── Bayesian optimisation (500 Optuna trials)      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   Bybit Testnet API            Dashboard (manus.space)
   (order execution)            (metrics display)
```

## Prerequisites

1. Google Cloud account with billing enabled
2. `gcloud` CLI installed and authenticated
3. Bybit Testnet API key and secret (get from https://testnet.bybit.com)

## Step 1: Create the GCE Instance

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create the instance (e2-small: 2 vCPU, 2 GB RAM — $15/month)
gcloud compute instances create trading-agents \
  --zone=us-central1-a \
  --machine-type=e2-small \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --tags=trading-agents \
  --metadata-from-file startup-script=deploy/gce-startup.sh
```

**Alternative (manual setup — recommended):**
```bash
# Create instance without startup script
gcloud compute instances create trading-agents \
  --zone=us-central1-a \
  --machine-type=e2-small \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --tags=trading-agents
```

## Step 2: Configure Firewall

```bash
# Allow metrics endpoint (port 8765) from anywhere
gcloud compute firewall-rules create allow-trading-metrics \
  --allow tcp:8765 \
  --target-tags=trading-agents \
  --source-ranges=0.0.0.0/0 \
  --description="Allow TradingAgents metrics endpoint"
```

## Step 3: SSH and Configure

```bash
# SSH into the instance
gcloud compute ssh trading-agents --zone=us-central1-a

# Clone the repository
sudo git clone https://github.com/virtualco/TradingAgents.git /opt/trading-agents
sudo chown -R ubuntu:ubuntu /opt/trading-agents

# Run the setup script (installs deps, configures firewall, installs systemd services)
sudo bash /opt/trading-agents/deploy/gce-startup.sh
```

Note: The setup script installs services but does **NOT** start them. You must configure `.env` first (Step 4).

## Step 4: Configure Environment Variables

```bash
sudo nano /opt/trading-agents/deploy/.env
```

Fill in the following values:

```env
# Bybit Testnet API credentials (from https://testnet.bybit.com)
BYBIT_TESTNET_API_KEY=your_api_key_here
BYBIT_TESTNET_API_SECRET=your_api_secret_here

# Dashboard endpoint (your deployed dashboard URL)
DASHBOARD_URL=https://tradingdash-dyvbtahs.manus.space

# Metrics server port
METRICS_PORT=8765

# Trading symbols
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,BNBUSDT,XRPUSDT,LINKUSDT,AVAXUSDT

# Risk management
MAX_POSITION_USD=100
DRY_RUN=false
```

## Step 5: Start the Services

```bash
# Restart to pick up new env
sudo systemctl restart trading-agents

# Verify it's running
sudo systemctl status trading-agents

# Check logs
sudo journalctl -u trading-agents -f --no-pager

# Verify metrics endpoint
curl http://localhost:8765/metrics | jq .
```

## Step 6: Connect the Dashboard

1. Get your instance's external IP:
   ```bash
   gcloud compute instances describe trading-agents \
     --zone=us-central1-a \
     --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
   ```

2. Go to your dashboard: https://tradingdash-dyvbtahs.manus.space

3. Navigate to **Settings** → **LIVE Mode Configuration**

4. Enter the metrics endpoint URL:
   ```
   http://YOUR_EXTERNAL_IP:8765/metrics
   ```

5. Click **Test Connection** — you should see a green checkmark

6. The dashboard will automatically switch from DEMO to LIVE mode

## Monitoring & Maintenance

### Check service status
```bash
sudo systemctl status trading-agents
sudo systemctl status trading-agents-reoptimise.timer
```

### View live logs
```bash
# Trader logs
sudo journalctl -u trading-agents -f

# Re-optimisation logs
sudo journalctl -u trading-agents-reoptimise -f
```

### Restart after code update
```bash
cd /opt/trading-agents
sudo -u trading git pull
sudo systemctl restart trading-agents
```

### Manual re-optimisation
```bash
sudo systemctl start trading-agents-reoptimise
```

## Cost Estimate

| Resource | Monthly Cost |
|----------|-------------|
| e2-small instance (2 vCPU, 2 GB) | ~$15 |
| 20 GB boot disk | ~$1 |
| Network egress (minimal) | ~$1 |
| **Total** | **~$17/month** |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Service won't start | Check `.env` file: `sudo cat /opt/trading-agents/deploy/.env` |
| Metrics unreachable | Verify firewall: `gcloud compute firewall-rules list --filter="name=allow-trading-metrics"` |
| Permission denied | Fix ownership: `sudo chown -R trading:trading /opt/trading-agents` |
| Out of memory | Upgrade to e2-medium: `gcloud compute instances set-machine-type trading-agents --machine-type=e2-medium --zone=us-central1-a` (requires stop first) |
| Model not loading | Ensure `data/models/regime_gbm_v1.joblib` exists — retrain with `python3 scripts/train_regime_model.py` |

## Security Notes

- The `.env` file has `chmod 600` (owner-only read)
- The service runs as a dedicated `trading` user (not root)
- UFW firewall only exposes ports 22 (SSH) and 8765 (metrics)
- For production: consider adding HTTPS via nginx reverse proxy + Let's Encrypt
- For production: restrict port 8765 to your dashboard's IP only
