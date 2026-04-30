#!/usr/bin/env python3
"""
test_alert.py — Send a test webhook alert to verify TRADINGAGENTS_WEBHOOK_URL is configured.

Usage:
    export TRADINGAGENTS_WEBHOOK_URL="<your-slack-webhook-url>"
    python3 scripts/test_alert.py

Exit codes:
    0 — alert sent successfully (HTTP 200)
    1 — TRADINGAGENTS_WEBHOOK_URL not set
    2 — HTTP error from webhook endpoint
"""
import os
import sys
import json
import datetime
import requests

WEBHOOK_URL = os.environ.get("TRADINGAGENTS_WEBHOOK_URL", "")

if not WEBHOOK_URL:
    print("ERROR: TRADINGAGENTS_WEBHOOK_URL is not set.")
    print()
    print("To configure:")
    print("  1. Get your Slack incoming webhook URL from:")
    print("     https://api.slack.com/messaging/webhooks")
    print("  2. Set it in Manus Settings → Secrets as TRADINGAGENTS_WEBHOOK_URL")
    print("  3. Re-run this script to verify.")
    sys.exit(1)

now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

payload = {
    "text": (
        ":white_check_mark: *TradingAgents Alert Test* — pipeline confirmed\n"
        f"Timestamp: {now}\n"
        "NAV: $112,387.83 (+12.39%) | Sharpe: 2.863 | Drawdown: 1.54%\n"
        "Status: *LIVE READY* — all readiness gates passed\n"
        "_This is a test message from scripts/test_alert.py_"
    )
}

print(f"Sending test alert to: {WEBHOOK_URL[:40]}...")
try:
    resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
    resp.raise_for_status()
    print(f"✓ Alert sent successfully (HTTP {resp.status_code})")
    print("Check your Slack channel for the test message.")
    sys.exit(0)
except requests.exceptions.HTTPError as e:
    print(f"ERROR: HTTP {resp.status_code} — {resp.text}")
    sys.exit(2)
except requests.exceptions.RequestException as e:
    print(f"ERROR: {e}")
    sys.exit(2)
