#!/usr/bin/env python3
"""
weekly_review_task.py — Creates a Manus API task to run the weekly LLM risk review.

This script is designed to be called by a Manus scheduled task (weekly, Monday morning).
It triggers a new Manus task in the Quant Trade project that runs weekly_review.py
and posts the narrative risk commentary to the webhook.

Usage:
    python3 scripts/weekly_review_task.py

Required environment variables:
    MANUS_API_KEY          — Manus API key (from Manus Settings → Integrations → API)
    MANUS_PROJECT_ID       — Quant Trade project ID (HhwARwD5jLK7zSUXmkF84H)

Optional:
    TRADINGAGENTS_WEBHOOK_URL — Slack webhook (passed into the task prompt)
"""
import os
import sys
import json
import datetime
import requests

MANUS_API_KEY = os.environ.get("MANUS_API_KEY", "")
MANUS_PROJECT_ID = os.environ.get("MANUS_PROJECT_ID", "HhwARwD5jLK7zSUXmkF84H")
WEBHOOK_URL = os.environ.get("TRADINGAGENTS_WEBHOOK_URL", "")
API_BASE = "https://api.manus.ai"

if not MANUS_API_KEY:
    print("ERROR: MANUS_API_KEY is not set.")
    print("Get your API key from: https://manus.im/app#settings/integrations/api")
    sys.exit(1)

today = datetime.date.today().isoformat()
week_start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

prompt = f"""Run the TradingAgents weekly LLM risk review for the week of {week_start} to {today}.

Steps:
1. Clone or update the virtualco/TradingAgents repository
2. Run: python3 scripts/weekly_review.py --days 7
3. Read the generated weekly report from data/observation_reports/weekly_{today}.md
4. Summarise the key findings: NAV performance, drawdown, signal activity, risk flags
5. If TRADINGAGENTS_WEBHOOK_URL is set, confirm the Slack alert was sent

The Quant Trade project budget rules apply: use Lite model, stay under 20 steps.
"""

headers = {
    "x-manus-api-key": MANUS_API_KEY,
    "Content-Type": "application/json",
}

payload = {
    "message": {
        "content": [{"type": "text", "text": prompt}]
    },
    "project_id": MANUS_PROJECT_ID,
    "title": f"Weekly Risk Review — {today}",
}

print(f"Creating Manus task: Weekly Risk Review — {today}")
try:
    resp = requests.post(f"{API_BASE}/v2/task.create", headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("ok"):
        task_id = data.get("task_id", "")
        task_url = data.get("task_url", "")
        print(f"✓ Task created: {task_id}")
        print(f"  View at: {task_url}")
    else:
        print(f"ERROR: {data.get('error', {}).get('message', 'Unknown error')}")
        sys.exit(1)
except requests.exceptions.RequestException as e:
    print(f"ERROR: {e}")
    sys.exit(1)
