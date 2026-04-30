# Alerts Setup Guide

This guide explains how to connect the TradingAgents daily runner to Slack (or
any webhook endpoint) so you receive automatic alerts after every daily cycle.

---

## What Gets Alerted

| Event | Severity | Example message |
|---|---|---|
| Daily cycle complete | Normal | NAV, daily P&L, signal count, readiness status |
| Drawdown > 10% | **Urgent** | :rotating_light: Drawdown 11.2% — review positions |
| Kill switch active | **Urgent** | :rotating_light: Kill switch ACTIVE — cycle halted |
| Circuit breaker triggered | **Urgent** | :rotating_light: Circuit breaker TRIGGERED |
| Reconciliation break | **Urgent** | :rotating_light: 2 reconciliation breaks detected |

---

## Step 1 — Create a Slack Incoming Webhook

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch**
2. Name it `TradingAgents` and select your workspace
3. In the left sidebar: **Incoming Webhooks** → toggle **Activate Incoming Webhooks** ON
4. Click **Add New Webhook to Workspace** → choose a channel (e.g. `#trading-alerts`)
5. Copy the webhook URL — it looks like:
   Format: `https://hooks.slack.com/services/<workspace>/<channel>/<token>`
   (never commit the real URL — store it as a Manus secret only)

> **Alternative:** Any HTTP endpoint that accepts a POST with `{"text": "..."}` works —
> Discord webhooks, Make.com, Zapier, n8n, or a custom server.

---

## Step 2 — Add the Secret to Manus

1. Open [Manus Settings → Secrets](https://manus.im/app#settings/scheduled-tasks)
2. Add a new secret:
   - **Name:** `TRADINGAGENTS_WEBHOOK_URL`
   - **Value:** your webhook URL from Step 1
3. Save. The secret is automatically injected as an environment variable on the
   next scheduled run.

---

## Step 3 — Verify the Pipeline

Run the test script from the sandbox:

```bash
cd /home/ubuntu/TradingAgents
export TRADINGAGENTS_WEBHOOK_URL="<paste your webhook URL here>"
python3 scripts/test_alert.py
```

Expected output:
```
Sending test alert to: https://hooks.slack.com/services/...
✓ Alert sent successfully (HTTP 200)
Check your Slack channel for the test message.
```

---

## Step 4 — Confirm Daily Runner Integration

The alert is automatically sent at the end of every daily cycle in
`scripts/run_daily.py`. No additional configuration is needed once the secret
is set. The payload format is:

```json
{
  "text": "📊 TradingAgents Daily — 2026-04-30\nNAV: $112,387.83 (+$245.12 today)\n..."
}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No message in Slack | `TRADINGAGENTS_WEBHOOK_URL` not set | Add secret in Manus Settings |
| HTTP 403 from webhook | Webhook URL revoked | Regenerate in Slack app settings |
| HTTP 400 from webhook | Malformed payload | Run `test_alert.py` and check output |
| Alert sent but no Slack message | Wrong channel permissions | Re-add webhook to the correct channel |
