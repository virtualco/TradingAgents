# Weekly LLM Risk Review — Setup Guide

This guide explains how to register the weekly risk review as a recurring
Manus scheduled task so you receive a narrative risk commentary every Monday.

---

## What the Weekly Review Does

Every Monday morning, the system:

1. Reads the last 7 daily JSON reports from `data/observation_reports/`
2. Calls GPT-4.1-mini with a structured prompt asking for a narrative risk commentary
3. Saves the output to `data/observation_reports/weekly_YYYY-MM-DD.md`
4. Posts a summary to Slack (if `TRADINGAGENTS_WEBHOOK_URL` is set)

Sample output sections:
- **Performance Summary** — NAV change, daily P&L distribution, best/worst day
- **Risk Assessment** — drawdown trend, kill switch proximity, position concentration
- **Signal Activity** — which tickers generated signals, conviction levels
- **Outlook** — forward-looking commentary based on signal scores
- **Action Items** — any recommended parameter adjustments

---

## Step 1 — Get Your Manus API Key

1. Go to [Manus Settings → Integrations → API](https://manus.im/app#settings/integrations/api)
2. Click **Create API Key** → name it `TradingAgents Weekly Review`
3. Copy the key — you will not be able to see it again

---

## Step 2 — Add Secrets to Manus

In [Manus Settings → Secrets](https://manus.im/app#settings/scheduled-tasks), add:

| Secret Name | Value |
|---|---|
| `MANUS_API_KEY` | Your API key from Step 1 |
| `MANUS_PROJECT_ID` | `HhwARwD5jLK7zSUXmkF84H` (Quant Trade project) |

---

## Step 3 — Create the Weekly Scheduled Task

1. Go to [Manus Settings → Scheduled Tasks](https://manus.im/app#settings/scheduled-tasks)
2. Click **New Scheduled Task**
3. Configure:
   - **Name:** `TradingAgents Weekly Review`
   - **Schedule:** Every Monday at 9:00 AM (your timezone)
   - **Prompt:**
     ```
     Run the TradingAgents weekly LLM risk review.
     cd /home/ubuntu/TradingAgents && python3 scripts/weekly_review.py --days 7
     Then summarise the key findings from the generated weekly report.
     ```
4. **Project:** Select `Quant Trade`
5. Save

---

## Step 4 — Verify (Manual Test Run)

To test without waiting for Monday:

```bash
cd /home/ubuntu/TradingAgents
export MANUS_API_KEY=your_key_here
export MANUS_PROJECT_ID=HhwARwD5jLK7zSUXmkF84H
python3 scripts/weekly_review_task.py
```

This creates a one-off Manus task immediately. Check the task URL printed to
stdout to see the review in progress.

---

## Output Location

| File | Description |
|---|---|
| `data/observation_reports/weekly_YYYY-MM-DD.md` | Full narrative review |
| Slack channel | Summary posted via webhook (if configured) |

---

## Relationship to Daily Runner

| Task | Frequency | Script | Model |
|---|---|---|---|
| Daily observation cycle | Every trading day | `scripts/run_daily.py` | No LLM (rule-based) |
| Weekly LLM risk review | Every Monday | `scripts/weekly_review.py` | GPT-4.1-mini (Lite) |
| AutoResearch optimisation | Quarterly (manual) | `autoresearch/eval.sh` | GPT-4.1 (triggered manually) |
