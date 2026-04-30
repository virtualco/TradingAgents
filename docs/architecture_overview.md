# TradingAgents — Architecture & Operations Overview

## 1. How Is the Daily Runner Triggered?

The daily runner is **not triggered from GitHub directly**. GitHub is purely the source-of-truth code repository. The execution model is:

```
Manus Scheduled Task (cron)
        │
        ▼
Manus Sandbox (this virtual machine)
        │  clones/pulls virtualco/TradingAgents
        │  runs: python3 scripts/run_daily.py --summary
        ▼
SQLite DB  →  data/paper_trading.db          (persists across runs)
JSON report →  data/observation_reports/daily_YYYY-MM-DD.json
```

**Key facts:**

- The task is triggered by a **Manus Scheduled Task** — a cron-style timer configured in Manus Settings → Scheduled Tasks.
- The sandbox is a persistent virtual machine. The DB and reports survive between runs because the sandbox hibernates (not destroys) between scheduled executions.
- The script pulls the latest code from `virtualco/TradingAgents` on each run (or uses the already-cloned copy), so any code pushed to GitHub is automatically picked up on the next scheduled execution.
- The `SCHEDULED_TASK_ENDPOINT_BASE` and `SCHEDULED_TASK_COOKIE` environment variables are injected by Manus at runtime, allowing the runner to POST results back to the Manus platform if needed.

**To change the schedule:** Go to [Manus Settings → Scheduled Tasks](https://manus.im/app#settings/scheduled-tasks).

---

## 2. Is This Built Into the virtualco Skills Database?

**Not currently — but it should be.** Here is the distinction:

| Layer | What It Is | Current State |
|---|---|---|
| **Manus Skills** (`/home/ubuntu/skills/`) | Reusable, modular capabilities injected into any Manus task | General skills exist (autoresearch, manus-api, etc.) — no TradingAgents-specific skill yet |
| **Project Instructions** (`Quant Trade` project) | Rules and context injected into every task in this project | Active — budget/step limits, model selection rules |
| **GitHub Repo** (`virtualco/TradingAgents`) | Versioned source code | Active — all scripts and modules live here |
| **Sandbox DB** (`data/paper_trading.db`) | Live paper trading state | Active — persists positions, orders, observations |

**Recommendation:** Create a `tradingagents-daily` skill in `/home/ubuntu/skills/` that encapsulates the playbook (how to run the daily cycle, what exit codes mean, how to interpret the summary). This would make the runner self-documenting and reusable across any Manus task in the project without needing to re-read the playbook each time.

---

## 3. Can AutoResearch Improve This Process?

**Yes — and it is well-suited for it.** The autoresearch skill runs a continuous propose-implement-evaluate-keep/discard loop using multiple LLMs. For TradingAgents, the natural application is:

### What AutoResearch Would Optimise

| Target | Metric | What Gets Improved |
|---|---|---|
| `strategy_rules.py` | Sharpe ratio from backfill | Signal weights, RSI/MACD thresholds, momentum windows |
| `run_daily.py` signal threshold | Trade count × win rate | The `0.20` score cutoff and conviction mapping |
| `ObservationConfig` | Max drawdown | `max_position_size_pct`, `min_conviction` |
| `kill_switch.py` | Circuit breaker events | `max_drawdown_pct`, `max_daily_loss_pct` thresholds |

### How It Would Work

```
AutoResearch Loop:
  1. Read program.md (defines: target files, eval command, goal metric)
  2. Propose a change to strategy_rules.py or ObservationConfig
  3. Run: python3 scripts/backfill_observations.py --days 89
         → captures final Sharpe ratio and max drawdown
  4. If Sharpe improved AND drawdown stayed <15%: git commit (keep)
     Else: git reset (discard)
  5. Repeat indefinitely
```

The eval command is already available (`backfill_observations.py` + `run_daily.py --summary`), and the test suite in `tests/test_execution_engine.py` provides a fast sanity-check gate before the full backfill eval.

**Important constraint:** AutoResearch must be run as a separate, explicitly-triggered Manus task — not as part of the daily scheduled run — to avoid burning credits on every cycle.

---

## 4. What Do We Do With the Output Data?

The system currently produces two output artefacts per day:

### Artefact 1 — Daily JSON Report
**Location:** `data/observation_reports/daily_YYYY-MM-DD.json`

**Contains:** NAV, cash, positions, P&L, drawdown, signal/order counts, kill switch status, reconciliation status.

**What to do with it:**

| Use Case | How |
|---|---|
| **Dashboard / monitoring** | Feed JSON files into a lightweight web dashboard (e.g., a Vite+React app deployed via Manus `web-static` scaffold) that charts NAV, drawdown, and trade history over time |
| **Alerting** | POST the JSON to a webhook (Slack, email, PagerDuty) when kill switch fires or drawdown exceeds a threshold |
| **Audit trail** | Archive to S3 or a cloud bucket for long-term record-keeping |
| **LLM analysis** | Pass the last N days of reports to an LLM agent (via Manus task) for a weekly narrative summary and risk commentary |

### Artefact 2 — SQLite Database
**Location:** `data/paper_trading.db`

**Contains:** Full order book, position history, daily observations, signal registry.

**What to do with it:**

| Use Case | How |
|---|---|
| **Performance analytics** | Query with DuckDB or pandas to compute rolling Sharpe, sector exposure, win/loss streaks |
| **Signal quality tracking** | Join `signals` table with `orders` to measure which technical rules generate the most profitable trades |
| **Walk-forward validation** | Feed into `tradingagents/research/walk_forward.py` to run out-of-sample strategy validation |
| **Transition to live trading** | When the system passes all readiness gates (currently: ✓ YES), the DB schema and order flow are already compatible with a real broker adapter |

### Recommended Next Steps for Output Data

1. **Build a monitoring dashboard** — a simple web app that reads the JSON reports and renders NAV curve, drawdown chart, and signal heatmap. This is a single Manus `web-static` task.
2. **Add a Slack/webhook alert** — post a daily summary message when the runner completes, and an urgent alert if kill switch or circuit breaker fires.
3. **Schedule a weekly LLM review** — a second scheduled Manus task (weekly, not daily) that reads the last 7 JSON reports and writes a narrative risk commentary.
4. **Run AutoResearch quarterly** — trigger an autoresearch session every quarter to re-optimise signal thresholds against the accumulated real-market data in the DB.
