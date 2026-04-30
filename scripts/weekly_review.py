#!/usr/bin/env python3
"""Weekly LLM Risk Review for TradingAgents.

Reads the last 7 daily JSON reports, queries GPT-4.1-mini for a narrative
risk commentary, and saves the output as a Markdown report.

Usage:
    python3 scripts/weekly_review.py
    python3 scripts/weekly_review.py --days 14 --output reports/week_review.md

Environment variables:
    OPENAI_API_KEY              Required (pre-configured in Manus sandbox)
    TRADINGAGENTS_REPORTS       Directory containing daily_YYYY-MM-DD.json files
    TRADINGAGENTS_DB            Path to SQLite DB (for observation period summary)
    TRADINGAGENTS_WEBHOOK_URL   Optional Slack webhook to post the review summary
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("weekly_review")


def load_recent_reports(report_dir: str, days: int = 7) -> list[dict]:
    """Load the most recent `days` daily JSON reports, oldest first."""
    report_path = Path(report_dir)
    reports = []
    for i in range(days - 1, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        f = report_path / f"daily_{d}.json"
        if f.exists():
            try:
                reports.append(json.loads(f.read_text()))
            except Exception as e:
                logger.warning(f"Could not read {f}: {e}")
    logger.info(f"Loaded {len(reports)} daily reports from {report_dir}")
    return reports


def build_prompt(reports: list[dict], db_path: str) -> str:
    """Build the LLM prompt from the weekly report data."""
    # Observation period summary
    obs_summary = ""
    try:
        from tradingagents.execution.observer import ObservationLogger
        obs_logger = ObservationLogger(db_path=db_path)
        summary = obs_logger.get_summary()
        obs_summary = summary.summary()
    except Exception as e:
        obs_summary = f"(Could not load observation summary: {e})"

    # Format the weekly data as a compact table
    rows = []
    for r in reports:
        rows.append(
            f"  {r['trade_date']}  NAV=${r['nav']:>12,.2f}  "
            f"DailyPnL={r['daily_pnl']:>+10,.2f}  "
            f"Drawdown={r['drawdown_pct']*100:.2f}%  "
            f"Signals={r['signals_received']}  Filled={r['orders_filled']}  "
            f"KS={'YES' if r['kill_switch_active'] else 'no'}  "
            f"CB={'YES' if r['circuit_breaker_triggered'] else 'no'}  "
            f"Recon={'CLEAN' if r['reconciliation_clean'] else 'BREAKS'}"
        )
    weekly_table = "\n".join(rows) if rows else "  (no data)"

    prompt = f"""You are a senior quantitative portfolio risk manager reviewing a paper trading system.

## Weekly Performance Data (last {len(reports)} trading days)

{weekly_table}

## Full Observation Period Summary

{obs_summary}

## Your Task

Write a concise weekly risk commentary in professional Markdown. Structure it as:

1. **Executive Summary** (2-3 sentences: overall performance, key highlights)
2. **P&L Analysis** (daily P&L trend, best/worst days, consistency)
3. **Risk Assessment** (drawdown profile, kill switch / circuit breaker events, reconciliation health)
4. **Signal & Execution Quality** (signal generation rate, fill rate, rejection analysis)
5. **Readiness Assessment** (progress toward live-trading readiness gates)
6. **Recommendations** (1-3 specific, actionable improvements for next week)

Be precise, data-driven, and avoid generic statements. Flag any anomalies or concerns clearly.
Keep the total response under 600 words.
"""
    return prompt


def call_llm(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """Call the OpenAI-compatible API and return the response text."""
    from openai import OpenAI
    client = OpenAI()  # API key and base URL pre-configured in sandbox
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def save_review(content: str, output_path: Path, trade_date: str) -> Path:
    """Save the review as a Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# TradingAgents Weekly Risk Review — {trade_date}\n\n"
    output_path.write_text(header + content + "\n")
    logger.info(f"Weekly review saved to {output_path}")
    return output_path


def send_slack_summary(content: str, trade_date: str) -> None:
    """Post a truncated summary to Slack if webhook is configured."""
    webhook_url = os.environ.get("TRADINGAGENTS_WEBHOOK_URL", "")
    if not webhook_url:
        return
    # Extract first paragraph as Slack message
    first_para = content.split("\n\n")[0].replace("**", "*").replace("##", "")
    text = f":memo: *TradingAgents Weekly Review — {trade_date}*\n{first_para}"
    try:
        import urllib.request
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(f"Weekly review summary posted to Slack (HTTP {resp.status})")
    except Exception as e:
        logger.warning(f"Slack post failed (non-fatal): {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TradingAgents Weekly LLM Risk Review")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of recent daily reports to include (default: 7)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output Markdown file path (default: data/observation_reports/weekly_YYYY-MM-DD.md)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="LLM model to use (default: gpt-4.1-mini)")
    parser.add_argument("--db", type=str,
                        default=os.environ.get(
                            "TRADINGAGENTS_DB",
                            str(REPO_ROOT / "data" / "paper_trading.db")
                        ))
    parser.add_argument("--reports", type=str,
                        default=os.environ.get(
                            "TRADINGAGENTS_REPORTS",
                            str(REPO_ROOT / "data" / "observation_reports")
                        ))
    args = parser.parse_args()

    trade_date = date.today().isoformat()
    output_path = Path(args.output) if args.output else (
        Path(args.reports) / f"weekly_{trade_date}.md"
    )

    logger.info(f"Weekly review — {trade_date} (last {args.days} days)")

    # Load reports
    reports = load_recent_reports(args.reports, days=args.days)
    if not reports:
        logger.error("No daily reports found — cannot generate review")
        return 1

    # Build prompt and call LLM
    logger.info(f"Calling {args.model} for risk commentary…")
    prompt = build_prompt(reports, args.db)
    try:
        review_text = call_llm(prompt, model=args.model)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return 1

    # Save and alert
    save_review(review_text, output_path, trade_date)
    send_slack_summary(review_text, trade_date)

    print(f"\n{'='*60}")
    print(f"WEEKLY REVIEW COMPLETE — {trade_date}")
    print(f"{'='*60}")
    print(review_text)
    print(f"\nSaved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
