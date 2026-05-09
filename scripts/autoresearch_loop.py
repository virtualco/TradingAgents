#!/usr/bin/env python3
"""
AutoResearch Loop Orchestrator — Crypto Day-Trading Algorithm
=============================================================
Runs the propose → implement → evaluate → ratchet loop autonomously.
Multi-LLM strategy:
  - Generator: gpt-4.1-mini (creative, proposes code changes)
  - Critic:    gemini-2.5-flash (analytical, catches bugs)
  - Evaluator: gpt-4.1-nano (fast, scores the improvement)

Key fix: commits the strategy file BEFORE running eval, so ratchet
reset --hard HEAD~1 correctly reverts to the previous good state.
"""
from __future__ import annotations
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_FILE = REPO_ROOT / "tradingagents" / "research" / "crypto_strategy.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_crypto_strategy.py"
RESULTS_TSV = REPO_ROOT / "results.tsv"
RATCHET_SCRIPT = Path("/home/ubuntu/skills/autoresearch/scripts/ratchet.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("autoresearch")

client = OpenAI()

# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""

def read_results_history() -> str:
    if not RESULTS_TSV.exists():
        return "No previous results."
    lines = RESULTS_TSV.read_text().strip().split("\n")
    return "\n".join(lines[-15:])

def run_eval() -> tuple[float, str]:
    """Run the evaluation harness and parse the metric."""
    try:
        result = subprocess.run(
            ["python3", str(EVAL_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        match = re.search(r"AUTORESEARCH_METRIC:\s*([\d.]+)", output)
        if match:
            return float(match.group(1)), output
        if "AUTORESEARCH_CRASH" in output:
            return -1.0, output
        return 0.0, output
    except subprocess.TimeoutExpired:
        return -1.0, "AUTORESEARCH_CRASH: timeout"
    except Exception as e:
        return -1.0, f"AUTORESEARCH_CRASH: {e}"

def run_ratchet(metric: float, description: str) -> str:
    try:
        result = subprocess.run(
            ["python3", str(RATCHET_SCRIPT), str(REPO_ROOT), str(metric), description],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Ratchet error: {e}"

def git_commit(message: str):
    subprocess.run(["git", "add", str(STRATEGY_FILE)], cwd=str(REPO_ROOT), capture_output=True)
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    return result.returncode == 0

def get_current_best() -> float:
    if not RESULTS_TSV.exists():
        return 0.0
    import csv
    best = 0.0
    try:
        with open(RESULTS_TSV) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("status") == "keep":
                    best = max(best, float(row.get("metric", 0)))
    except Exception:
        pass
    return best

# ─── Multi-LLM Pipeline ──────────────────────────────────────────────────────

GENERATOR_SYSTEM = """You are a world-class quantitative researcher specialising in crypto algorithmic trading.
Current best score: {best_score:.1f}/100 (target: 100/100)

The scoring metric is:
  40% weekly return (target: 30%/week = score 1.0, capped)
  40% Sharpe ratio (target: 3.0 = score 1.0, capped)
  20% win rate (0-1)
  minus 30% penalty for drawdown > 50%

The CURRENT strategy already achieves ~260% weekly return and Sharpe ~15.
The main bottleneck is WIN RATE (currently ~47%) and DRAWDOWN (17%).

Focus your improvement on:
1. Improving win rate by tightening entry conditions
2. Reducing max drawdown with better exit logic
3. Adding ATR-based position sizing to reduce volatility
4. Adding a regime filter (trending vs ranging) to avoid false signals
5. Improving the hold/exit logic to capture more of each move

Rules:
- Output EXACTLY ONE complete Python file as JSON: {{"hypothesis": "...", "rationale": "...", "code": "...complete file..."}}
- Class MUST be CryptoDayTradingStrategy with generate_signals(df) returning pd.Series of +1/-1/0
- NO external TA libraries (only pandas, numpy)
- NO look-ahead bias (always shift signals by 1 bar)
- Keep the strategy simple and elegant"""

CRITIC_SYSTEM = """You are a rigorous quant analyst reviewing a crypto trading strategy.
Check for:
1. Look-ahead bias (using future data in generate_signals)
2. Missing imports
3. Syntax errors
4. Violations: class must be CryptoDayTradingStrategy, method must be generate_signals(df)
5. The signal shift(1) must be present to avoid look-ahead

Output JSON: {{"approved": true/false, "issues": ["..."], "fixed_code": "...or null"}}"""


def generator_propose(current_code: str, results_history: str, iteration: int, best_score: float) -> dict:
    system = GENERATOR_SYSTEM.format(best_score=best_score)
    prompt = f"""Iteration #{iteration} | Best score: {best_score:.1f}/100

Current crypto_strategy.py:
```python
{current_code[:3000]}
```

Recent experiment history:
{results_history}

Propose a SINGLE focused improvement. Think step by step about what would most improve the score."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"hypothesis": "parse error", "rationale": "", "code": ""}


def critic_review(proposed_code: str, hypothesis: str) -> dict:
    prompt = f"""Review this crypto trading strategy:
Hypothesis: {hypothesis}

Code:
```python
{proposed_code[:3000]}
```"""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"approved": True, "issues": [], "fixed_code": None}


# ─── Main Loop ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("AutoResearch Loop — Crypto Day-Trading Algorithm")
    logger.info(f"  Target: 30%+ weekly returns, Sharpe > 3, Win Rate > 55%")
    logger.info(f"  Strategy: {STRATEGY_FILE}")
    logger.info("=" * 60)

    iteration = 0

    while True:
        iteration += 1
        best_score = get_current_best()

        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration} | Best Score: {best_score:.2f}/100")
        logger.info(f"{'='*60}")

        current_code = read_file(STRATEGY_FILE)
        results_history = read_results_history()

        # Step 1: Generator proposes
        logger.info("Step 1/4 — Generator proposing change...")
        try:
            proposal = generator_propose(current_code, results_history, iteration, best_score)
            hypothesis = proposal.get("hypothesis", "unknown")
            proposed_code = proposal.get("code", "")
            logger.info(f"  Hypothesis: {hypothesis[:120]}")
        except Exception as e:
            logger.error(f"Generator failed: {e}")
            time.sleep(10)
            continue

        if not proposed_code or len(proposed_code) < 200:
            logger.warning("Generator returned insufficient code — skipping")
            continue

        # Step 2: Critic reviews
        logger.info("Step 2/4 — Critic reviewing...")
        try:
            review = critic_review(proposed_code, hypothesis)
            approved = review.get("approved", True)
            issues = review.get("issues", [])
            fixed_code = review.get("fixed_code")

            if issues:
                logger.info(f"  Issues found: {issues[:2]}")

            if not approved and not fixed_code:
                logger.warning("  Critic rejected without fix — skipping")
                continue

            final_code = fixed_code if (fixed_code and len(fixed_code) > 200) else proposed_code
        except Exception as e:
            logger.warning(f"Critic failed: {e}")
            final_code = proposed_code

        # Step 3: Apply change and commit
        logger.info("Step 3/4 — Applying change...")
        STRATEGY_FILE.write_text(final_code)

        committed = git_commit(f"autoresearch iter{iteration}: {hypothesis[:60]}")
        if not committed:
            logger.warning("  Nothing to commit (code unchanged) — skipping")
            continue

        # Step 4: Evaluate
        logger.info("Step 4/4 — Running evaluation...")
        metric, eval_output = run_eval()
        logger.info(f"  Metric: {metric:.4f}")

        if metric < 0:
            logger.warning("  Eval crashed — reverting via ratchet crash handler")
            run_ratchet(0.0, f"CRASH iter{iteration}: {hypothesis[:50]}")
            # Restore the last good strategy
            subprocess.run(["git", "checkout", "HEAD", "--", str(STRATEGY_FILE.relative_to(REPO_ROOT))],
                           cwd=str(REPO_ROOT), capture_output=True)
            continue

        # Ratchet: keep or revert
        description = f"iter{iteration}: {hypothesis[:80]}"
        ratchet_output = run_ratchet(metric, description)
        first_line = ratchet_output.strip().split("\n")[0]
        logger.info(f"  Ratchet: {first_line}")

        if metric > best_score:
            logger.info(f"  *** NEW BEST: {metric:.2f}/100 (was {best_score:.2f}) ***")
            # Log key metrics from eval output
            for line in eval_output.split("\n"):
                if any(k in line for k in ["Weekly Return", "Sharpe", "Win Rate", "Drawdown"]):
                    logger.info(f"    {line.strip()}")

        time.sleep(3)


if __name__ == "__main__":
    main()
