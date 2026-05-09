#!/usr/bin/env python3
"""
AutoResearch Loop Orchestrator — Crypto Day-Trading Algorithm
=============================================================
Runs the propose → implement → evaluate → ratchet loop autonomously.
Uses multi-LLM strategy:
  - Generator: gpt-4.1-mini (creative, proposes code changes)
  - Critic:    gemini-2.5-flash (analytical, catches bugs)
  - Evaluator: gpt-4.1-nano (fast, scores the improvement)

Runs indefinitely until manually interrupted.
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
EVAL_RUNNER = Path("/home/ubuntu/skills/autoresearch/scripts/run_eval.sh")

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

def write_file(path: Path, content: str):
    path.write_text(content)

def read_results_history() -> str:
    if not RESULTS_TSV.exists():
        return "No previous results."
    lines = RESULTS_TSV.read_text().strip().split("\n")
    return "\n".join(lines[-20:])  # Last 20 experiments

def run_eval() -> tuple[float, str]:
    """Run the evaluation harness and parse the metric. Returns (metric, raw_output)."""
    try:
        result = subprocess.run(
            ["python3", str(EVAL_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        # Parse metric
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
    """Run the ratchet script and return its output."""
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

def git_add_commit(message: str):
    """Stage and commit the strategy file."""
    subprocess.run(["git", "add", str(STRATEGY_FILE)], cwd=str(REPO_ROOT), capture_output=True)
    subprocess.run(["git", "commit", "-m", message, "--allow-empty"],
                   cwd=str(REPO_ROOT), capture_output=True)

def git_reset():
    """Reset the strategy file to last committed state."""
    subprocess.run(["git", "checkout", "HEAD", "--", str(STRATEGY_FILE.relative_to(REPO_ROOT))],
                   cwd=str(REPO_ROOT), capture_output=True)

# ─── Multi-LLM Pipeline ──────────────────────────────────────────────────────

GENERATOR_SYSTEM = """You are a world-class quantitative researcher and algorithmic trading engineer with 20+ years of experience in crypto markets.
Your task is to propose a SINGLE, focused improvement to a crypto day-trading algorithm targeting 30% weekly returns.

The algorithm is in Python and uses pandas/numpy for vectorised operations.
The evaluation metric is a composite score (0-100) based on:
  - 40% weekly return (target: 30% per week)
  - 40% Sharpe ratio (target: 3.0+)
  - 20% win rate
  - Penalty for drawdown > 50%

Rules:
- Propose EXACTLY ONE change
- Output valid Python code for the COMPLETE replacement of crypto_strategy.py
- Include ALL necessary imports
- The class MUST be named CryptoDayTradingStrategy with a generate_signals(df) method
- generate_signals must return a pd.Series of +1 (long), -1 (short), 0 (flat)
- Think about: multi-indicator confluence, regime detection, volatility-adaptive thresholds, momentum filters, mean-reversion on overbought/oversold, volume confirmation
- NEVER use future data (no look-ahead bias)
- Output JSON: {"hypothesis": "...", "rationale": "...", "code": "...complete python file..."}"""

CRITIC_SYSTEM = """You are a rigorous quantitative analyst reviewing a proposed crypto trading algorithm change.
Your job is to catch:
1. Look-ahead bias (using future data)
2. Survivorship bias
3. Overfitting to specific market conditions
4. Missing imports or syntax errors
5. Violations of the constraints (must have generate_signals returning pd.Series of +1/-1/0)
6. Unrealistic assumptions

Output JSON: {"approved": true/false, "issues": ["..."], "fixed_code": "...or null if no fix needed..."}"""

EVALUATOR_SYSTEM = """You are a trading strategy evaluator. Given backtest results, determine if the improvement is meaningful.
Output JSON: {"verdict": "keep/discard", "reasoning": "..."}"""

def generator_propose(current_code: str, results_history: str, iteration: int) -> dict:
    """Use gpt-4.1-mini to propose a code change."""
    prompt = f"""Iteration #{iteration}

Current crypto_strategy.py:
```python
{current_code}
```

Previous experiment results (last 20):
{results_history}

The baseline strategy is losing money badly (-38% weekly). We need a completely different approach.

Consider these proven crypto day-trading approaches:
1. RSI + MACD confluence with volume filter
2. Bollinger Band mean-reversion with momentum confirmation
3. VWAP deviation scalping
4. EMA crossover with ATR-based position sizing
5. Dual-timeframe momentum (fast signal + slow trend filter)
6. Volatility breakout with Keltner channels
7. Adaptive RSI with regime detection (trending vs ranging)

Propose the best strategy for 30% weekly returns with controlled drawdown.
Remember: the backtest uses 1-hour candles for BTC-USD and ETH-USD over 60 days."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"hypothesis": "unknown", "rationale": "parse error", "code": current_code}

def critic_review(proposed_code: str, hypothesis: str) -> dict:
    """Use gemini-2.5-flash to review the proposed change."""
    prompt = f"""Review this proposed crypto trading strategy:

Hypothesis: {hypothesis}

Proposed code:
```python
{proposed_code}
```

Check for look-ahead bias, missing imports, syntax errors, and constraint violations."""

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"approved": True, "issues": [], "fixed_code": None}

def evaluator_score(eval_output: str, metric: float, hypothesis: str) -> dict:
    """Use gpt-4.1-nano to evaluate the result."""
    prompt = f"""Hypothesis: {hypothesis}
Composite Score: {metric:.2f}/100
Eval output (last 30 lines):
{chr(10).join(eval_output.strip().split(chr(10))[-30:])}

Should we keep or discard this change?"""

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"verdict": "keep" if metric > 0 else "discard", "reasoning": "auto"}

# ─── Main Loop ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("AutoResearch Loop — Crypto Day-Trading Algorithm")
    logger.info(f"  Target: 30% weekly returns")
    logger.info(f"  Strategy file: {STRATEGY_FILE}")
    logger.info("=" * 60)

    iteration = 0
    best_score = 0.0

    while True:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration} | Best Score So Far: {best_score:.2f}/100")
        logger.info(f"{'='*60}")

        # Step 1: Read current state
        current_code = read_file(STRATEGY_FILE)
        results_history = read_results_history()

        # Step 2: Generator proposes a change
        logger.info("Step 1/4 — Generator proposing change...")
        try:
            proposal = generator_propose(current_code, results_history, iteration)
            hypothesis = proposal.get("hypothesis", "unknown")
            rationale = proposal.get("rationale", "")
            proposed_code = proposal.get("code", "")
            logger.info(f"  Hypothesis: {hypothesis[:100]}")
        except Exception as e:
            logger.error(f"Generator failed: {e}")
            time.sleep(5)
            continue

        if not proposed_code or len(proposed_code) < 100:
            logger.warning("Generator returned empty/short code — skipping")
            continue

        # Step 3: Critic reviews the change
        logger.info("Step 2/4 — Critic reviewing change...")
        try:
            review = critic_review(proposed_code, hypothesis)
            approved = review.get("approved", True)
            issues = review.get("issues", [])
            fixed_code = review.get("fixed_code")

            if issues:
                logger.info(f"  Critic issues: {issues[:3]}")

            if not approved and not fixed_code:
                logger.warning("  Critic rejected change without fix — skipping")
                continue

            final_code = fixed_code if fixed_code and len(fixed_code) > 100 else proposed_code
        except Exception as e:
            logger.warning(f"Critic failed: {e} — proceeding with generator output")
            final_code = proposed_code

        # Step 4: Apply the change
        logger.info("Step 3/4 — Applying change and running evaluation...")
        write_file(STRATEGY_FILE, final_code)

        # Step 5: Run evaluation
        metric, eval_output = run_eval()
        logger.info(f"  Evaluation metric: {metric:.4f}")

        if metric < 0:
            logger.warning("  Evaluation crashed — reverting")
            git_reset()
            run_ratchet(0.0, f"CRASH: {hypothesis[:60]}")
            continue

        # Step 6: Evaluator scores the result
        try:
            eval_verdict = evaluator_score(eval_output, metric, hypothesis)
            verdict = eval_verdict.get("verdict", "keep" if metric > best_score else "discard")
            reasoning = eval_verdict.get("reasoning", "")
            logger.info(f"  Evaluator verdict: {verdict} — {reasoning[:80]}")
        except Exception as e:
            logger.warning(f"Evaluator failed: {e}")
            verdict = "keep" if metric > best_score else "discard"

        # Step 7: Ratchet — commit or revert
        description = f"iter{iteration}: {hypothesis[:80]}"
        ratchet_output = run_ratchet(metric, description)
        logger.info(f"  Ratchet: {ratchet_output.strip().split(chr(10))[0]}")

        if metric > best_score:
            best_score = metric
            logger.info(f"  *** NEW BEST: {best_score:.2f}/100 ***")

        # Brief pause between iterations to avoid rate limits
        time.sleep(3)


if __name__ == "__main__":
    main()
