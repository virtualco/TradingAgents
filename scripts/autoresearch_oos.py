"""
AutoResearch OOS Loop — Optimise CryptoDayTradingStrategy against 4-year dataset
================================================================================
Uses the full 2022-2026 Binance historical data as the evaluation target.
Runs propose → implement → evaluate → ratchet cycles using LLM.
Target: OOS WFA profitable folds > 55%, avg weekly return > 0%, Sharpe > 0.5

Usage:
    python3 scripts/autoresearch_oos.py
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] autoresearch_oos: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("autoresearch_oos")

STRATEGY_FILE = Path("tradingagents/research/crypto_strategy.py")
EVAL_SCRIPT   = "python3 scripts/validate_oos.py"
RESULTS_FILE  = Path("data/oos_validation_results.json")
RESULTS_TSV   = Path("results_oos.tsv")
MAX_ITERS     = 15
MODEL         = "gpt-4.1-mini"

client = OpenAI()


def get_current_score() -> tuple[float, dict]:
    """Run eval and extract composite score from results JSON."""
    try:
        result = subprocess.run(
            EVAL_SCRIPT.split(),
            capture_output=True, text=True, timeout=300
        )
        if RESULTS_FILE.exists():
            data = json.loads(RESULTS_FILE.read_text())
            summary = data.get("summary", {})
            wfa_weekly = summary.get("wfa_avg_weekly_pct", -99)
            wfa_sharpe = summary.get("wfa_avg_sharpe", -99)
            wfa_pf     = summary.get("wfa_profitable_folds_pct", 0)
            avg_dd     = summary.get("avg_drawdown_pct", 100)

            # Composite score: weighted combination
            # 40% profitable folds, 30% weekly return, 20% Sharpe, 10% drawdown control
            score = (
                0.40 * min(wfa_pf, 100) +
                0.30 * min(max(wfa_weekly + 5, 0) * 10, 100) +  # normalise: 0%=50pts, +5%=100pts
                0.20 * min(max(wfa_sharpe + 3, 0) * 20, 100) +  # normalise: 0=60pts, 1.5=90pts
                0.10 * max(100 - avg_dd * 2, 0)                  # normalise: 0%DD=100pts, 50%DD=0
            )
            return round(score, 4), summary
    except Exception as e:
        log.warning(f"Eval error: {e}")
    return 0.0, {}


def read_strategy() -> str:
    return STRATEGY_FILE.read_text()


def write_strategy(code: str):
    STRATEGY_FILE.write_text(code)


def propose_improvement(current_code: str, history: list, best_score: float) -> tuple[str, str]:
    """Ask LLM to propose a strategy improvement."""
    history_summary = "\n".join([f"- {h}" for h in history[-5:]]) if history else "None yet."

    prompt = f"""You are a senior quant researcher with 10+ years in crypto algorithmic trading.

TASK: Improve the Python crypto trading strategy below to achieve consistent positive returns 
across ALL market regimes (2022 bear, 2023 recovery, 2024 bull, 2025 mixed).

CURRENT BEST SCORE: {best_score:.2f}/100
SCORING: 40% profitable WFA folds, 30% avg weekly return, 20% Sharpe, 10% drawdown control

RECENT ATTEMPTS (avoid repeating these):
{history_summary}

KEY INSIGHT: The strategy must work in BEAR markets (2022) and RANGING markets, not just bull runs.
Consider:
- Mean reversion in ranging markets (Bollinger Bands, RSI extremes)
- Momentum following in trending markets (current approach)
- Dynamic regime switching between mean-reversion and momentum
- Better exit logic to lock in profits quickly in volatile regimes
- Reducing trade frequency to only take highest-conviction setups
- Using longer lookback periods for more stable signals

CURRENT STRATEGY CODE:
```python
{current_code}
```

Respond with EXACTLY this format:
HYPOTHESIS: [one sentence describing the specific change and why it should improve OOS performance]
CODE:
```python
[complete replacement Python file — must define class CryptoDayTradingStrategy with generate_signals method]
```"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.7,
    )
    content = response.choices[0].message.content

    hypothesis = ""
    code = ""
    if "HYPOTHESIS:" in content:
        hypothesis = content.split("HYPOTHESIS:")[1].split("\n")[0].strip()
    if "```python" in content:
        code = content.split("```python")[1].split("```")[0].strip()

    return hypothesis, code


def record_result(commit: str, score: float, status: str, description: str):
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tmetric\tstatus\tdescription\n")
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{score:.6f}\t{status}\t{description[:80]}\n")


def main():
    log.info("=" * 65)
    log.info("AutoResearch OOS Loop — CryptoDayTradingStrategy")
    log.info(f"  Target: OOS profitable folds > 55%, weekly > 0%, Sharpe > 0.5")
    log.info(f"  Eval: {EVAL_SCRIPT}")
    log.info("=" * 65)

    # Get baseline
    log.info("\nEstablishing baseline score...")
    best_score, best_summary = get_current_score()
    log.info(f"  Baseline score: {best_score:.2f}/100")
    log.info(f"  Summary: {best_summary}")

    best_code = read_strategy()
    record_result("baseline", best_score, "keep", "v3 ADX+regime-adaptive baseline")

    history = []

    for iteration in range(1, MAX_ITERS + 1):
        log.info(f"\n{'='*65}")
        log.info(f"ITERATION {iteration} | Best Score: {best_score:.2f}/100")
        log.info(f"{'='*65}")

        # Step 1: Propose
        log.info("Step 1/3 — Proposing improvement...")
        hypothesis, new_code = propose_improvement(best_code, history, best_score)
        log.info(f"  Hypothesis: {hypothesis[:100]}")

        if not new_code or len(new_code) < 200:
            log.warning("  LLM returned empty/short code — skipping iteration")
            history.append(f"SKIP: {hypothesis[:60]}")
            continue

        # Step 2: Apply
        log.info("Step 2/3 — Applying change...")
        original_code = read_strategy()
        write_strategy(new_code)

        # Step 3: Evaluate
        log.info("Step 3/3 — Running OOS evaluation...")
        score, summary = get_current_score()
        log.info(f"  Score: {score:.4f}")
        log.info(f"  WFA weekly: {summary.get('wfa_avg_weekly_pct', 'N/A')}% | "
                 f"Sharpe: {summary.get('wfa_avg_sharpe', 'N/A')} | "
                 f"Profitable folds: {summary.get('wfa_profitable_folds_pct', 'N/A')}%")

        # Ratchet
        if score > best_score:
            best_score = score
            best_code = new_code
            status = "keep"
            log.info(f"  *** NEW BEST: {score:.2f}/100 ***")
            history.append(f"KEEP({score:.1f}): {hypothesis[:60]}")
        else:
            write_strategy(original_code)
            status = "discard"
            log.info(f"  DISCARD: {score:.4f} <= {best_score:.4f}")
            history.append(f"DISCARD({score:.1f}): {hypothesis[:60]}")

        record_result(f"iter{iteration}", score, status, hypothesis)

        # Check if target met
        pf = summary.get("wfa_profitable_folds_pct", 0)
        ww = summary.get("wfa_avg_weekly_pct", -99)
        sh = summary.get("wfa_avg_sharpe", -99)
        if pf > 55 and ww > 0 and sh > 0.5:
            log.info("\n>>> TARGET MET — STRATEGY CLEARED FOR BYBIT TESTNET <<<")
            break

        time.sleep(2)

    log.info(f"\nFinal best score: {best_score:.2f}/100")
    log.info(f"Results saved to {RESULTS_TSV}")


if __name__ == "__main__":
    main()
