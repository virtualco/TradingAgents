#!/usr/bin/env python3
"""
AutoResearch v2 — Two-Phase Strategy Optimisation
====================================================
Phase 1 (LLM-driven): Propose structural strategy changes (new indicators,
         filters, entry/exit logic). Config-only JSON output.
Phase 2 (Bayesian):   Tune parameters with Optuna TPE sampler.

Modes:
  --mode llm       LLM-only with config-only JSON output (Phase 1)
  --mode bayesian  Optuna TPE parameter tuning (Phase 2)
  --mode grid      Systematic grid search
  --mode full      Phase 1 → Phase 2 pipeline
  --mode validate  Walk-forward validation of current params

Usage:
  python3 scripts/autoresearch_v2.py --mode bayesian --trials 200
  python3 scripts/autoresearch_v2.py --mode llm --iterations 10
  python3 scripts/autoresearch_v2.py --mode full --iterations 5 --trials 200
  python3 scripts/autoresearch_v2.py --mode validate
"""
import os
import sys
import subprocess
import json
import time
import re
import datetime
import argparse
import copy

sys.path.insert(0, '/home/ubuntu/TradingAgents')

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_FILE = 'tradingagents/research/per_asset_router.py'
EVAL_CMD = 'python3 scripts/eval_per_asset_oos.py'
METRIC_REGEX = r'METRIC:\s*([-\d.]+)'
REPO_DIR = '/home/ubuntu/TradingAgents'

# Research direction prompts for diversity (rotated each iteration)
RESEARCH_DIRECTIONS = [
    {
        "focus": "entry_filters",
        "instruction": "Focus on ENTRY FILTER parameters. The entry filters (ADX, Hurst, volume multiplier) "
                       "control how many signals pass through. Relaxing them increases trade count but may "
                       "reduce quality. Tightening them reduces noise but may miss opportunities. "
                       "Try adjusting adx_min, adx_trend, hurst_min, vol_mult, vol_atr_max.",
    },
    {
        "focus": "risk_management",
        "instruction": "Focus on RISK MANAGEMENT parameters. Stop-loss and take-profit ratios determine "
                       "the R:R profile. Tighter stops (lower stop_mult) cut losses faster but increase "
                       "whipsaw. Wider TPs (higher tp_mult) capture bigger moves but reduce win rate. "
                       "Try adjusting stop_mult, tp_mult, max_hold_bars.",
    },
    {
        "focus": "breakout_sensitivity",
        "instruction": "Focus on BREAKOUT SENSITIVITY parameters. The Donchian period and ATR expansion "
                       "multiplier control how sensitive the strategy is to breakouts. Shorter periods "
                       "catch breakouts earlier but generate more false signals. "
                       "Try adjusting donchian_period, expansion_mult, atr_period.",
    },
    {
        "focus": "eth_improvement",
        "instruction": "Focus EXCLUSIVELY on ETH parameters — it is the weaker asset. "
                       "The ETH Donchian Momentum strategy has lower Sharpe than BTC. "
                       "Consider: removing vol_atr_max (set to None), adding atr_donchian_factor "
                       "for adaptive periods, or fundamentally changing the filter combination.",
    },
    {
        "focus": "btc_improvement",
        "instruction": "Focus EXCLUSIVELY on BTC parameters. The BTC ATR Expansion strategy is strong "
                       "but may benefit from: different atr_period, tighter/wider expansion_mult, "
                       "or different hold time. Small improvements here compound with ETH.",
    },
    {
        "focus": "combined_rr",
        "instruction": "Focus on COMBINED risk-reward optimisation across BOTH assets. "
                       "The key insight from grid search is that tighter stops + wider TPs improve Sharpe. "
                       "Try pushing this further: stop_mult < 1.5, tp_mult > 6.0. "
                       "Also consider if max_hold_bars should differ between assets.",
    },
]


# ── LLM Helper ────────────────────────────────────────────────────────────────
def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 4000) -> str:
    """Call the Forge LLM API."""
    import requests
    api_url = os.environ.get('BUILT_IN_FORGE_API_URL', 'https://forge.manus.ai')
    api_key = os.environ.get('BUILT_IN_FORGE_API_KEY', '')

    if not api_key:
        try:
            pids = subprocess.check_output(['pgrep', '-f', 'tsx watch'], text=True).strip().split('\n')
            if pids and pids[0]:
                env_data = open(f'/proc/{pids[0]}/environ').read()
                for pair in env_data.split('\x00'):
                    if pair.startswith('BUILT_IN_FORGE_API_KEY='):
                        api_key = pair.split('=', 1)[1]
                    elif pair.startswith('BUILT_IN_FORGE_API_URL='):
                        api_url = pair.split('=', 1)[1]
        except Exception:
            pass

    resp = requests.post(
        f"{api_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


# ── Git Helpers ───────────────────────────────────────────────────────────────
def git_commit(msg: str):
    subprocess.run(['git', 'add', TARGET_FILE], cwd=REPO_DIR, capture_output=True)
    subprocess.run(['git', 'commit', '-m', msg], cwd=REPO_DIR, capture_output=True)

def git_reset():
    subprocess.run(['git', 'checkout', 'HEAD', '--', TARGET_FILE],
                   cwd=REPO_DIR, capture_output=True)


# ── Eval ──────────────────────────────────────────────────────────────────────
def run_eval() -> float:
    try:
        result = subprocess.run(
            EVAL_CMD.split(), cwd=REPO_DIR,
            capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        match = re.search(METRIC_REGEX, output)
        if match:
            return float(match.group(1))
        print(f"  [EVAL] No metric found:\n{output[-300:]}")
        return -999.0
    except subprocess.TimeoutExpired:
        print("  [EVAL] Timeout!")
        return -999.0
    except Exception as e:
        print(f"  [EVAL] Error: {e}")
        return -999.0


# ── Config Reader/Writer ─────────────────────────────────────────────────────
def read_configs() -> tuple:
    """Read BTC_CONFIG and ETH_CONFIG from per_asset_router.py."""
    import importlib
    mod_name = 'tradingagents.research.per_asset_router'
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)
    return dict(mod.BTC_CONFIG), dict(mod.ETH_CONFIG)


def write_configs(btc_cfg: dict, eth_cfg: dict):
    """Write updated configs back to per_asset_router.py by patching the dicts."""
    with open(os.path.join(REPO_DIR, TARGET_FILE)) as f:
        content = f.read()

    # Build BTC_CONFIG string
    btc_lines = ["BTC_CONFIG = {"]
    for k, v in btc_cfg.items():
        if isinstance(v, str):
            btc_lines.append(f"    '{k}': '{v}',")
        elif isinstance(v, float):
            btc_lines.append(f"    '{k}': {v},")
        elif v is None:
            btc_lines.append(f"    '{k}': None,")
        else:
            btc_lines.append(f"    '{k}': {v},")
    btc_lines.append("}")
    btc_block = "\n".join(btc_lines)

    # Build ETH_CONFIG string
    eth_lines = ["ETH_CONFIG = {"]
    for k, v in eth_cfg.items():
        if isinstance(v, str):
            eth_lines.append(f"    '{k}': '{v}',")
        elif isinstance(v, float):
            eth_lines.append(f"    '{k}': {v},")
        elif v is None:
            eth_lines.append(f"    '{k}': None,")
        else:
            eth_lines.append(f"    '{k}': {v},")
    eth_lines.append("}")
    eth_block = "\n".join(eth_lines)

    # Replace BTC_CONFIG block
    btc_pattern = re.compile(r'BTC_CONFIG\s*=\s*\{[^}]+\}', re.DOTALL)
    content = btc_pattern.sub(btc_block, content, count=1)

    # Replace ETH_CONFIG block
    eth_pattern = re.compile(r'ETH_CONFIG\s*=\s*\{[^}]+\}', re.DOTALL)
    content = eth_pattern.sub(eth_block, content, count=1)

    with open(os.path.join(REPO_DIR, TARGET_FILE), 'w') as f:
        f.write(content)


# ══════════════════════════════════════════════════════════════════════════════
# MODE: LLM (Phase 1 — Config-Only JSON Output)
# ══════════════════════════════════════════════════════════════════════════════
def run_llm_mode(iterations: int = 10, temperature: float = 0.8):
    """LLM proposes config changes as JSON. No full-file rewrites."""
    os.chdir(REPO_DIR)
    best_metric = run_eval()
    print(f"\nBASELINE: {best_metric:.6f}")

    btc_cfg, eth_cfg = read_configs()
    results_log = []

    for i in range(1, iterations + 1):
        direction = RESEARCH_DIRECTIONS[(i - 1) % len(RESEARCH_DIRECTIONS)]
        temp = temperature + 0.1 * ((i - 1) % 3)  # Vary temperature: 0.8, 0.9, 1.0

        print(f"\n{'─'*60}")
        print(f"ITER {i}/{iterations} | Best: {best_metric:.4f} | Focus: {direction['focus']} | Temp: {temp:.1f}")
        print(f"{'─'*60}")

        prompt = f"""You are an expert quantitative researcher optimising a crypto trading strategy.

CURRENT METRIC: {best_metric:.6f} (Portfolio Sharpe Ratio, annualised)
TARGET: >= 1.5

CURRENT BTC_CONFIG: {json.dumps(btc_cfg, indent=2)}
CURRENT ETH_CONFIG: {json.dumps(eth_cfg, indent=2)}

PREVIOUS RESULTS (last 5):
{json.dumps(results_log[-5:], indent=2) if results_log else "No previous iterations"}

RESEARCH DIRECTION: {direction['instruction']}

OUTPUT FORMAT — respond with ONLY a JSON object, no other text:
{{
  "reasoning": "2-3 sentences explaining your hypothesis",
  "btc_config": {{ ... only keys you want to change ... }},
  "eth_config": {{ ... only keys you want to change ... }}
}}

RULES:
- Change at most 2-3 parameters per proposal
- Use null for None values (e.g., "vol_atr_max": null)
- Only include keys you want to change, not the full config
- Valid BTC keys: atr_period, expansion_mult, vol_mult, max_hold_bars, stop_mult, tp_mult
- Valid ETH keys: donchian_period, adx_min, adx_trend, vol_mult, hurst_min, vol_atr_max, max_hold_bars, stop_mult, tp_mult, atr_donchian_factor
- Parameter ranges: stop_mult [0.5-4.0], tp_mult [2.0-12.0], vol_mult [0.5-3.0], expansion_mult [1.5-5.0], donchian_period [10-40], max_hold_bars [4-120]"""

        print(f"  [LLM] Requesting config change...")
        try:
            response = call_llm(prompt, temperature=temp, max_tokens=1000)
        except Exception as e:
            print(f"  [LLM] Error: {e}")
            continue

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown wrapping)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                print(f"  [LLM] No JSON found in response")
                continue
            proposal = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"  [LLM] JSON parse error: {e}")
            continue

        reasoning = proposal.get('reasoning', 'No reasoning provided')
        btc_changes = proposal.get('btc_config', {})
        eth_changes = proposal.get('eth_config', {})
        print(f"  [LLM] Reasoning: {reasoning[:120]}...")
        print(f"  [LLM] BTC changes: {btc_changes}")
        print(f"  [LLM] ETH changes: {eth_changes}")

        # Apply changes
        new_btc = copy.deepcopy(btc_cfg)
        new_eth = copy.deepcopy(eth_cfg)
        for k, v in btc_changes.items():
            if k in ('atr_period', 'expansion_mult', 'vol_mult', 'max_hold_bars', 'stop_mult', 'tp_mult'):
                new_btc[k] = v
        for k, v in eth_changes.items():
            if k in ('donchian_period', 'adx_min', 'adx_trend', 'vol_mult', 'hurst_min',
                      'vol_atr_max', 'max_hold_bars', 'stop_mult', 'tp_mult', 'atr_donchian_factor'):
                new_eth[k] = v

        # Write and evaluate
        write_configs(new_btc, new_eth)
        print(f"  [EVAL] Running...")
        new_metric = run_eval()
        print(f"  [EVAL] Metric: {new_metric:.6f} (best: {best_metric:.6f})")

        if new_metric > best_metric:
            improvement = new_metric - best_metric
            print(f"  [KEEP] ✓ +{improvement:.6f}")
            git_commit(f"autoresearch-v2 iter {i}: Sharpe {best_metric:.4f} → {new_metric:.4f} ({direction['focus']})")
            btc_cfg = new_btc
            eth_cfg = new_eth
            best_metric = new_metric
            results_log.append({
                'iter': i, 'metric': new_metric, 'decision': 'KEEP',
                'focus': direction['focus'], 'btc_changes': btc_changes,
                'eth_changes': eth_changes, 'reasoning': reasoning,
            })
        else:
            print(f"  [DISCARD] ✗")
            git_reset()
            results_log.append({
                'iter': i, 'metric': new_metric, 'decision': 'DISCARD',
                'focus': direction['focus'], 'btc_changes': btc_changes,
                'eth_changes': eth_changes, 'reasoning': reasoning,
            })

        if best_metric >= 1.5:
            print(f"\n  TARGET REACHED: {best_metric:.4f}")
            break

    print(f"\n{'='*60}")
    print(f"LLM MODE COMPLETE | Best: {best_metric:.6f}")
    print(f"Kept: {sum(1 for r in results_log if r['decision'] == 'KEEP')}/{len(results_log)}")
    print(f"{'='*60}")

    with open(os.path.join(REPO_DIR, 'results_llm.json'), 'w') as f:
        json.dump({'best_metric': best_metric, 'iterations': results_log}, f, indent=2)

    return best_metric


# ══════════════════════════════════════════════════════════════════════════════
# MODE: BAYESIAN (Phase 2 — Optuna TPE)
# ══════════════════════════════════════════════════════════════════════════════
def run_bayesian_mode(trials: int = 200):
    """Run Optuna Bayesian optimisation."""
    print(f"\nRunning Bayesian optimisation ({trials} trials)...")
    result = subprocess.run(
        ['python3', '-u', 'scripts/optimise_bayesian.py',
         '--trials', str(trials), '--asset', 'portfolio'],
        cwd=REPO_DIR, capture_output=True, text=True, timeout=600)
    print(result.stdout[-1000:] if result.stdout else "No output")
    if result.stderr:
        print(f"STDERR: {result.stderr[-500:]}")

    # Extract metric
    match = re.search(METRIC_REGEX, result.stdout)
    return float(match.group(1)) if match else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MODE: GRID
# ══════════════════════════════════════════════════════════════════════════════
def run_grid_mode():
    """Run systematic grid search."""
    print("\nRunning grid search...")
    result = subprocess.run(
        ['python3', '-u', 'scripts/grid_search_fine.py'],
        cwd=REPO_DIR, capture_output=True, text=True, timeout=1200)
    print(result.stdout[-1500:] if result.stdout else "No output")
    match = re.search(METRIC_REGEX, result.stdout)
    return float(match.group(1)) if match else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MODE: VALIDATE (Walk-Forward)
# ══════════════════════════════════════════════════════════════════════════════
def run_validate_mode():
    """Run walk-forward validation on current params."""
    print("\nRunning walk-forward validation...")
    result = subprocess.run(
        ['python3', '-u', 'scripts/walk_forward.py', '--validate-only'],
        cwd=REPO_DIR, capture_output=True, text=True, timeout=120)
    print(result.stdout if result.stdout else "No output")
    match = re.search(METRIC_REGEX, result.stdout)
    return float(match.group(1)) if match else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MODE: FULL (Phase 1 → Phase 2 Pipeline)
# ══════════════════════════════════════════════════════════════════════════════
def run_full_mode(iterations: int = 5, trials: int = 200):
    """Two-phase pipeline: LLM proposes structure, Bayesian tunes params."""
    print("=" * 60)
    print("AUTORESEARCH v2 — FULL PIPELINE")
    print(f"Phase 1: {iterations} LLM iterations (structural changes)")
    print(f"Phase 2: {trials} Bayesian trials (parameter tuning)")
    print("=" * 60)

    # Phase 1: LLM structural changes
    print(f"\n{'━'*60}")
    print("PHASE 1: LLM-Driven Structural Changes")
    print(f"{'━'*60}")
    llm_metric = run_llm_mode(iterations=iterations, temperature=0.8)

    # Phase 2: Bayesian parameter tuning
    print(f"\n{'━'*60}")
    print("PHASE 2: Bayesian Parameter Tuning")
    print(f"{'━'*60}")
    bayesian_metric = run_bayesian_mode(trials=trials)

    # Validation
    print(f"\n{'━'*60}")
    print("VALIDATION: Walk-Forward")
    print(f"{'━'*60}")
    wf_metric = run_validate_mode()

    print(f"\n{'='*60}")
    print("FULL PIPELINE COMPLETE")
    print(f"  Phase 1 (LLM):      {llm_metric:.4f}")
    print(f"  Phase 2 (Bayesian):  {bayesian_metric:.4f}")
    print(f"  Walk-Forward Avg:    {wf_metric:.4f}")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='AutoResearch v2 — Two-Phase Optimisation')
    parser.add_argument('--mode', choices=['llm', 'bayesian', 'grid', 'validate', 'full'],
                        default='full', help='Optimisation mode')
    parser.add_argument('--iterations', type=int, default=10,
                        help='LLM iterations (for llm/full modes)')
    parser.add_argument('--trials', type=int, default=200,
                        help='Bayesian trials (for bayesian/full modes)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Base LLM temperature (for llm mode)')
    args = parser.parse_args()

    print(f"AutoResearch v2 | Mode: {args.mode} | {datetime.datetime.now().isoformat()}")

    if args.mode == 'llm':
        run_llm_mode(iterations=args.iterations, temperature=args.temperature)
    elif args.mode == 'bayesian':
        run_bayesian_mode(trials=args.trials)
    elif args.mode == 'grid':
        run_grid_mode()
    elif args.mode == 'validate':
        run_validate_mode()
    elif args.mode == 'full':
        run_full_mode(iterations=args.iterations, trials=args.trials)


if __name__ == '__main__':
    main()
