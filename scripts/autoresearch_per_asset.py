"""
AutoResearch Loop — Per-Asset Strategy Optimisation
====================================================
Runs multi-LLM propose → implement → evaluate → ratchet cycles
targeting portfolio Sharpe > 1.5 on 4-year OOS data.
Target file: tradingagents/research/per_asset_router.py
Eval: scripts/eval_per_asset_oos.py (1.8s runtime)
"""
import os
import sys
import subprocess
import json
import time
import re
import datetime

sys.path.insert(0, '/home/ubuntu/TradingAgents')

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_FILE = 'tradingagents/research/per_asset_router.py'
EVAL_CMD = 'python3 scripts/eval_per_asset_oos.py'
MAX_ITERATIONS = 30
METRIC_REGEX = r'METRIC:\s*([-\d.]+)'
REPO_DIR = '/home/ubuntu/TradingAgents'

# ── LLM Helper ────────────────────────────────────────────────────────────────
def call_llm(prompt: str, max_tokens: int = 8000) -> str:
    """Call the Forge LLM API."""
    import requests
    api_url = os.environ.get('BUILT_IN_FORGE_API_URL', 'https://forge.manus.ai')
    api_key = os.environ.get('BUILT_IN_FORGE_API_KEY', '')
    
    if not api_key:
        # Get from running webdev server process
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
            "temperature": 0.7,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']

# ── Git Helpers ───────────────────────────────────────────────────────────────
def git_commit(msg: str):
    subprocess.run(['git', 'add', TARGET_FILE], cwd=REPO_DIR, capture_output=True)
    subprocess.run(['git', 'commit', '-m', msg], cwd=REPO_DIR, capture_output=True)

def git_reset_hard():
    subprocess.run(['git', 'checkout', 'HEAD', '--', TARGET_FILE], cwd=REPO_DIR, capture_output=True)

# ── Eval ──────────────────────────────────────────────────────────────────────
def run_eval() -> float:
    """Run eval and extract metric."""
    try:
        result = subprocess.run(
            EVAL_CMD.split(),
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        match = re.search(METRIC_REGEX, output)
        if match:
            return float(match.group(1))
        print(f"  [EVAL] No metric found in output:\n{output[-500:]}")
        return -999.0
    except subprocess.TimeoutExpired:
        print("  [EVAL] Timeout!")
        return -999.0
    except Exception as e:
        print(f"  [EVAL] Error: {e}")
        return -999.0

# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    os.chdir(REPO_DIR)
    
    # Get baseline metric
    print("=" * 60)
    print("AUTORESEARCH: Per-Asset Strategy Optimisation")
    print(f"Target: Portfolio Sharpe >= 1.5")
    print(f"Start time: {datetime.datetime.now().isoformat()}")
    print("=" * 60)
    
    best_metric = run_eval()
    print(f"\nBASELINE METRIC: {best_metric:.6f}")
    
    # Read program.md for context
    with open('program.md') as f:
        program = f.read()
    
    results_log = []
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS} | Best: {best_metric:.4f} | Target: 1.50")
        print(f"{'─' * 60}")
        
        # Read current strategy
        with open(TARGET_FILE) as f:
            current_code = f.read()
        
        # Build prompt
        prompt = f"""You are an expert quantitative researcher optimising a crypto trading strategy.

PROGRAM SPEC:
{program}

CURRENT METRIC: {best_metric:.6f} (Portfolio Sharpe Ratio, annualised)
TARGET: >= 1.5

CURRENT RESULTS LOG (last 5):
{json.dumps(results_log[-5:], indent=2) if results_log else "No previous iterations"}

CURRENT CODE ({TARGET_FILE}):
```python
{current_code}
```

YOUR TASK:
Propose ONE specific, targeted improvement to increase the portfolio Sharpe ratio.
The eval harness imports BTC_CONFIG and ETH_CONFIG dicts and uses their parameters.
Focus on modifying ONLY the config dict values — that is what moves the metric.

Current breakdown:
- BTC ATR Expansion: Sharpe ~0.82, 46.5% win rate, 623 trades (strong)
- ETH Donchian Momentum: Sharpe ~0.39, 44.5% win rate, 339 trades (weak link)

ETH is the weak link. High-impact parameter changes to try:
1. Set vol_atr_max to None (removes restrictive volatility filter)
2. Reduce hurst_min (try 0.40-0.44)
3. Reduce adx_trend (try 18-20)
4. Reduce vol_mult (try 1.3-1.7)
5. Shorten donchian_period (try 18-22)
6. Adjust stop_mult/tp_mult ratio (try 2.0/6.0 or 1.5/5.0)
7. Reduce max_hold_bars (try 24-36)
8. Add atr_donchian_factor (try 0.3-0.7) for adaptive period

For BTC, try: expansion_mult 2.5-2.8, vol_mult 1.2-1.4, max_hold 18-24

OUTPUT FORMAT:
First explain your reasoning in 2-3 sentences, then output the COMPLETE updated file wrapped in:
```python
<full file content>
```

IMPORTANT: Output the ENTIRE file content, not just the changed parts. The file must be self-contained and importable."""

        # Call LLM
        print(f"  [LLM] Requesting improvement proposal...")
        try:
            response = call_llm(prompt)
        except Exception as e:
            print(f"  [LLM] Error: {e}")
            time.sleep(5)
            continue
        
        # Extract code
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if not code_match:
            print(f"  [LLM] No code block found in response")
            continue
        
        new_code = code_match.group(1).strip()
        
        # Validate basic structure
        if 'class PerAssetRouter' not in new_code:
            print(f"  [LLM] Missing PerAssetRouter class — skipping")
            continue
        if 'def generate_signals' not in new_code:
            print(f"  [LLM] Missing generate_signals method — skipping")
            continue
        
        # Write new code
        with open(TARGET_FILE, 'w') as f:
            f.write(new_code)
        
        # Evaluate
        print(f"  [EVAL] Running evaluation...")
        new_metric = run_eval()
        print(f"  [EVAL] New metric: {new_metric:.6f} (best: {best_metric:.6f})")
        
        # Ratchet decision
        if new_metric > best_metric:
            improvement = new_metric - best_metric
            print(f"  [KEEP] ✓ Improvement: +{improvement:.6f}")
            git_commit(f"autoresearch iter {iteration}: Sharpe {best_metric:.4f} → {new_metric:.4f}")
            best_metric = new_metric
            results_log.append({
                'iter': iteration,
                'metric': new_metric,
                'decision': 'KEEP',
                'improvement': improvement,
            })
        else:
            print(f"  [DISCARD] ✗ No improvement (got {new_metric:.4f} vs best {best_metric:.4f})")
            git_reset_hard()
            results_log.append({
                'iter': iteration,
                'metric': new_metric,
                'decision': 'DISCARD',
            })
        
        # Check if target reached
        if best_metric >= 1.5:
            print(f"\n{'=' * 60}")
            print(f"TARGET REACHED! Portfolio Sharpe = {best_metric:.4f} >= 1.5")
            print(f"{'=' * 60}")
            break
        
        # Brief pause between iterations
        time.sleep(2)
    
    # Final summary
    print(f"\n{'=' * 60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"Final best metric: {best_metric:.6f}")
    print(f"Total iterations: {len(results_log)}")
    print(f"Kept: {sum(1 for r in results_log if r['decision'] == 'KEEP')}")
    print(f"Discarded: {sum(1 for r in results_log if r['decision'] == 'DISCARD')}")
    print(f"End time: {datetime.datetime.now().isoformat()}")
    print(f"{'=' * 60}")
    
    # Write results
    with open('results_per_asset.json', 'w') as f:
        json.dump({'best_metric': best_metric, 'iterations': results_log}, f, indent=2)

if __name__ == '__main__':
    main()
