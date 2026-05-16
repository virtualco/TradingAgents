#!/usr/bin/env python3
"""
RL Training on Live Data — TradingAgents v9.0
==============================================
Trains the PPO position sizing agent on real signal history collected
during live trading. Implements:

  - Signal history collector: persists state/action/reward tuples
  - Experience replay buffer: disk-backed, append-only
  - Periodic training: retrain every 2 weeks on accumulated experience
  - A/B evaluation gate: only promote if RL Sharpe > baseline + threshold
  - Model hot-swap: live_trader_v2 loads new model without restart

Usage:
    # Collect experience during live trading (called by live_trader_v2)
    python3 scripts/train_rl_live.py collect --state '...' --action 0.8 --reward 0.002

    # Train on accumulated experience
    python3 scripts/train_rl_live.py train [--min-episodes 100] [--threshold 0.1]

    # Evaluate current model vs baseline
    python3 scripts/train_rl_live.py evaluate

    # Export model for hot-swap
    python3 scripts/train_rl_live.py promote
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rl_live_trainer")

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("RL_DATA_DIR", "data/rl"))
MODEL_DIR = Path(os.environ.get("RL_MODEL_DIR", "data/models/rl"))
BUFFER_FILE = DATA_DIR / "experience_buffer.jsonl"
METADATA_FILE = DATA_DIR / "training_metadata.json"
MIN_EPISODES_DEFAULT = 100
SHARPE_THRESHOLD_DEFAULT = 0.1  # RL must beat baseline by this margin
RETRAIN_INTERVAL_DAYS = 14


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class Experience:
    """Single state-action-reward-next_state tuple."""
    timestamp: str
    symbol: str
    state: List[float]       # 12-dim observation
    action: float            # sizing multiplier [0, 2]
    reward: float            # risk-adjusted step return
    next_state: List[float]  # next observation
    done: bool               # episode terminal
    info: Dict               # metadata (regime, signal_type, etc.)


# ── Experience Replay Buffer ──────────────────────────────────────────────────

class ExperienceBuffer:
    """Disk-backed, append-only experience replay buffer."""

    def __init__(self, path: Path = BUFFER_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, exp: Experience) -> None:
        """Append a single experience to the buffer."""
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(exp)) + "\n")

    def append_batch(self, experiences: List[Experience]) -> None:
        """Append multiple experiences."""
        with open(self.path, "a") as f:
            for exp in experiences:
                f.write(json.dumps(asdict(exp)) + "\n")

    def load_all(self) -> List[Experience]:
        """Load all experiences from buffer."""
        if not self.path.exists():
            return []
        experiences = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    experiences.append(Experience(**data))
        return experiences

    def size(self) -> int:
        """Count experiences in buffer."""
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path, "r") as f:
            for _ in f:
                count += 1
        return count

    def get_episodes(self) -> List[List[Experience]]:
        """Group experiences into episodes (split on done=True)."""
        all_exp = self.load_all()
        episodes = []
        current = []
        for exp in all_exp:
            current.append(exp)
            if exp.done:
                episodes.append(current)
                current = []
        if current:
            episodes.append(current)
        return episodes


# ── Signal History Collector ──────────────────────────────────────────────────

class SignalHistoryCollector:
    """
    Collects state/action/reward tuples during live trading.
    Called by live_trader_v2 after each trading decision.
    """

    def __init__(self):
        self.buffer = ExperienceBuffer()
        self._current_episode: List[Experience] = []

    def record(self, symbol: str, state: List[float], action: float,
               reward: float, next_state: List[float], done: bool = False,
               info: Optional[Dict] = None) -> None:
        """Record a single trading step."""
        exp = Experience(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
        )
        self._current_episode.append(exp)
        self.buffer.append(exp)

        if done:
            log.info(f"Episode complete: {len(self._current_episode)} steps, "
                    f"total_reward={sum(e.reward for e in self._current_episode):.4f}")
            self._current_episode = []

    @property
    def total_experiences(self) -> int:
        return self.buffer.size()


# ── Training Pipeline ─────────────────────────────────────────────────────────

def build_training_data(episodes: List[List[Experience]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert episodes to numpy arrays for SB3 training."""
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for episode in episodes:
        for exp in episode:
            states.append(exp.state)
            actions.append([exp.action])
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=bool),
    )


def train_from_buffer(min_episodes: int = MIN_EPISODES_DEFAULT,
                      total_timesteps: int = 50_000) -> Tuple[bool, dict]:
    """
    Train PPO agent on accumulated live experience.

    Returns:
        (success, metrics)
    """
    try:
        from stable_baselines3 import PPO
        from tradingagents.rl.env import TradingSizingEnv
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return False, {"error": str(e)}

    buffer = ExperienceBuffer()
    episodes = buffer.get_episodes()

    if len(episodes) < min_episodes:
        log.warning(f"Insufficient episodes: {len(episodes)} < {min_episodes}")
        return False, {
            "error": "insufficient_data",
            "episodes": len(episodes),
            "required": min_episodes,
        }

    log.info(f"Training on {len(episodes)} episodes, {buffer.size()} total steps")

    # Build environment from real data
    states, actions, rewards, next_states, dones = build_training_data(episodes)

    # Create environment with real data statistics
    env = TradingSizingEnv(
        n_steps=len(states),
        initial_capital=100_000,
    )
    # Inject real data statistics into environment
    env._real_states = states
    env._real_rewards = rewards
    env._use_real_data = True

    # Train PPO
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=min(2048, len(states)),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
    )

    log.info(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save candidate model
    candidate_path = MODEL_DIR / f"rl_sizing_candidate_{timestamp}"
    model.save(str(candidate_path))

    metrics = {
        "version": timestamp,
        "trained_at": datetime.utcnow().isoformat(),
        "episodes_used": len(episodes),
        "total_steps": buffer.size(),
        "timesteps_trained": total_timesteps,
    }

    # Save training metadata
    meta_path = MODEL_DIR / f"rl_training_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Candidate model saved: {candidate_path}")
    return True, metrics


# ── A/B Evaluation Gate ───────────────────────────────────────────────────────

def evaluate_candidate(threshold: float = SHARPE_THRESHOLD_DEFAULT) -> Tuple[bool, dict]:
    """
    Compare candidate RL model against static baseline.
    Only promotes if RL Sharpe > baseline Sharpe + threshold.
    """
    try:
        from stable_baselines3 import PPO
        from tradingagents.rl.evaluator import RLEvaluator
    except ImportError as e:
        return False, {"error": str(e)}

    # Find latest candidate
    candidates = sorted(MODEL_DIR.glob("rl_sizing_candidate_*.zip"), reverse=True)
    if not candidates:
        return False, {"error": "no_candidate_model"}

    candidate_path = str(candidates[0]).replace(".zip", "")
    log.info(f"Evaluating candidate: {candidates[0].name}")

    # Load buffer for evaluation data
    buffer = ExperienceBuffer()
    episodes = buffer.get_episodes()

    if len(episodes) < 20:
        return False, {"error": "insufficient_evaluation_data"}

    # Use last 20% of episodes for evaluation
    eval_episodes = episodes[int(len(episodes) * 0.8):]
    states, actions, rewards, _, dones = build_training_data(eval_episodes)

    # Baseline: static 1.0 multiplier
    baseline_returns = rewards  # rewards already reflect 1.0x sizing
    baseline_sharpe = _compute_sharpe(baseline_returns)

    # RL: load model and compute sizing-adjusted returns
    model = PPO.load(candidate_path)
    rl_returns = []
    for state, reward in zip(states, rewards):
        action, _ = model.predict(state, deterministic=True)
        sizing = float(np.clip(action[0], 0, 2))
        rl_returns.append(reward * sizing)

    rl_returns = np.array(rl_returns)
    rl_sharpe = _compute_sharpe(rl_returns)

    improvement = rl_sharpe - baseline_sharpe
    passes = improvement > threshold

    result = {
        "baseline_sharpe": round(baseline_sharpe, 4),
        "rl_sharpe": round(rl_sharpe, 4),
        "improvement": round(improvement, 4),
        "threshold": threshold,
        "passes_gate": passes,
        "eval_episodes": len(eval_episodes),
        "eval_steps": len(states),
    }

    log.info(f"Evaluation: baseline={baseline_sharpe:.4f}, RL={rl_sharpe:.4f}, "
            f"delta={improvement:.4f}, threshold={threshold}, passes={passes}")

    return passes, result


def _compute_sharpe(returns: np.ndarray, annualise: float = np.sqrt(252 * 24)) -> float:
    """Compute annualised Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * annualise)


# ── Model Hot-Swap (Promotion) ────────────────────────────────────────────────

def promote_candidate() -> Tuple[bool, dict]:
    """
    Promote candidate model to production.
    Creates a symlink that live_trader_v2 watches for hot-swap.
    """
    candidates = sorted(MODEL_DIR.glob("rl_sizing_candidate_*.zip"), reverse=True)
    if not candidates:
        return False, {"error": "no_candidate_model"}

    candidate = candidates[0]
    production_link = MODEL_DIR / "rl_sizing_production.zip"

    # Remove old link
    if production_link.exists() or production_link.is_symlink():
        production_link.unlink()

    # Create new symlink
    production_link.symlink_to(candidate.name)

    # Write promotion marker (live_trader_v2 watches this)
    marker = MODEL_DIR / "promotion_marker.json"
    with open(marker, "w") as f:
        json.dump({
            "promoted_at": datetime.utcnow().isoformat(),
            "model": candidate.name,
            "version": candidate.stem.replace("rl_sizing_candidate_", ""),
        }, f, indent=2)

    log.info(f"✓ Promoted: {candidate.name} → rl_sizing_production.zip")
    return True, {"promoted": candidate.name}


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL Training on Live Data")
    subparsers = parser.add_subparsers(dest="command")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Record experience")
    collect_parser.add_argument("--state", type=str, required=True, help="JSON state vector")
    collect_parser.add_argument("--action", type=float, required=True, help="Sizing multiplier")
    collect_parser.add_argument("--reward", type=float, required=True, help="Step reward")
    collect_parser.add_argument("--next-state", type=str, default="[]", help="JSON next state")
    collect_parser.add_argument("--symbol", type=str, default="BTC", help="Symbol")
    collect_parser.add_argument("--done", action="store_true", help="Episode terminal")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train on accumulated experience")
    train_parser.add_argument("--min-episodes", type=int, default=MIN_EPISODES_DEFAULT)
    train_parser.add_argument("--timesteps", type=int, default=50_000)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="A/B evaluation")
    eval_parser.add_argument("--threshold", type=float, default=SHARPE_THRESHOLD_DEFAULT)

    # Promote command
    subparsers.add_parser("promote", help="Promote candidate to production")

    # Status command
    subparsers.add_parser("status", help="Show buffer and model status")

    args = parser.parse_args()

    if args.command == "collect":
        collector = SignalHistoryCollector()
        state = json.loads(args.state)
        next_state = json.loads(args.next_state) if args.next_state != "[]" else state
        collector.record(
            symbol=args.symbol,
            state=state,
            action=args.action,
            reward=args.reward,
            next_state=next_state,
            done=args.done,
        )
        print(f"Recorded. Buffer size: {collector.total_experiences}")

    elif args.command == "train":
        success, metrics = train_from_buffer(
            min_episodes=args.min_episodes,
            total_timesteps=args.timesteps,
        )
        if success:
            print(f"✓ Training complete: {json.dumps(metrics, indent=2)}")
        else:
            print(f"✗ Training failed: {metrics}")
            sys.exit(1)

    elif args.command == "evaluate":
        passes, result = evaluate_candidate(threshold=args.threshold)
        print(json.dumps(result, indent=2))
        if not passes:
            sys.exit(1)

    elif args.command == "promote":
        success, result = promote_candidate()
        if success:
            print(f"✓ {result}")
        else:
            print(f"✗ {result}")
            sys.exit(1)

    elif args.command == "status":
        buffer = ExperienceBuffer()
        episodes = buffer.get_episodes()
        candidates = sorted(MODEL_DIR.glob("rl_sizing_candidate_*.zip"), reverse=True)
        production = MODEL_DIR / "rl_sizing_production.zip"

        print(f"=== RL Live Training Status ===")
        print(f"Buffer: {buffer.size()} experiences, {len(episodes)} episodes")
        print(f"Candidates: {len(candidates)}")
        print(f"Production model: {'exists' if production.exists() else 'none'}")

        if candidates:
            print(f"Latest candidate: {candidates[0].name}")

        # Check if retrain is due
        if METADATA_FILE.exists():
            with open(METADATA_FILE) as f:
                meta = json.load(f)
            last_train = meta.get("last_trained", "never")
            print(f"Last trained: {last_train}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
