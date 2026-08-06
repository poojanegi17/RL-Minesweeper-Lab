"""Generate replay artifacts for interactive playback in the frontend.

Plays episodes with a chosen agent and records each one's full timeline via
`evaluation.replay.ReplayRecorder`, writing one JSON file per episode to
`rl/results/replays/{agent}_episode_{n}.json`.

DQN and PPO load a trained checkpoint (defaulting to the same checkpoints
`evaluate_agents.py` deploys, `results/checkpoints_evaluate_agents/best_model.pt`
and `results/checkpoints_ppo_evaluate_agents/best_policy.pt`; pass
`--experiment-id` to load a specific experiment's checkpoint instead, e.g.
`--experiment-id exp_E_combined`). CSP and Random need no trained state.
Q-Learning is intentionally not supported here: it has no persisted
checkpoint anywhere in this project (`evaluate_agents.py` trains its Q-table
in-memory and never saves it), so there is nothing to replay against outside
that one training run.

Re-running this script for the same agent does not overwrite previously
generated replays -- episode numbers continue from whatever's already in the
output directory, so replay files accumulate across multiple invocations.

Security: replay files never contain mine positions -- see
`evaluation.replay`'s module docstring for why, and the README's Replay
Visualization section for the full picture.

Run with:
    python -m evaluation.generate_replays --agent random --episodes 10
    python -m evaluation.generate_replays --agent csp --episodes 10
    python -m evaluation.generate_replays --agent dqn --episodes 10
    python -m evaluation.generate_replays --agent dqn --episodes 5 --experiment-id exp_E_combined
    python -m evaluation.generate_replays --agent ppo --episodes 10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch

from agents.csp_solver import CSPAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from environment.minesweeper_env import MinesweeperEnv
from evaluation.replay import ReasoningFn, ReplayRecorder, build_replay, csp_reasoning, dqn_reasoning, ppo_reasoning

ROWS = 5
COLS = 5
NUM_MINES = 5
SEED = 42

AGENT_DISPLAY_NAMES = {"csp": "CSP", "dqn": "DQN", "ppo": "PPO", "random": "Random"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for replay generation."""
    parser = argparse.ArgumentParser(description="Generate replay artifacts for interactive playback.")
    parser.add_argument("--agent", required=True, choices=["csp", "dqn", "ppo", "random"], help="Which agent to record.")
    parser.add_argument("--episodes", type=int, required=True, help="Number of replay episodes to generate.")
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="For dqn/ppo: load this experiment's checkpoint (e.g. exp_E_combined) instead of the default.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Base seed; episode i uses seed + i, so each replay's board differs.")
    parser.add_argument("--output-dir", type=str, default="results/replays", help="Where to write replay JSON files.")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Where to look up experiment checkpoints from (used with --experiment-id)."
    )
    parser.add_argument("--torch-threads", type=int, default=None, help="Cap CPU threads PyTorch uses.")
    return parser.parse_args()


def _default_checkpoint_path(agent: str, results_dir: Path) -> Path:
    if agent == "dqn":
        return results_dir / "checkpoints_evaluate_agents" / "best_model.pt"
    return results_dir / "checkpoints_ppo_evaluate_agents" / "best_policy.pt"


def _experiment_checkpoint_path(agent: str, experiment_id: str, results_dir: Path) -> Path:
    experiment_dir = results_dir / experiment_id
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"No experiment directory found at {experiment_dir}")

    checkpoint_dirs = sorted(experiment_dir.glob("checkpoints_*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints_* directory found under {experiment_dir}")
    checkpoint_dir = checkpoint_dirs[0]

    best_name = "best_model.pt" if agent == "dqn" else "best_policy.pt"
    final_name = "final_model.pt" if agent == "dqn" else "final_policy.pt"
    for name in (best_name, final_name):
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {best_name} or {final_name} found under {checkpoint_dir}")


def _load_dqn_agent(checkpoint_path: Path, seed: int) -> DQNAgent:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = DQNAgent(rows=raw["rows"], cols=raw["cols"], network_size=raw.get("network_size", "default"), seed=seed)
    agent.load_checkpoint(checkpoint_path)
    return agent


def _load_ppo_agent(checkpoint_path: Path, seed: int) -> PPOAgent:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = PPOAgent(rows=raw["rows"], cols=raw["cols"], seed=seed)
    agent.load_checkpoint(checkpoint_path)
    return agent


def _build_agent(
    agent_kind: str, seed: int, experiment_id: Optional[str], results_dir: Path
) -> Tuple[Any, Callable[[Any], int], Optional[ReasoningFn], Optional[Callable[[], None]]]:
    """Construct the agent plus its action_fn/reasoning_fn/on_episode_start, mirroring evaluate_agents.py's
    per-agent wiring against the same generic recorder -- see evaluation.replay's module docstring."""
    if agent_kind == "random":
        agent = RandomAgent(seed=seed)
        return agent, agent.select_action, None, None

    if agent_kind == "csp":
        agent = CSPAgent(rows=ROWS, cols=COLS, num_mines=NUM_MINES, seed=seed)
        return agent, agent.choose_action, (lambda board, action: csp_reasoning(agent, board, action)), agent.reset

    checkpoint_path = (
        _experiment_checkpoint_path(agent_kind, experiment_id, results_dir)
        if experiment_id is not None
        else _default_checkpoint_path(agent_kind, results_dir)
    )
    if agent_kind == "dqn":
        agent = _load_dqn_agent(checkpoint_path, seed)
        return (
            agent,
            lambda obs: agent.select_action(obs, explore=False),
            (lambda board, action: dqn_reasoning(agent, board, action)),
            None,
        )

    agent = _load_ppo_agent(checkpoint_path, seed)
    return (
        agent,
        lambda obs: agent.select_action(obs, explore=False),
        (lambda board, action: ppo_reasoning(agent, board, action)),
        None,
    )


def _next_episode_number(output_dir: Path, agent_slug: str) -> int:
    """One past the highest existing `{agent_slug}_episode_N.json` found, so replays accumulate
    across multiple invocations instead of being overwritten."""
    max_existing = 0
    for path in output_dir.glob(f"{agent_slug}_episode_*.json"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            max_existing = max(max_existing, int(suffix))
    return max_existing + 1


def main() -> None:
    """Generate `--episodes` replay files for the chosen agent."""
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    agent_slug = args.agent
    agent_display_name = AGENT_DISPLAY_NAMES[agent_slug]
    experiment_id_for_checkpoint = args.experiment_id if agent_slug in ("dqn", "ppo") else None

    agent, action_fn, reasoning_fn, on_episode_start = _build_agent(
        agent_slug, args.seed, experiment_id_for_checkpoint, results_dir
    )

    env = MinesweeperEnv(rows=ROWS, cols=COLS, num_mines=NUM_MINES)
    recorder = ReplayRecorder()
    episode_number = _next_episode_number(output_dir, agent_slug)

    wins = 0
    for i in range(args.episodes):
        episode = recorder.record_episode(
            env,
            action_fn,
            reasoning_fn=reasoning_fn,
            on_episode_start=on_episode_start,
            seed=args.seed + i,
        )
        replay = build_replay(
            agent_name=agent_display_name,
            experiment_id=experiment_id_for_checkpoint,
            board_size=f"{ROWS}x{COLS}",
            mines=NUM_MINES,
            seed=args.seed + i,
            episode_number=episode_number,
            episode=episode,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        path = output_dir / f"{agent_slug}_episode_{episode_number}.json"
        path.write_text(json.dumps(replay, indent=2))

        wins += int(episode["won"])
        print(f"  wrote {path.name} ({episode['steps_taken']} steps, {'WIN' if episode['won'] else 'loss'})")
        episode_number += 1

    print(f"Generated {args.episodes} replays for {agent_display_name} ({wins}/{args.episodes} wins) -> {output_dir}")


if __name__ == "__main__":
    main()
