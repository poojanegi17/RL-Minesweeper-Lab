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

import torch

from environment.minesweeper_env import MinesweeperEnv
from evaluation.agent_loading import AGENT_DISPLAY_NAMES, build_agent
from evaluation.replay import ReplayRecorder, build_replay

ROWS = 5
COLS = 5
NUM_MINES = 5
SEED = 42


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
    parser.add_argument("--rows", type=int, default=ROWS, help="Board rows (see board_configs.py for the level/density presets).")
    parser.add_argument("--cols", type=int, default=COLS, help="Board columns.")
    parser.add_argument("--mines", type=int, default=NUM_MINES, help="Mine count.")
    parser.add_argument("--output-dir", type=str, default="results/replays", help="Where to write replay JSON files.")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Where to look up experiment checkpoints from (used with --experiment-id)."
    )
    parser.add_argument("--torch-threads", type=int, default=None, help="Cap CPU threads PyTorch uses.")
    return parser.parse_args()


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

    agent, action_fn, reasoning_fn, on_episode_start = build_agent(
        agent_slug, args.seed, experiment_id_for_checkpoint, results_dir, rows=args.rows, cols=args.cols, num_mines=args.mines
    )

    env = MinesweeperEnv(rows=args.rows, cols=args.cols, num_mines=args.mines)
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
            board_size=f"{args.rows}x{args.cols}",
            mines=args.mines,
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
