"""Generate "race" artifacts: Random, CSP, DQN, and PPO all playing the *same*
seeded board, for the frontend's side-by-side race viewer.

Q-Learning is intentionally not included here, for the same reason
`generate_replays.py` already excludes it: `evaluate_agents.py` trains its
Q-table in memory and never persists it, so there is no checkpoint to replay
against outside that one training run.

Each of `--episodes` shared seeds produces one `race_{seed}.json` bundling
all four agents' full timelines. The shared seed is what makes this a fair
comparison rather than four independent episodes: `MinesweeperEnv.reset(seed=X)`
rebuilds the game from scratch, and mine placement depends only on the seed
(see `environment.minesweeper.Minesweeper`), so every agent recorded on the
same seed sees the identical mine layout.

Run with:
    python -m evaluation.generate_race --episodes 6
    python -m evaluation.generate_race --episodes 6 --dqn-experiment-id exp_E_combined
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from environment.minesweeper_env import MinesweeperEnv
from evaluation.agent_loading import AGENT_DISPLAY_NAMES, build_agent
from evaluation.replay import ReplayRecorder, build_race

ROWS = 5
COLS = 5
NUM_MINES = 5
SEED = 42

# Fixed order so every race bundle's `agents` dict is consistently ordered
# (JSON preserves insertion order) -- not load-bearing for correctness, just
# a nicer diff/read experience.
AGENT_ORDER = ["random", "csp", "dqn", "ppo"]

# Deliberately *not* the same value as the env's mine-placement seed for this
# race: `random.Random(seed)`'s very first draw correlates when two instances
# are freshly seeded identically, and both RandomAgent's action choice and
# CSPAgent's no-information fallback are single `self._rng.choice(...)` calls
# against the same 25-cell candidate list `Minesweeper._place_mines()` samples
# from -- seeding both with the identical race_seed made the agent's first
# move land on a mine on every single race tested (confirmed empirically
# across 20+ seeds: 100% mine rate vs. the expected ~20%). Offsetting the
# agent's own seed breaks that correlation while staying fully deterministic.
_AGENT_SEED_OFFSET = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate shared-seed race artifacts across Random/CSP/DQN/PPO.")
    parser.add_argument("--episodes", type=int, required=True, help="Number of race boards to generate.")
    parser.add_argument("--seed", type=int, default=SEED, help="Base seed; race i uses seed + i.")
    parser.add_argument("--dqn-experiment-id", type=str, default=None, help="Load this experiment's DQN checkpoint instead of the default.")
    parser.add_argument("--ppo-experiment-id", type=str, default=None, help="Load this experiment's PPO checkpoint instead of the default.")
    parser.add_argument("--output-dir", type=str, default="results/races", help="Where to write race JSON files.")
    parser.add_argument("--results-dir", type=str, default="results", help="Where to look up experiment checkpoints from.")
    parser.add_argument("--torch-threads", type=int, default=None, help="Cap CPU threads PyTorch uses.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    experiment_ids = {
        "random": None,
        "csp": None,
        "dqn": args.dqn_experiment_id,
        "ppo": args.ppo_experiment_id,
    }

    env = MinesweeperEnv(rows=ROWS, cols=COLS, num_mines=NUM_MINES)
    recorder = ReplayRecorder()

    for i in range(args.episodes):
        race_seed = args.seed + i
        agent_episodes = {}

        for agent_slug in AGENT_ORDER:
            agent, action_fn, reasoning_fn, on_episode_start = build_agent(
                agent_slug,
                race_seed + _AGENT_SEED_OFFSET,
                experiment_ids[agent_slug],
                results_dir,
                rows=ROWS,
                cols=COLS,
                num_mines=NUM_MINES,
            )
            episode = recorder.record_episode(
                env, action_fn, reasoning_fn=reasoning_fn, on_episode_start=on_episode_start, seed=race_seed
            )
            agent_episodes[AGENT_DISPLAY_NAMES[agent_slug]] = (experiment_ids[agent_slug], episode)

        race = build_race(
            seed=race_seed,
            board_size=f"{ROWS}x{COLS}",
            mines=NUM_MINES,
            generated_at=datetime.now(timezone.utc).isoformat(),
            agent_episodes=agent_episodes,
        )

        path = output_dir / f"race_{race_seed}.json"
        path.write_text(json.dumps(race, indent=2))

        summary = ", ".join(
            f"{name}: {'WIN' if ep[1]['won'] else 'loss'} ({ep[1]['steps_taken']} steps)" for name, ep in agent_episodes.items()
        )
        print(f"  wrote {path.name} -- {summary}")

    print(f"Generated {args.episodes} races -> {output_dir}")


if __name__ == "__main__":
    main()
