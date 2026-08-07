"""Generate "race" artifacts: Random, CSP, DQN, and PPO taking turns on the
*same physically shared* board -- not four independent episodes matched by
seed (see `evaluation.shared_race`'s module docstring for why that
distinction actually matters, both for fairness and for keeping DQN/PPO's
observations in-distribution).

Q-Learning is intentionally not included here, for the same reason
`generate_replays.py` already excludes it: `evaluate_agents.py` trains its
Q-table in memory and never persists it, so there is no checkpoint to replay
against outside that one training run.

Each of `--episodes` seeds produces one `race_{seed}.json`: one shared
`Minesweeper` instance, four agents taking turns revealing cells on it in a
fixed round-robin order, until the board is collectively cleared or every
agent has been eliminated.

Run with:
    python -m evaluation.generate_race --episodes 6
    python -m evaluation.generate_race --episodes 6 --dqn-experiment-id exp_E_combined
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch

from environment.minesweeper import Minesweeper
from evaluation.agent_loading import AGENT_DISPLAY_NAMES, build_agent
from evaluation.replay import build_shared_race
from evaluation.shared_race import AgentEntry, simulate_shared_race

ROWS = 5
COLS = 5
NUM_MINES = 5
SEED = 42

# Fixed round-robin turn order.
AGENT_ORDER = ["random", "csp", "dqn", "ppo"]

# Deliberately *not* the same value as the game's mine-placement seed for
# this race -- see `generate_race.py`'s git history / rl/tests for the
# empirical finding: `random.Random(seed)`'s very first draw correlates when
# two instances are freshly seeded identically, and both RandomAgent's action
# choice and CSPAgent's no-information fallback are single
# `self._rng.choice(...)` calls against the same candidate list
# `Minesweeper._place_mines()` samples from -- seeding both with the
# identical race_seed made the first mover's first move land on a mine on
# every single race tested (100% vs. the expected ~20%). Still relevant here:
# whichever agent moves first in the round robin faces exactly that fully-
# hidden-board scenario.
_AGENT_SEED_OFFSET = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate shared-board turn-based race artifacts across Random/CSP/DQN/PPO.")
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

    for i in range(args.episodes):
        race_number = i + 1
        race_seed = args.seed + i

        agents: List[AgentEntry] = []
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
            if on_episode_start is not None:
                on_episode_start()
            agents.append((AGENT_DISPLAY_NAMES[agent_slug], action_fn, reasoning_fn))

        game = Minesweeper(rows=ROWS, cols=COLS, num_mines=NUM_MINES, seed=race_seed)
        result = simulate_shared_race(game, agents)

        race = build_shared_race(
            race_number=race_number,
            seed=race_seed,
            board_size=f"{ROWS}x{COLS}",
            mines=NUM_MINES,
            generated_at=datetime.now(timezone.utc).isoformat(),
            turn_order=[AGENT_DISPLAY_NAMES[slug] for slug in AGENT_ORDER],
            result=result,
        )

        path = output_dir / f"race_{race_number}.json"
        path.write_text(json.dumps(race, indent=2))

        outcome = f"WON in {result['total_turns']} turns (survivors: {', '.join(result['surviving_agents']) or 'none'})" if result["won"] else f"no winner after {result['total_turns']} turns -- everyone eliminated"
        print(f"  wrote {path.name} -- {outcome}; eliminated: {result['eliminated_agents']}")

    print(f"Generated {args.episodes} races -> {output_dir}")


if __name__ == "__main__":
    main()
