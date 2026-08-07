"""Evaluate one agent at one (level, density) board configuration, writing a
small, uniform result file -- deliberately *not* shaped like a training
"experiment" (no history, no hyperparameters), since most of these runs
involve no training at all: CSP/Random need no checkpoint at any size, and
DQN/PPO are trained once per level (at that level's "standard" density, see
board_configs.py) then *evaluated* at every density on that same level using
the one checkpoint -- a generalization test, not a matched-condition result.

Checkpoint resolution for DQN/PPO reuses `agent_loading.build_agent`'s
existing `experiment_id` mechanism unchanged (see `checkpoint_experiment_id`):
  - At the "beginner" level, no `experiment_id` is passed at all, so
    `build_agent` falls back to its default checkpoint path -- the same
    `results/checkpoints_evaluate_agents/best_model.pt` /
    `checkpoints_ppo_evaluate_agents/best_policy.pt` every other script in
    this project already uses. Beginner needs no new training.
  - At "intermediate"/"expert", `experiment_id` is set to
    `"levels/{level}/standard"`, a relative path `build_agent` joins onto
    `results_dir` -- so training that level's standard-density run with
    `dqn_experiment.py --output-dir results/levels/{level}/standard ...`
    is all that's needed for this script to find it; nothing here needs to
    change once that training exists.

Run with:
    python -m evaluation.evaluate_board_config --agent random --level beginner --density sparse
    python -m evaluation.evaluate_board_config --agent csp --level expert --density dense
    python -m evaluation.evaluate_board_config --agent dqn --level beginner --density dense
    python -m evaluation.evaluate_board_config --agent dqn --level intermediate --density sparse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from board_configs import resolve
from environment.minesweeper_env import MinesweeperEnv
from evaluation.agent_loading import AGENT_DISPLAY_NAMES, build_agent
from evaluation.metrics import evaluate_agent

EVAL_EPISODES = 200
SEED = 42


def checkpoint_experiment_id(level: str) -> Optional[str]:
    """Which `experiment_id` to pass to `agent_loading.build_agent` for a
    DQN/PPO checkpoint at this level -- `None` at "beginner" (use the
    project's original default checkpoint), otherwise that level's own
    standard-density training run."""
    return None if level == "beginner" else f"levels/{level}/standard"


def run_evaluation(
    agent_kind: str,
    level: str,
    density: str,
    *,
    eval_episodes: int = EVAL_EPISODES,
    seed: int = SEED,
    results_dir: Path = Path("results"),
) -> Dict[str, Any]:
    """Evaluate `agent_kind` at `(level, density)` and return the result dict
    (see module docstring for shape) -- no file I/O, so this is directly
    testable and directly the JSON serialized by `main()`."""
    rows, cols, mines = resolve(level, density)
    experiment_id = checkpoint_experiment_id(level)

    _agent, action_fn, _reasoning_fn, on_episode_start = build_agent(
        agent_kind, seed, experiment_id, results_dir, rows=rows, cols=cols, num_mines=mines
    )

    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=mines, seed=seed)
    eval_results = evaluate_agent(env, action_fn, num_episodes=eval_episodes, on_episode_start=on_episode_start)

    return {
        "agent": AGENT_DISPLAY_NAMES[agent_kind],
        "level": level,
        "density": density,
        "rows": rows,
        "cols": cols,
        "mines": mines,
        "eval_episodes": eval_episodes,
        "win_rate": eval_results["win_rate"],
        "avg_episode_length": eval_results["avg_episode_length"],
        "avg_reward": eval_results["avg_reward"],
        "failures": eval_results["failures"],
        "checkpoint_source": experiment_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one agent at one board level/density.")
    parser.add_argument("--agent", required=True, choices=["random", "csp", "dqn", "ppo"])
    parser.add_argument("--level", required=True, choices=["beginner", "intermediate", "expert"])
    parser.add_argument("--density", required=True, choices=["sparse", "standard", "dense"])
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-dir", type=str, default="results", help="Where to look up level checkpoints from.")
    parser.add_argument("--output-dir", type=str, default="results/levels", help="Base directory to write {level}/{density}/{agent}_board_result.json into.")
    parser.add_argument("--torch-threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    result = run_evaluation(
        args.agent,
        args.level,
        args.density,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        results_dir=Path(args.results_dir),
    )

    output_dir = Path(args.output_dir) / args.level / args.density
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{args.agent}_board_result.json"
    path.write_text(json.dumps(result, indent=2))

    print(
        f"{result['agent']} @ {args.level}/{args.density} ({result['rows']}x{result['cols']}, {result['mines']} mines): "
        f"win_rate={result['win_rate'] * 100:.1f}% avg_length={result['avg_episode_length']:.2f} -> {path}"
    )


if __name__ == "__main__":
    main()
