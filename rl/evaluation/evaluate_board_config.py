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
    this project already uses. That default was trained under the v1
    environment, so pass `--checkpoint-experiment` whenever scoring a newly
    trained beginner run; without it a v1 checkpoint is silently evaluated on
    v2 boards and reported as a v2 result.
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

from agents.q_learning_agent import QLearningAgent
from board_configs import resolve
from environment.minesweeper_env import MinesweeperEnv
from evaluation.agent_loading import AGENT_DISPLAY_NAMES, build_agent
from evaluation.metrics import evaluate_agent
from evaluation.reevaluate_checkpoints import wilson_interval

EVAL_EPISODES = 2000
SEED = 42

# Board-generation provenance. `first_click_safe` and `guarantee_solvable`
# change the *board distribution*, so results produced under different settings
# are not comparable -- not even for the reward-blind Random and CSP baselines.
# Every result file records which environment it was measured under, and
# `env_version` is derived from the settings rather than passed in, so a run
# cannot be mislabelled by a stale flag.
#
#   "v1" -- mines placed before the first click, layouts unfiltered. Every
#           result committed under `results_public/levels/` is v1.
#   "v2" -- any first-click safety and/or no-guess generation in effect.
ENV_VERSION_LEGACY = "v1"
ENV_VERSION_SAFE_START = "v2"


def env_version(first_click_safe: str, guarantee_solvable: bool) -> str:
    """Derive the results-schema version from the board-generation settings."""
    if first_click_safe == "none" and not guarantee_solvable:
        return ENV_VERSION_LEGACY
    return ENV_VERSION_SAFE_START

# Q-Learning trains here rather than loading a checkpoint (see `run_evaluation`).
# Matches `evaluate_agents.py`'s `Q_TRAIN_EPISODES` so the figure stays
# comparable to the project's original 5x5 Q-Learning result.
Q_TRAIN_EPISODES = 20000

# `agent_loading.AGENT_DISPLAY_NAMES` covers only the agents `build_agent`
# constructs; Q-Learning is handled directly in `run_evaluation`, so its display
# name is added here rather than there.
DISPLAY_NAMES = {**AGENT_DISPLAY_NAMES, "q_learning": "Q-Learning"}


def checkpoint_experiment_id(
    level: str, agent_kind: Optional[str] = None, results_dir: Optional[Path] = None
) -> Optional[str]:
    """Which `experiment_id` to pass to `agent_loading.build_agent` for a
    DQN/PPO checkpoint at this level.

    `None` at "beginner" (use the project's original default checkpoint).
    Above that, two layouts exist and both are supported:

    - `{agent}_{level}_A_baseline/` -- where this project's per-level training
      runs actually landed, and what every committed board result at
      intermediate/expert was produced against. Preferred when it exists, so
      this script reproduces the committed numbers rather than drifting from
      them.
    - `levels/{level}/standard/` -- the convention this module's docstring
      describes, kept as the fallback so training output placed there is still
      found. Also what CSP/Random resolve to, harmlessly: neither loads a
      checkpoint at all, so the value is only recorded for provenance.

    `agent_kind`/`results_dir` are optional purely so the documented default
    remains callable without them.
    """
    if level == "beginner":
        return None
    if agent_kind in ("dqn", "ppo") and results_dir is not None:
        per_level = f"{agent_kind}_{level}_A_baseline"
        if (results_dir / per_level).is_dir():
            return per_level
    return f"levels/{level}/standard"


def run_evaluation(
    agent_kind: str,
    level: str,
    density: str,
    *,
    eval_episodes: int = EVAL_EPISODES,
    seed: int = SEED,
    results_dir: Path = Path("results"),
    q_train_episodes: int = Q_TRAIN_EPISODES,
    first_click_safe: str = "none",
    guarantee_solvable: bool = False,
    checkpoint_experiment: Optional[str] = None,
    checkpoint_file: str = "auto",
    transfer_board: bool = False,
) -> Dict[str, Any]:
    """Evaluate `agent_kind` at `(level, density)` and return the result dict
    (see module docstring for shape) -- no file I/O, so this is directly
    testable and directly the JSON serialized by `main()`.

    `first_click_safe`/`guarantee_solvable` are passed straight to
    `MinesweeperEnv` and recorded in the result under `env`. They default to
    the legacy behaviour, so calling this without them reproduces the committed
    v1 numbers exactly.

    `checkpoint_experiment` overrides where a DQN/PPO checkpoint is loaded from,
    as a directory name under `results_dir`. Without it, `checkpoint_experiment_id`
    resolves the path automatically -- and at the "beginner" level that resolution
    returns `None`, meaning `build_agent` falls back to this project's *original*
    default checkpoint. That default was trained under v1, so evaluating it on v2
    boards would silently report transfer from a stale checkpoint as if it were a
    v2 result. Pass this explicitly whenever a newly trained run is being scored.

    `checkpoint_file` picks which of that run's two saved checkpoints to score.
    "auto" follows what the run recorded deploying, so this figure describes the
    same weights the run's published win rate does -- see
    `agent_loading.experiment_checkpoint_path`.

    Note there is deliberately no `reward_scale` parameter. That setting exists
    only to keep TD targets inside `smooth_l1_loss`'s quadratic regime during
    *training*; applying it at evaluation would rescale `avg_reward` without
    changing any policy, breaking comparability with every existing result for
    no benefit. This mirrors the existing rule that shaped-reward runs are
    always evaluated under the default reward.
    """
    rows, cols, mines = resolve(level, density)
    env_options = {
        "first_click_safe": first_click_safe,
        "guarantee_solvable": guarantee_solvable,
    }

    if agent_kind == "q_learning":
        # The only agent here that trains inside this script: its Q-table is
        # keyed by exact board pattern, so a table learned at one density
        # transfers to none other and there is no checkpoint to reuse. Matches
        # evaluate_agents.py's budget so the figure stays comparable to the
        # project's original Q-Learning result.
        #
        # Training gets its *own* environment. Sharing one would advance the
        # env's RNG by `q_train_episodes` resets before evaluation started, so
        # Q-Learning would be scored on a different board sequence than every
        # other agent at this cell -- exactly the confound that made the
        # project's original 200-episode figures non-comparable.
        # Trains on the same board distribution it is scored on -- a baseline
        # measured on boards it never saw in training would understate it.
        train_env = MinesweeperEnv(rows=rows, cols=cols, num_mines=mines, seed=seed, **env_options)
        agent = QLearningAgent(rows=rows, cols=cols, seed=seed)
        agent.train(train_env, episodes=q_train_episodes)
        action_fn, on_episode_start = agent.select_action, None
        experiment_id = None
        trained_episodes: Optional[int] = q_train_episodes
    else:
        experiment_id = (
            checkpoint_experiment
            if checkpoint_experiment is not None
            else checkpoint_experiment_id(level, agent_kind, results_dir)
        )
        _agent, action_fn, _reasoning_fn, on_episode_start = build_agent(
            agent_kind,
            seed,
            experiment_id,
            results_dir,
            rows=rows,
            cols=cols,
            num_mines=mines,
            checkpoint_file=checkpoint_file,
            transfer_board=transfer_board,
        )
        trained_episodes = None

    # Constructed after any training, and never reused for it, so every agent
    # faces the identical `eval_episodes` boards at this (level, density).
    eval_env = MinesweeperEnv(rows=rows, cols=cols, num_mines=mines, seed=seed, **env_options)
    eval_results = evaluate_agent(eval_env, action_fn, num_episodes=eval_episodes, on_episode_start=on_episode_start)

    result = {
        "agent": DISPLAY_NAMES[agent_kind],
        "level": level,
        "density": density,
        "rows": rows,
        "cols": cols,
        "mines": mines,
        "env_version": env_version(first_click_safe, guarantee_solvable),
        "env": env_options,
        "eval_episodes": eval_episodes,
        "win_rate": eval_results["win_rate"],
        "win_rate_ci95": [round(v, 3) for v in wilson_interval(eval_results["wins"], eval_episodes)],
        "avg_episode_length": eval_results["avg_episode_length"],
        "avg_reward": eval_results["avg_reward"],
        "failures": eval_results["failures"],
        "checkpoint_source": experiment_id,
    }
    if trained_episodes is not None:
        result["train_episodes"] = trained_episodes
    if transfer_board:
        # Recorded because this number means something different from every
        # other result file in this tree: the weights never trained at this
        # board size. Without the flag a zero-shot figure sits alongside
        # matched-condition ones and reads as though it were one.
        # The board it *was* trained at is recoverable from `checkpoint_source`'s
        # own summary, so it is not duplicated here where it could drift.
        result["board_transfer"] = True
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one agent at one board level/density.")
    parser.add_argument("--agent", required=True, choices=["random", "csp", "dqn", "ppo", "q_learning"])
    parser.add_argument("--level", required=True, choices=["beginner", "intermediate", "expert"])
    parser.add_argument("--density", required=True, choices=["sparse", "standard", "dense"])
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument(
        "--q-train-episodes",
        type=int,
        default=Q_TRAIN_EPISODES,
        help="Episodes to train a Q-table for, when --agent q_learning (ignored otherwise).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-dir", type=str, default="results", help="Where to look up level checkpoints from.")
    parser.add_argument("--output-dir", type=str, default="results/levels", help="Base directory to write {level}/{density}/{agent}_board_result.json into.")
    parser.add_argument(
        "--first-click-safe",
        choices=["none", "cell", "area"],
        default="none",
        help="Board-generation policy for the opening move. Changes the board distribution, so "
        "results are only comparable to others measured under the same value (default: none, the "
        "setting every committed v1 result used).",
    )
    parser.add_argument(
        "--guarantee-solvable",
        action="store_true",
        help="Only generate boards clearable by deduction alone. Intended for training, not "
        "benchmarking -- CSP scores 100%% on these by construction.",
    )
    parser.add_argument(
        "--checkpoint-experiment",
        type=str,
        default=None,
        help="Directory under --results-dir to load the DQN/PPO checkpoint from, overriding the "
        "automatic resolution. Required when scoring a newly trained run at the beginner level, "
        "where automatic resolution falls back to this project's original v1 checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-file",
        choices=["auto", "best", "final"],
        default="auto",
        help="Which of a run's two saved checkpoints to score. \"auto\" scores the one the run "
        "recorded deploying (`used_checkpoint`), matching what its published win rate describes.",
    )
    parser.add_argument(
        "--transfer-board",
        action="store_true",
        help="Evaluate the checkpoint at THIS level's board size even though it was trained at "
        "another -- a zero-shot generalization probe, not a matched result, and flagged as "
        "`board_transfer` in the output so it can never be read as one. Only works for "
        "conv-head presets ('fully_conv'), whose weights are board-size-independent; a "
        "Linear-head checkpoint raises instead.",
    )
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
        q_train_episodes=args.q_train_episodes,
        first_click_safe=args.first_click_safe,
        guarantee_solvable=args.guarantee_solvable,
        checkpoint_experiment=args.checkpoint_experiment,
        checkpoint_file=args.checkpoint_file,
        transfer_board=args.transfer_board,
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
