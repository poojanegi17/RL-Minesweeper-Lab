"""Regenerate `{agent}_board_result.json` from each level's best current run.

Why this exists rather than `evaluate_board_config.py`: that script resolves a
checkpoint by directory convention and prefers `best_model.pt`, but several of
this project's runs deployed *final* weights (`--no-best-checkpoint`) and are
reported on that basis. Scoring `best_model.pt` instead would publish a number
the pipeline never claimed. Here the exact deployed checkpoint is named per
entry, alongside the board distribution it belongs to, so the published grid
and the research pages can never drift apart.

One entry per (agent, level, board distribution). Each is evaluated across all
three mine densities of its level -- trained once at standard density and
scored at the other two without retraining, which is the same protocol every
other board result on the site uses.

Random, CSP and Q-Learning are not listed: they need no checkpoint and their
grids are already produced by `rebaseline_board_configs.py`.

Run with:
    python -m evaluation.refresh_board_results --dry-run
    python -m evaluation.refresh_board_results
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from board_configs import DENSITY_ORDER, resolve
from environment.minesweeper_env import MinesweeperEnv
from evaluation.metrics import evaluate_agent

EVAL_EPISODES = 2000
SEED = 42


@dataclass(frozen=True)
class Entry:
    agent: str
    level: str
    #: "none" (original boards) or "area" (first click safe).
    first_click_safe: str
    #: Run directory under `results/`, recorded as `checkpoint_source`.
    run: str
    #: The weights that run actually deployed.
    checkpoint: str


ENTRIES: List[Entry] = [
    # DQN -- Beginner
    Entry("DQN", "beginner", "none", "exp_M_fully_conv", "final_model.pt"),
    Entry("DQN", "beginner", "area", "dqn_v2_A_baseline", "best_model.pt"),
    # DQN -- Intermediate
    Entry("DQN", "intermediate", "none", "dqn_intermediate_D_carried_config", "final_model.pt"),
    Entry("DQN", "intermediate", "area", "dqn_intermediate_E_env_v2", "final_model.pt"),
    # PPO -- Beginner
    Entry("PPO", "beginner", "none", "ppo_long_B_shaped", "final_policy.pt"),
    Entry("PPO", "beginner", "area", "ppo_v2_F_shaped_matched", "final_policy.pt"),
]

TREE_FOR_POLICY = {"none": "v1", "area": "v2"}


def load_agent(entry: Entry, results_dir: Path):
    """Load the exact deployed weights for this entry."""
    matches = sorted((results_dir / entry.run).glob("checkpoints_*"))
    if not matches:
        raise FileNotFoundError(f"No checkpoints_* directory under {results_dir / entry.run}")
    path = matches[0] / entry.checkpoint
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if entry.agent == "DQN":
        agent = DQNAgent(rows=raw["rows"], cols=raw["cols"], network_size=raw.get("network_size", "default"), seed=SEED)
    else:
        agent = PPOAgent(rows=raw["rows"], cols=raw["cols"], seed=SEED)
    agent.load_checkpoint(path)
    return agent


def ci95(win_rate: float, episodes: int) -> List[float]:
    half = 1.96 * math.sqrt(win_rate * (1 - win_rate) / episodes)
    return [round(max(0.0, win_rate - half) * 100, 3), round(min(1.0, win_rate + half) * 100, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate board results from each level's best current run.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Where the runs and their checkpoints live.")
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=Path("results_public"),
        help="Root holding the version-scoped level trees (v1/levels, v2/levels).",
    )
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--only", type=str, default=None, help="Only entries whose run name contains this substring.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without evaluating.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = [e for e in ENTRIES if args.only is None or args.only in e.run]

    for entry in entries:
        tree = TREE_FOR_POLICY[entry.first_click_safe]
        print(f"\n{entry.agent} @ {entry.level}  [{entry.first_click_safe}]  <- {entry.run}/{entry.checkpoint}")
        if args.dry_run:
            for density in DENSITY_ORDER:
                out = args.public_dir / tree / "levels" / entry.level / density / f"{entry.agent.lower()}_board_result.json"
                print(f"    would write {out}")
            continue

        agent = load_agent(entry, args.results_dir)
        for density in DENSITY_ORDER:
            rows, cols, mines = resolve(entry.level, density)
            env = MinesweeperEnv(
                rows=rows,
                cols=cols,
                num_mines=mines,
                seed=SEED,
                # `guarantee_solvable` is a training aid; every published board
                # result is measured on the unfiltered distribution.
                first_click_safe=entry.first_click_safe,
                guarantee_solvable=False,
            )
            result = evaluate_agent(env, lambda obs: agent.select_action(obs, explore=False), args.eval_episodes)

            payload = {
                "agent": entry.agent,
                "level": entry.level,
                "density": density,
                "rows": rows,
                "cols": cols,
                "mines": mines,
                "env_version": tree,
                "env": {"first_click_safe": entry.first_click_safe, "guarantee_solvable": False},
                "eval_episodes": args.eval_episodes,
                "win_rate": result["win_rate"],
                "win_rate_ci95": ci95(result["win_rate"], args.eval_episodes),
                "avg_episode_length": round(result["avg_episode_length"], 3),
                "avg_reward": round(result["avg_reward"], 3),
                "failures": result["failures"],
                "checkpoint_source": entry.run,
            }
            out = args.public_dir / tree / "levels" / entry.level / density / f"{entry.agent.lower()}_board_result.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=1))
            print(f"    {density:<9} {result['win_rate'] * 100:6.2f}%   -> {out}")

    print()


if __name__ == "__main__":
    main()
