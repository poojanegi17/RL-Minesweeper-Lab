"""Re-score each DQN/PPO ablation variant's own checkpoint at the other two Beginner densities.

The frontend's `lib/variantDensityResults.ts` shows, per ablation variant, how
that variant's checkpoint does at beginner/sparse (3 mines) and beginner/dense
(8 mines) -- the same 5x5 board it trained on, at a density it never saw.
beginner/standard isn't included there because that is exactly the run's own
`evaluation_metrics`.

Those figures were 200 episodes each, the same sample size
`reevaluate_checkpoints.py` exists to replace. This script re-scores them.

Each variant is scored on the checkpoint it actually deployed, resolved from
its own summary's `used_checkpoint` by `experiment_checkpoint_path`. That
matters here: a run's headline win rate at beginner/standard describes those
weights, so scoring a different file would put two checkpoints of one run in a
single row and read as a density effect. (This previously always preferred
`best_model.pt`/`best_policy.pt`, which for the 31-of-47 runs that deployed
their final weights was not the model the published number describes.)

Run with:
    python -m evaluation.rescore_variant_densities --episodes 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from board_configs import resolve
from environment.minesweeper_env import MinesweeperEnv
from evaluation.agent_loading import build_agent
from evaluation.metrics import evaluate_agent
from evaluation.reevaluate_checkpoints import wilson_interval

SEED = 42
LEVEL = "beginner"
DENSITIES = ["sparse", "dense"]

# The ablation variants the frontend table covers, in display order. Each must
# be a real experiment directory under `--results-dir`.
# Order matches the frontend's `VARIANT_DISPLAY_ORDER` for DQN Beginner, then
# the PPO Beginner family. These are exactly the runs the research pipeline
# shows as cards -- the previous list covered the five DQN variants since
# retired to `HIDDEN_VARIANTS` plus four PPO runs no longer shipped in
# `results_public/` at all, so every figure it produced keyed a run the site
# never renders while every visible run had none.
VARIANT_IDS = [
    "exp_F_masked_target",
    "exp_G_masked_lr_decay",
    "exp_H_masked_shaped",
    "exp_I_masked_slow_epsilon",
    "exp_J_masked_deep",
    "exp_K_masked_longer",
    "exp_L_tuned",
    "exp_N_no_reward_scale",
    "exp_O_short_epsilon",
    "exp_P_train_every_1",
    "exp_M_fully_conv",
    "ppo_exp_E_longer_matched",
    "ppo_exp_F_shaped_matched",
    "ppo_exp_G_shaped_ckpt_matched",
    # PPO Beginner's budget and architecture chapters (the `ppo_long` family).
    # Absent until now, which is why those chapters' cards rendered with no
    # density table at all while every other chapter had one.
    "ppo_long_A_baseline",
    "ppo_long_B_shaped",
    "ppo_long_D_gamma09_matched",
    "ppo_long_E_fully_conv",
    "ppo_long_F_shaped_reseeded",
    # The v2 environment chapters' runs. Scored under their own `eval_env`
    # (`first_click_safe: area`) rather than the v1 default -- see `eval_env_for`.
    "dqn_v2_A_baseline",
    "ppo_v2_F_shaped_matched",
    "ppo_v2_G_solvable_matched",
    "ppo_v2_H_gamma09_matched",
    "ppo_v2_I_gamma09_solvable_matched",
    "ppo_v2_J_fully_conv",
    "ppo_v2_K_solvable_fully_conv",
]


def agent_kind_for(variant_id: str) -> str:
    """"dqn" or "ppo", from the experiment id's own prefix."""
    return "ppo" if variant_id.startswith("ppo_") else "dqn"


def eval_env_for(variant_id: str, results_dir: Path) -> Dict[str, object]:
    """The board distribution this run's own headline win rate was measured
    under, read from its summary's `eval_env`.

    The sparse/dense rows sit in one table beside that headline number, so all
    three have to describe the same game. Scoring every run on the v1 default
    would put a `first_click_safe: area` run's Standard cell next to two cells
    measured with the opening click able to lose -- a distribution change read
    as a density effect, the same failure mode the `used_checkpoint` resolution
    above exists to prevent.

    `guarantee_solvable` is deliberately not carried over: it is a training-only
    curriculum (CSP clears such boards 100%% by construction, so it is not a
    benchmark setting), and the runs that use it already record
    `guarantee_solvable: false` in their own `eval_env`.
    """
    summaries = sorted((results_dir / variant_id).glob("*_summary.json"))
    if not summaries:
        return {"first_click_safe": "none"}
    summary = json.loads(summaries[0].read_text())
    eval_env = summary.get("eval_env") or {}
    return {"first_click_safe": eval_env.get("first_click_safe", "none")}


def score_variant(variant_id: str, density: str, episodes: int, results_dir: Path) -> Dict[str, float]:
    """Evaluate one variant's checkpoint at one Beginner density."""
    rows, cols, mines = resolve(LEVEL, density)
    _agent, action_fn, _reasoning, on_episode_start = build_agent(
        agent_kind_for(variant_id), SEED, variant_id, results_dir, rows=rows, cols=cols, num_mines=mines
    )
    env = MinesweeperEnv(
        rows=rows, cols=cols, num_mines=mines, seed=SEED, **eval_env_for(variant_id, results_dir)
    )
    result = evaluate_agent(env, action_fn, num_episodes=episodes, on_episode_start=on_episode_start)
    low, high = wilson_interval(result["wins"], episodes)
    return {
        "win_rate": result["win_rate"],
        "win_rate_ci95": [round(low, 3), round(high, 3)],
        "avg_episode_length": result["avg_episode_length"],
        "avg_reward": round(result["avg_reward"], 4),
        "wins": result["wins"],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Re-score ablation variants at the other two Beginner densities.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="analysis/variant_density_report.json")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Score every variant at every covered density and write a JSON report."""
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    results_dir = Path(args.results_dir)
    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    missing: List[str] = []

    print(f"Re-scoring {len(VARIANT_IDS)} variants x {len(DENSITIES)} densities at {args.episodes} episodes\n")
    for variant_id in VARIANT_IDS:
        if not (results_dir / variant_id).is_dir():
            missing.append(variant_id)
            continue
        report[variant_id] = {}
        cells = []
        for density in DENSITIES:
            scored = score_variant(variant_id, density, args.episodes, results_dir)
            report[variant_id][density] = scored
            cells.append(
                f"{density}={scored['win_rate'] * 100:5.2f}% "
                f"CI[{scored['win_rate_ci95'][0]:.2f},{scored['win_rate_ci95'][1]:.2f}]"
            )
        print(f"  {variant_id:<30} " + "  ".join(cells))

    if missing:
        print("\nSkipped (no experiment directory):")
        for variant_id in missing:
            print(f"  {variant_id}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"eval_episodes": args.episodes, "seed": SEED, "variants": report}, indent=2))
    print(f"\nWrote {output}")

    print("\nTypeScript block for frontend/src/lib/variantDensityResults.ts:\n")
    for variant_id, densities in report.items():
        print(f"  {variant_id}: {{")
        for density in DENSITIES:
            cell = densities[density]
            print(
                f"    {density}: {{ win_rate: {cell['win_rate']}, "
                f"avg_episode_length: {cell['avg_episode_length']}, avg_reward: {cell['avg_reward']} }},"
            )
        print("  },")


if __name__ == "__main__":
    main()
