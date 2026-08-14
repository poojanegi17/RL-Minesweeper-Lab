"""Train a PPOAgent for a configurable number of episodes and record the run.

Trains on the standard 5x5/5-mine benchmark board, exports the full per-episode
training history to JSON and CSV, checkpoints the best-scoring policy seen
during training (see `--checkpoint-every`), evaluates it, and writes a summary
JSON alongside the history files. Mirrors `evaluation.dqn_experiment`'s shape
so PPO and DQN experiments can be compared with the same tooling
(`compare_ppo_experiments`, `history_export`).

Reward shaping (`--reward-mode shaped`) only ever changes the reward the
*training* environment hands back; the final 2,000-episode evaluation always
runs against `reward_mode="default"` (a separate env instance, same seed), so
`avg_reward` in the summary is on a comparable scale across every experiment
regardless of how it was trained -- otherwise a shaped run's inflated
per-step rewards would make "avg_reward" meaningless as a cross-experiment
comparison column. Win rate needs no such adjustment since it's already
reward-scale-invariant.

By default the final evaluation loads `best_policy.pt` (the best-scoring
checkpoint seen during training) rather than whatever weights happen to be in
memory when training ends -- pass `--no-best-checkpoint` to instead evaluate
the raw final weights, which is useful for isolating the effect of checkpoint
selection itself in a controlled comparison (mirroring `dqn_experiment`).

`--first-click-safe` is the one board-generation option that reaches the
evaluation environment too, because it defines the benchmark rather than
shaping training: under "none" the opening click can hit a mine, which decides
roughly a fifth of 5x5 games before the agent acts at all. `--guarantee-solvable`
is training-only, like `--reward-mode`. Both default to the settings every
committed PPO result used, so existing runs reproduce byte-for-byte.

Run with:
    python -m evaluation.ppo_experiment --episodes 6000
    python -m evaluation.ppo_experiment --episodes 25000 --reward-mode shaped
    python -m evaluation.ppo_experiment --episodes 100000 --reward-mode shaped \\
        --first-click-safe area --output-dir results/ppo_v2_A_shaped
    python -m evaluation.ppo_experiment --episodes 25000 --reward-mode shaped \\
        --checkpoint-every 2500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from agents.ppo_agent import PPOAgent
from models.ppo_network import NETWORK_PRESETS
from environment.minesweeper_env import MinesweeperEnv
from evaluation.evaluate_board_config import env_version
from evaluation.metrics import evaluate_agent
from training.history_export import save_history_csv, save_history_json

ROWS = 5
COLS = 5
NUM_MINES = 5
# 2,000 rather than a couple of hundred: at this benchmark's 1-3% win rates a
# 200-episode evaluation rests on a handful of wins, wide enough that it
# reordered this project's own ablation conclusions once re-scored (see the
# README's "Evaluation sample size" and evaluation/reevaluate_checkpoints.py).
# The extra episodes cost seconds against a training run measured in minutes.
EVAL_EPISODES = 2000
SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training run."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a PPOAgent, exporting its training history."
    )
    parser.add_argument("--episodes", type=int, required=True, help="Number of training episodes.")
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="default",
        choices=["default", "shaped"],
        help="Training-environment reward shaping (see module docstring). Evaluation always uses 'default'.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE bias/variance tradeoff parameter.")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO probability-ratio clip range.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus weight.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss weight.")
    parser.add_argument("--rollout-length", type=int, default=256, help="Environment steps collected per rollout.")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Passes over each rollout per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size within a PPO epoch.")
    parser.add_argument(
        "--network-size",
        choices=list(NETWORK_PRESETS),
        default="default",
        help="Network preset (models/ppo_network.py). \"default\" is the two-conv, Linear-head "
        "design every PPO run before this flag used, kept as the default so published checkpoints "
        "still load. \"fully_conv\" swaps the Linear head for 1x1 convolutions and a pooled critic, "
        "making one set of weights valid at any board size -- the same change that let DQN carry a "
        "5x5 configuration to 9x9.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=500, help="Episodes between checkpoint evaluations."
    )
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=50,
        help="Episodes played (greedily) per checkpoint evaluation.",
    )
    parser.add_argument(
        "--no-best-checkpoint",
        action="store_true",
        help="Evaluate the raw final weights instead of reloading best_policy.pt after training.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to write history/summary/checkpoint files into."
    )
    parser.add_argument("--rows", type=int, default=ROWS, help="Board rows (see board_configs.py for the level/density presets).")
    parser.add_argument("--cols", type=int, default=COLS, help="Board columns.")
    parser.add_argument("--mines", type=int, default=NUM_MINES, help="Mine count.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=SEED,
        help="Seed for the evaluation environment, deliberately independent of --seed and fixed "
        "by default, so every run is scored on the *same* 2,000 boards as every other agent "
        "(mirrors evaluation.dqn_experiment). Tying it to --seed instead would mean a replication "
        "seed changed both the training run and the test set at once.",
    )
    parser.add_argument(
        "--first-click-safe",
        choices=["none", "cell", "area"],
        default="none",
        help="Opening-move policy, applied to BOTH training and evaluation because it defines the "
        "benchmark. Changes the board distribution, so results are only comparable to others "
        "measured under the same value (default: none, matching every committed PPO result).",
    )
    parser.add_argument(
        "--guarantee-solvable",
        action="store_true",
        help="Train on no-guess boards only. Training-only, like --reward-mode: evaluation always "
        "runs on the unfiltered distribution, since an agent never shown a forced guess learns no "
        "risk-ranking for one.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Cap CPU threads PyTorch uses (useful when running experiments in parallel).",
    )
    return parser.parse_args()


def main() -> None:
    """Train, checkpoint, export history, evaluate, and print a one-line summary."""
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoints_{args.episodes}"

    train_env = MinesweeperEnv(
        rows=args.rows,
        cols=args.cols,
        num_mines=args.mines,
        seed=args.seed,
        reward_mode=args.reward_mode,
        first_click_safe=args.first_click_safe,
        guarantee_solvable=args.guarantee_solvable,
    )
    agent = PPOAgent(
        rows=args.rows,
        cols=args.cols,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        rollout_length=args.rollout_length,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        network_size=args.network_size,
    )

    start = time.time()
    history = agent.train(
        train_env,
        episodes=args.episodes,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
    )
    train_seconds = time.time() - start

    stem = f"ppo_history_{args.episodes}"
    save_history_json(history, output_dir / f"{stem}.json")
    save_history_csv(history, output_dir / f"{stem}.csv")

    best_checkpoint_metadata = None
    best_policy_path = checkpoint_dir / "best_policy.pt"
    used_checkpoint = "final_in_memory_weights"
    if not args.no_best_checkpoint and best_policy_path.exists():
        best_checkpoint_metadata = agent.load_checkpoint(best_policy_path)
        used_checkpoint = "best_policy.pt"

    # Always evaluate under the *default* reward and on the unfiltered board
    # distribution -- see module docstring. `first_click_safe` is the one
    # option that *does* carry over, because it defines the benchmark rather
    # than shaping training.
    eval_env = MinesweeperEnv(
        rows=args.rows,
        cols=args.cols,
        num_mines=args.mines,
        seed=args.eval_seed,
        reward_mode="default",
        first_click_safe=args.first_click_safe,
        guarantee_solvable=False,
    )
    eval_results = evaluate_agent(
        eval_env,
        lambda observation: agent.select_action(observation, explore=False),
        num_episodes=EVAL_EPISODES,
    )

    summary = {
        "episodes": args.episodes,
        "rows": args.rows,
        "cols": args.cols,
        "mines": args.mines,
        "reward_mode": args.reward_mode,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_epsilon": args.clip_epsilon,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "rollout_length": args.rollout_length,
        "ppo_epochs": args.ppo_epochs,
        "batch_size": args.batch_size,
        "network_size": args.network_size,
        "env_version": env_version(args.first_click_safe, args.guarantee_solvable),
        "env": {"first_click_safe": args.first_click_safe, "guarantee_solvable": args.guarantee_solvable},
        "eval_env": {"first_click_safe": args.first_click_safe, "guarantee_solvable": False},
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "checkpoint_every": args.checkpoint_every,
        "used_checkpoint": used_checkpoint,
        "best_checkpoint_metadata": best_checkpoint_metadata,
        "train_seconds": round(train_seconds, 1),
        "eval_episodes": EVAL_EPISODES,
        "win_rate": eval_results["win_rate"],
        "avg_episode_length": eval_results["avg_episode_length"],
        "avg_reward": eval_results["avg_reward"],
        "failures": eval_results["failures"],
    }
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"episodes={args.episodes} reward_mode={args.reward_mode} used={used_checkpoint} "
        f"train_seconds={train_seconds:.1f} "
        f"win_rate={eval_results['win_rate'] * 100:.1f}% "
        f"avg_length={eval_results['avg_episode_length']:.2f} "
        f"failures={eval_results['failures']}"
        + (f" best_ckpt_win_rate={best_checkpoint_metadata['win_rate'] * 100:.1f}%" if best_checkpoint_metadata else "")
    )


if __name__ == "__main__":
    main()
