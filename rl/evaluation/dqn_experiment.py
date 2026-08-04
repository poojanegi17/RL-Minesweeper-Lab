"""Train a DQNAgent for a configurable number of episodes and record the run.

Trains on the standard 5x5/5-mine benchmark board, exports the full per-episode
training history to JSON and CSV, checkpoints the best-scoring policy seen
during training (see `--checkpoint-every`), evaluates it, and writes a summary
JSON alongside the history files.

By default the final evaluation loads `best_model.pt` (the best-scoring
checkpoint seen during training) rather than whatever weights happen to be in
memory when training ends -- pass `--no-best-checkpoint` to instead evaluate
the raw final weights, which is useful for isolating the effect of checkpoint
selection itself in a controlled comparison.

Run with:
    python -m evaluation.dqn_experiment --episodes 5000
    python -m evaluation.dqn_experiment --episodes 25000 --network-size small
    python -m evaluation.dqn_experiment --episodes 25000 \\
        --lr-schedule "0:1e-4,10000:5e-5,20000:1e-5"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import torch

from agents.dqn_agent import DQNAgent
from environment.minesweeper_env import MinesweeperEnv
from evaluation.metrics import evaluate_agent
from training.history_export import save_history_csv, save_history_json

ROWS = 5
COLS = 5
NUM_MINES = 5
EVAL_EPISODES = 200
SEED = 42


def parse_lr_schedule(text: str) -> List[Tuple[int, float]]:
    """Parse a "episode:lr,episode:lr,..." string into a schedule list.

    Example: "0:1e-4,20000:5e-5,40000:1e-5" -> [(0, 1e-4), (20000, 5e-5), (40000, 1e-5)].
    """
    pairs = []
    for chunk in text.split(","):
        episode_str, lr_str = chunk.split(":")
        pairs.append((int(episode_str), float(lr_str)))
    return pairs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training run."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a DQNAgent, exporting its training history."
    )
    parser.add_argument("--episodes", type=int, required=True, help="Number of training episodes.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate (ignored if --lr-schedule is set).")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        help='Optional decay schedule as "episode:lr,episode:lr,...", e.g. "0:1e-4,20000:5e-5,40000:1e-5".',
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size per gradient step.")
    parser.add_argument(
        "--target-update-every", type=int, default=25, help="Episodes between target-network syncs."
    )
    parser.add_argument(
        "--network-size",
        type=str,
        default="default",
        choices=["default", "small"],
        help="Network capacity preset (see models.dqn_network.NETWORK_PRESETS).",
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
        help="Evaluate the raw final weights instead of reloading best_model.pt after training.",
    )
    parser.add_argument("--rolling-window", type=int, default=100, help="Window size for rolling metrics.")
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to write history/summary/checkpoint files into."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
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

    lr_schedule = parse_lr_schedule(args.lr_schedule) if args.lr_schedule else None

    env = MinesweeperEnv(rows=ROWS, cols=COLS, num_mines=NUM_MINES, seed=args.seed)
    agent = DQNAgent(
        rows=ROWS,
        cols=COLS,
        lr=args.lr,
        lr_schedule=lr_schedule,
        batch_size=args.batch_size,
        target_update_every=args.target_update_every,
        network_size=args.network_size,
        rolling_window=args.rolling_window,
        seed=args.seed,
    )

    start = time.time()
    history = agent.train(
        env,
        episodes=args.episodes,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
    )
    train_seconds = time.time() - start

    stem = f"dqn_history_{args.episodes}"
    save_history_json(history, output_dir / f"{stem}.json")
    save_history_csv(history, output_dir / f"{stem}.csv")

    best_checkpoint_metadata = None
    best_model_path = checkpoint_dir / "best_model.pt"
    used_checkpoint = "final_in_memory_weights"
    if not args.no_best_checkpoint and best_model_path.exists():
        best_checkpoint_metadata = agent.load_checkpoint(best_model_path)
        used_checkpoint = "best_model.pt"

    eval_results = evaluate_agent(
        env,
        lambda observation: agent.select_action(observation, explore=False),
        num_episodes=EVAL_EPISODES,
    )

    summary = {
        "episodes": args.episodes,
        "lr": args.lr,
        "lr_schedule": lr_schedule,
        "batch_size": args.batch_size,
        "target_update_every": args.target_update_every,
        "network_size": args.network_size,
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
        f"episodes={args.episodes} network={args.network_size} "
        f"lr_schedule={'yes' if lr_schedule else 'no'} used={used_checkpoint} "
        f"train_seconds={train_seconds:.1f} "
        f"win_rate={eval_results['win_rate'] * 100:.1f}% "
        f"avg_length={eval_results['avg_episode_length']:.2f} "
        f"failures={eval_results['failures']}"
        + (f" best_ckpt_win_rate={best_checkpoint_metadata['win_rate'] * 100:.1f}%" if best_checkpoint_metadata else "")
    )


if __name__ == "__main__":
    main()
