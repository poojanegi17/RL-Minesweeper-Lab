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

Reward shaping (`--reward-mode shaped`) only ever changes the reward the
*training* environment hands back; the final evaluation always runs against
`reward_mode="default"` on a separately-seeded env, so `win_rate` and
`avg_reward` stay comparable across every experiment regardless of how it was
trained (mirroring `evaluation.ppo_experiment`).

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
from evaluation.evaluate_board_config import env_version
from evaluation.metrics import evaluate_agent
from models.dqn_network import NETWORK_PRESETS
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
        # Derived from the presets rather than listed, so adding a preset does
        # not silently leave it unreachable from the CLI.
        choices=list(NETWORK_PRESETS),
        help="Network architecture preset -- 'small' varies width, 'deep' varies depth, and "
        "'fully_conv' replaces the board-size-specific Linear head with 1x1 convolutions "
        "(see models.dqn_network.NETWORK_PRESETS).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=20000,
        help="Replay buffer capacity in transitions (see DQNAgent's own default) -- lower this on memory-constrained "
        "machines for larger boards, since each transition's size scales with rows*cols.",
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
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative epsilon decay per episode. The 0.995 default reaches the 0.05 floor at "
        "episode 598, so a 25,000-episode run spends ~98%% of its time near-greedy; 0.9998 stretches "
        "that to ~15,000 episodes.",
    )
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Floor epsilon decays towards.")
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="default",
        choices=["default", "shaped"],
        help="Training-environment reward shaping (see module docstring). Evaluation always uses 'default'.",
    )
    parser.add_argument("--rows", type=int, default=ROWS, help="Board rows (see board_configs.py for the level/density presets).")
    parser.add_argument("--cols", type=int, default=COLS, help="Board columns.")
    parser.add_argument("--mines", type=int, default=NUM_MINES, help="Mine count.")
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Training seed -- board sequence the agent trains on, network init, epsilon-greedy "
        "draws, and replay sampling. Vary this (and only this) for seed replication.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=SEED,
        help="Seed for the evaluation environment, deliberately independent of --seed and fixed "
        "by default. Every run is then scored on the *same* 2,000 boards, so a difference between "
        "runs is a difference in play rather than in luck of the draw -- which is the property the "
        "Research page's variant comparisons already claim. Tying it to --seed instead would mean a "
        "replication seed changed both the training run and the test set at once, leaving "
        "training variance and evaluation-board variance inseparable.",
    )
    parser.add_argument(
        "--train-every",
        type=int,
        default=1,
        help="Environment steps between gradient updates. The default of 1 is one update per "
        "transition -- four times standard DQN's replay ratio. --train-every 4 spends the same "
        "gradient budget on roughly four times as much collected experience per wall-clock hour.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiplies training rewards only; evaluation always uses 1.0 so avg_reward stays "
        "comparable across runs. Use 0.1 to keep TD targets inside smooth_l1_loss's quadratic "
        "regime -- at the default +-10 rewards nearly every transition lands in the linear regime, "
        "where the gradient magnitude is constant and only the error's sign survives.",
    )
    parser.add_argument(
        "--first-click-safe",
        choices=["none", "cell", "area"],
        default="none",
        help="Opening-move policy, applied to BOTH training and evaluation because it defines the "
        "benchmark. Changes the board distribution, so results are only comparable to others "
        "measured under the same value (default: none, matching every committed result).",
    )
    parser.add_argument(
        "--guarantee-solvable",
        action="store_true",
        help="Train on no-guess boards only. Training-only, like --reward-mode: evaluation always "
        "runs on the unfiltered distribution, since CSP scores 100%% on no-guess boards by "
        "construction and an agent never shown a forced guess learns no risk-ranking for one.",
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

    lr_schedule = parse_lr_schedule(args.lr_schedule) if args.lr_schedule else None

    env = MinesweeperEnv(
        rows=args.rows,
        cols=args.cols,
        num_mines=args.mines,
        seed=args.seed,
        reward_mode=args.reward_mode,
        reward_scale=args.reward_scale,
        first_click_safe=args.first_click_safe,
        guarantee_solvable=args.guarantee_solvable,
    )
    agent = DQNAgent(
        rows=args.rows,
        cols=args.cols,
        lr=args.lr,
        lr_schedule=lr_schedule,
        buffer_capacity=args.buffer_size,
        batch_size=args.batch_size,
        target_update_every=args.target_update_every,
        train_every=args.train_every,
        network_size=args.network_size,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
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

    # A dedicated evaluation environment, for three reasons. It always uses the
    # *default* reward so `avg_reward` stays comparable across experiments no
    # matter how the run was trained (mirroring ppo_experiment.py); it is
    # freshly seeded, so evaluation isn't scored on whatever board sequence the
    # training env's RNG happened to reach after 25,000 resets -- which would
    # make this run's boards a function of its own training length; and its seed
    # is `--eval-seed`, held fixed across runs rather than following `--seed`,
    # so that varying the training seed doesn't silently vary the test set too.
    # `select_action(explore=False)` is a pure argmax with no RNG draw, so a
    # fixed eval seed makes the whole evaluation deterministic given the weights.
    #
    # `--first-click-safe` *is* carried over, unlike the other new options: it
    # defines which game is being benchmarked, so training and evaluation must
    # agree on it. `--guarantee-solvable` and `--reward-scale` are not, for the
    # same reason `--reward-mode` isn't -- they are training aids, and applying
    # them here would either hand evaluation an easier board distribution than
    # the benchmark or rescale `avg_reward` out of comparability.
    eval_env = MinesweeperEnv(
        rows=args.rows,
        cols=args.cols,
        num_mines=args.mines,
        seed=args.eval_seed,
        reward_mode="default",
        first_click_safe=args.first_click_safe,
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
        "lr": args.lr,
        "lr_schedule": lr_schedule,
        "batch_size": args.batch_size,
        "target_update_every": args.target_update_every,
        "network_size": args.network_size,
        "buffer_size": args.buffer_size,
        "reward_mode": args.reward_mode,
        "reward_scale": args.reward_scale,
        "train_every": args.train_every,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        # Board-generation provenance. `env_version` is derived from the
        # settings rather than declared, so a run cannot be mislabelled by a
        # stale flag; anything above "v1" was measured on a different board
        # distribution and must not be compared against the committed results.
        # `guarantee_solvable` is training-only, hence the explicit False for
        # the evaluation environment these numbers actually come from.
        "env_version": env_version(args.first_click_safe, args.guarantee_solvable),
        "env": {
            "first_click_safe": args.first_click_safe,
            "guarantee_solvable": args.guarantee_solvable,
        },
        "eval_env": {
            "first_click_safe": args.first_click_safe,
            "guarantee_solvable": False,
        },
        # Recorded so a replication set can be told apart from the original run
        # it replicates -- every figure in the README predating this field was a
        # single run at seed 42, evaluated on seed-42 boards.
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        # Recorded so the Research page's mechanical variant diff can surface it:
        # this is a code-level correctness change (see DQNAgent._double_dqn_targets),
        # not a CLI flag, so without it a masked run looks configuration-identical to
        # a pre-fix one despite a ~20x difference in win rate.
        "bootstrap_target_masked": True,
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
