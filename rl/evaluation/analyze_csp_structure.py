"""Measure *how* CSP wins and loses, not just how often.

The research narrative's CSP chapter rests on a structural claim -- that CSP
runs out of information rather than out of logic -- and quotes per-episode
deduction and guess counts to support it. Those figures previously lived in
`analysis/csp_structure.json` with no committed script behind them, so they
could not be regenerated or re-measured under a different environment. This is
that script.

Per (level, density) it reports:

    win             win rate
    first_die       episodes lost on the very first move. Under
                    `first_click_safe="area"` this is 0 by construction, which
                    is the entire point of the v2 environment.
    win_given_surv  win rate among episodes that survived the opening move --
                    the part of CSP's performance that is actually about
                    deduction rather than the coin flip
    ded             mean moves per episode taken from a *proven* safe cell
    guess           mean moves per episode with no provably safe cell available
    ratio           ded / guess
    lostbet         fraction of guesses that hit a mine
    win_moves_med   median moves in a *winning* episode -- "how long a typical
                    win actually runs", which the Random chapter contrasts
                    against how few moves a blind policy survives

A move counts as a deduction when `CSPAgent.find_safe_moves` is non-empty
before the move is chosen; those moves can never lose, so every loss is
attributable to a guess. Mine positions are never consulted -- the
classification uses only the board the agent itself sees.

Run with:
    python -m evaluation.analyze_csp_structure                       # v1
    python -m evaluation.analyze_csp_structure --first-click-safe area
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

from agents.csp_solver import CSPAgent
from board_configs import BOARD_LEVELS, DENSITY_ORDER, LEVEL_ORDER
from environment.minesweeper_env import MinesweeperEnv
from evaluation.evaluate_board_config import EVAL_EPISODES, SEED, env_version


def analyze_cell(
    rows: int,
    cols: int,
    mines: int,
    *,
    episodes: int,
    seed: int,
    first_click_safe: str,
) -> Dict[str, float]:
    """Play `episodes` CSP games at one board config and summarize their structure."""
    env = MinesweeperEnv(
        rows=rows, cols=cols, num_mines=mines, seed=seed, first_click_safe=first_click_safe
    )
    agent = CSPAgent(rows=rows, cols=cols, num_mines=mines, seed=seed)

    wins = first_deaths = deductions = guesses = lost_bets = 0
    survived = survived_wins = 0
    winning_move_counts: List[int] = []

    for _ in range(episodes):
        observation, _ = env.reset()
        agent.reset()
        terminated = truncated = False
        info: Dict[str, Any] = {"won": False}
        move_index = 0

        while not (terminated or truncated):
            board = observation.tolist()
            # Classify the move *before* taking it. `choose_action` recomputes
            # constraints internally; calling update_constraints first is
            # idempotent for a given board and lets us see what it will find.
            agent.update_constraints(board)
            is_deduction = bool(agent.find_safe_moves(board))

            action = agent.choose_action(board)
            observation, _, terminated, truncated, info = env.step(action)

            if is_deduction:
                deductions += 1
            else:
                guesses += 1
                if terminated and not info["won"]:
                    lost_bets += 1

            if move_index == 0 and terminated and not info["won"]:
                first_deaths += 1
            move_index += 1

        won = bool(info["won"])
        wins += won
        if won:
            winning_move_counts.append(move_index)
        # "Survived the opening" means the first move didn't end the game in a
        # loss; a one-move win (a full-board cascade) counts as survival.
        if not (move_index == 1 and not won):
            survived += 1
            survived_wins += won

    return {
        "win": wins / episodes,
        "first_die": first_deaths / episodes,
        "win_given_surv": (survived_wins / survived) if survived else 0.0,
        "ded": deductions / episodes,
        "guess": guesses / episodes,
        "ratio": (deductions / guesses) if guesses else float("inf"),
        "lostbet": (lost_bets / guesses) if guesses else 0.0,
        "win_moves_med": median(winning_move_counts) if winning_move_counts else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure CSP's deduction/guess structure per board config.")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--first-click-safe", choices=["none", "cell", "area"], default="none")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to analysis/csp_structure_{env_version}.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = env_version(args.first_click_safe, False)
    output = Path(args.output) if args.output else Path("analysis") / f"csp_structure_{version}.json"

    report: Dict[str, Dict[str, float]] = {}
    for level in LEVEL_ORDER:
        config = BOARD_LEVELS[level]
        for density in DENSITY_ORDER:
            if density not in config.densities:
                continue
            key = f"{level}/{density}"
            report[key] = analyze_cell(
                config.rows,
                config.cols,
                config.densities[density],
                episodes=args.episodes,
                seed=args.seed,
                first_click_safe=args.first_click_safe,
            )
            row = report[key]
            print(
                f"{key:22} win={row['win']:.4f} first_die={row['first_die']:.4f} "
                f"ded={row['ded']:6.3f} guess={row['guess']:5.3f} ratio={row['ratio']:7.3f} "
                f"lostbet={row['lostbet']:.4f} win_moves_med={row['win_moves_med']:5.1f}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nenv_version={version} -> {output}")


if __name__ == "__main__":
    main()
