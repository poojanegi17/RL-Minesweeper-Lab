"""Compare RandomAgent and CSPAgent on the same Minesweeper configuration.

Run with:
    python -m evaluation.evaluate_agents
"""

from __future__ import annotations

from agents.csp_solver import CSPAgent
from agents.random_agent import RandomAgent
from environment.minesweeper_env import MinesweeperEnv
from evaluation.metrics import evaluate_agent

ROWS = 5
COLS = 5
NUM_MINES = 5
NUM_EPISODES = 200
SEED = 42


def _print_results(name: str, results: dict) -> None:
    print(f"{name}:")
    print(f"  Games played: {results['games_played']}")
    print(f"  Win rate: {results['win_rate'] * 100:.1f}%")
    print(f"  Avg episode length: {results['avg_episode_length']:.2f}")
    print(f"  Failures: {results['failures']}")
    print()


def main() -> None:
    """Evaluate RandomAgent and CSPAgent on identical board configurations and print a report."""
    env = MinesweeperEnv(rows=ROWS, cols=COLS, num_mines=NUM_MINES, seed=SEED)

    random_agent = RandomAgent(seed=SEED)
    random_results = evaluate_agent(env, random_agent.select_action, num_episodes=NUM_EPISODES)

    csp_agent = CSPAgent(rows=ROWS, cols=COLS, num_mines=NUM_MINES, seed=SEED)
    csp_results = evaluate_agent(
        env,
        csp_agent.choose_action,
        num_episodes=NUM_EPISODES,
        on_episode_start=csp_agent.reset,
    )

    print(f"Minesweeper {ROWS}x{COLS}, {NUM_MINES} mines, {NUM_EPISODES} episodes each\n")
    _print_results("Random Agent", random_results)
    _print_results("CSP Agent", csp_results)


if __name__ == "__main__":
    main()
