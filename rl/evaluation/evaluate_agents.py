"""Compare RandomAgent, CSPAgent, QLearningAgent, DQNAgent, and PPOAgent on the same Minesweeper configuration.

Run with:
    python -m evaluation.evaluate_agents
"""

from __future__ import annotations

from pathlib import Path

from agents.csp_solver import CSPAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from environment.minesweeper_env import MinesweeperEnv
from evaluation.metrics import evaluate_agent
from training.history_export import save_history_csv, save_history_json

ROWS = 5
COLS = 5
NUM_MINES = 5
NUM_EPISODES = 200
Q_TRAIN_EPISODES = 20000
DQN_TRAIN_EPISODES = 6000
PPO_TRAIN_EPISODES = 6000
SEED = 42
RESULTS_DIR = "results"


def _print_results(name: str, results: dict) -> None:
    print(f"{name}:")
    print(f"  Games played: {results['games_played']}")
    print(f"  Win rate: {results['win_rate'] * 100:.1f}%")
    print(f"  Avg episode length: {results['avg_episode_length']:.2f}")
    print(f"  Failures: {results['failures']}")
    print()


def main() -> None:
    """Evaluate RandomAgent, CSPAgent, QLearningAgent, DQNAgent, and PPOAgent on identical board configurations and print a report."""
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

    q_agent = QLearningAgent(rows=ROWS, cols=COLS, seed=SEED)
    q_agent.train(env, episodes=Q_TRAIN_EPISODES)
    q_results = evaluate_agent(env, q_agent.select_action, num_episodes=NUM_EPISODES)

    dqn_agent = DQNAgent(rows=ROWS, cols=COLS, seed=SEED)
    checkpoint_dir = Path(RESULTS_DIR) / "checkpoints_evaluate_agents"
    dqn_history = dqn_agent.train(env, episodes=DQN_TRAIN_EPISODES, checkpoint_dir=checkpoint_dir)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    save_history_json(dqn_history, Path(RESULTS_DIR) / "dqn_evaluate_agents_history.json")
    save_history_csv(dqn_history, Path(RESULTS_DIR) / "dqn_evaluate_agents_history.csv")

    # Evaluate the best checkpoint seen during training, not just whatever
    # weights happen to be left in memory when the loop ends -- the whole
    # point of checkpoint selection is that those can differ.
    best_model_path = checkpoint_dir / "best_model.pt"
    used_checkpoint = "final in-memory weights (no checkpoint improved on -1.0)"
    if best_model_path.exists():
        dqn_agent.load_checkpoint(best_model_path)
        used_checkpoint = "best_model.pt"

    dqn_results = evaluate_agent(
        env,
        lambda observation: dqn_agent.select_action(observation, explore=False),
        num_episodes=NUM_EPISODES,
    )

    # Recommended PPO configuration, per the PPO improvement experiments
    # (see README): shaped reward (the one validated, real improvement) and
    # best-checkpoint deployment (a no-cost safety net -- see README for why
    # it didn't move the needle in that round, but is kept regardless).
    # reward_mode="shaped" only affects the *training* env; a separate
    # env is used for training vs. the shared `env` above so PPO's own
    # training rewards don't leak a different reward scale into other
    # agents' shared env/RNG stream, and evaluation below still runs on the
    # shared default-reward `env` for a fair comparison against every other
    # agent's numbers.
    ppo_train_env = MinesweeperEnv(rows=ROWS, cols=COLS, num_mines=NUM_MINES, seed=SEED, reward_mode="shaped")
    ppo_agent = PPOAgent(rows=ROWS, cols=COLS, seed=SEED)
    ppo_checkpoint_dir = Path(RESULTS_DIR) / "checkpoints_ppo_evaluate_agents"
    ppo_history = ppo_agent.train(
        ppo_train_env, episodes=PPO_TRAIN_EPISODES, checkpoint_dir=ppo_checkpoint_dir
    )
    save_history_json(ppo_history, Path(RESULTS_DIR) / "ppo_evaluate_agents_history.json")
    save_history_csv(ppo_history, Path(RESULTS_DIR) / "ppo_evaluate_agents_history.csv")

    ppo_best_policy_path = ppo_checkpoint_dir / "best_policy.pt"
    ppo_used_checkpoint = "final in-memory weights (no checkpoint improved during training)"
    if ppo_best_policy_path.exists():
        ppo_agent.load_checkpoint(ppo_best_policy_path)
        ppo_used_checkpoint = "best_policy.pt"

    ppo_results = evaluate_agent(
        env,
        lambda observation: ppo_agent.select_action(observation, explore=False),
        num_episodes=NUM_EPISODES,
    )

    print(f"Minesweeper {ROWS}x{COLS}, {NUM_MINES} mines, {NUM_EPISODES} episodes each")
    print(f"(Q-Learning trained for {Q_TRAIN_EPISODES} episodes, {len(q_agent.q_table)} states learned)")
    print(f"(DQN trained for {DQN_TRAIN_EPISODES} episodes, evaluated using {used_checkpoint})")
    print(f"(PPO trained for {PPO_TRAIN_EPISODES} episodes with shaped reward, evaluated using {ppo_used_checkpoint})\n")
    _print_results("Random Agent", random_results)
    _print_results("CSP Agent", csp_results)
    _print_results("Q-Learning Agent", q_results)
    _print_results("DQN Agent", dqn_results)
    _print_results("PPO Agent", ppo_results)


if __name__ == "__main__":
    main()
