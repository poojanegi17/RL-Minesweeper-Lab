"""Tests for the tabular Q-learning agent."""

import numpy as np
import pytest

from agents.q_learning_agent import QLearningAgent
from environment.minesweeper_env import MinesweeperEnv


def test_encode_state_produces_hashable_flattened_tuple():
    board = np.array([[-1, 1], [2, -1]])
    agent = QLearningAgent(rows=2, cols=2, seed=0)

    state = agent.encode_state(board)

    assert state == (-1, 1, 2, -1)
    assert hash(state) is not None  # usable as a dict key
    # Nested-list input should encode identically to a numpy array.
    assert agent.encode_state([[-1, 1], [2, -1]]) == state


def test_choose_action_only_returns_hidden_cells():
    board = np.array([[-1, 1], [2, -1]])
    agent = QLearningAgent(rows=2, cols=2, seed=0)

    for _ in range(20):
        action = agent.choose_action(board, explore=True)
        assert action in (0, 3)  # flattened indices of the two -1 cells


def test_epsilon_one_always_explores_randomly():
    board = np.full((3, 3), -1)
    agent = QLearningAgent(rows=3, cols=3, epsilon=1.0, seed=1)
    # Force one action to look best; with epsilon=1.0 exploration should
    # still pick a variety of actions rather than always the greedy one.
    agent.q_table[agent.encode_state(board)] = np.array(
        [10.0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    actions = {agent.choose_action(board, explore=True) for _ in range(50)}
    assert len(actions) > 1


def test_epsilon_zero_is_fully_greedy():
    board = np.full((3, 3), -1)
    agent = QLearningAgent(rows=3, cols=3, epsilon=0.0, seed=2)
    state = agent.encode_state(board)
    agent.q_table[state] = np.array([0, 0, 5.0, 0, 0, 0, 0, 0, 0])

    for _ in range(10):
        assert agent.choose_action(board, explore=True) == 2


def test_select_action_never_explores_regardless_of_epsilon():
    board = np.full((2, 2), -1)
    agent = QLearningAgent(rows=2, cols=2, epsilon=1.0, seed=3)
    state = agent.encode_state(board)
    agent.q_table[state] = np.array([0, 0, 0, 9.0])

    for _ in range(10):
        assert agent.select_action(board) == 3


def test_update_q_value_applies_the_bellman_update():
    agent = QLearningAgent(rows=2, cols=2, alpha=0.5, gamma=0.9, seed=0)
    state = np.full((2, 2), -1)
    next_state = np.array([[1, -1], [-1, -1]])

    # Seed a next-state Q-value so the bootstrapped target is predictable.
    next_key = agent.encode_state(next_state)
    agent.q_table[next_key] = np.array([0.0, 4.0, 0.0, 0.0])

    agent.update_q_value(state, action=0, reward=1.0, next_state=next_state, done=False)

    # target = reward + gamma * max_a' Q(s', a') = 1.0 + 0.9 * 4.0 = 4.6
    # new_q = old_q(0) + alpha * (target - old_q(0)) = 0 + 0.5 * 4.6 = 2.3
    assert agent.get_q_value(state, 0) == 2.3


def test_update_q_value_ignores_future_return_when_done():
    agent = QLearningAgent(rows=2, cols=2, alpha=1.0, gamma=0.9, seed=0)
    state = np.full((2, 2), -1)
    next_state = np.array([[-1, -1], [-1, -1]])  # would-be mine-hit board

    agent.update_q_value(state, action=1, reward=-10.0, next_state=next_state, done=True)

    # alpha=1.0 so the Q-value becomes exactly the reward, ignoring next-state value.
    assert agent.get_q_value(state, 1) == -10.0


def test_agent_can_complete_training_episodes():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=3, seed=5)
    agent = QLearningAgent(rows=4, cols=4, seed=5)

    history = agent.train(env, episodes=20)

    assert len(history) == 20
    assert all(entry["steps"] > 0 for entry in history)
    assert all(isinstance(entry["won"], bool) for entry in history)
    # Epsilon should have decayed (not increased) over training.
    assert history[-1]["epsilon"] <= history[0]["epsilon"]
    assert len(agent.q_table) > 0


def test_trained_agent_interacts_correctly_with_minesweeper_env():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=3, seed=6)
    agent = QLearningAgent(rows=4, cols=4, seed=6)
    agent.train(env, episodes=10)

    observation, info = env.reset()
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < 16:
        action = agent.select_action(observation)
        assert 0 <= action < 16
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps > 0


def test_agent_never_accesses_hidden_mine_positions():
    # Sanity check that the agent's public surface only takes observations,
    # never the environment/game object itself.
    import inspect

    for name in ("choose_action", "select_action", "update_q_value", "encode_state"):
        method = getattr(QLearningAgent, name)
        params = list(inspect.signature(method).parameters)
        assert "env" not in params and "game" not in params and "mines" not in params


def test_unseen_state_gives_all_zero_values_so_greedy_is_a_uniform_guess():
    """The property that decides this agent's whole story.

    A never-visited board returns an all-zero row, so every legal move ties and
    the greedy branch of `choose_action` breaks that tie at random. On an unseen
    state the agent is behaviourally identical to `RandomAgent` -- which is why
    its win rate tracks state repetition rather than board difficulty.
    """
    agent = QLearningAgent(rows=4, cols=4, seed=0)
    board = [[-1] * 4 for _ in range(4)]

    assert agent.encode_state(board) not in agent.q_table
    values = agent._q_values(agent.encode_state(board))
    assert np.all(values == 0.0)

    # Greedy over an all-zero row must spread across cells, not fix on one.
    picks = {QLearningAgent(rows=4, cols=4, seed=s).choose_action(board, explore=False) for s in range(40)}
    assert len(picks) > 1


def test_greedy_prefers_a_learned_action_once_values_differ():
    # The other half: where the table *has* learned something, greedy uses it.
    agent = QLearningAgent(rows=4, cols=4, seed=0)
    board = [[-1] * 4 for _ in range(4)]
    agent._q_values(agent.encode_state(board))[5] = 1.0

    assert all(QLearningAgent.choose_action(agent, board, explore=False) == 5 for _ in range(5))


def test_bellman_update_matches_the_documented_arithmetic():
    """Pins the worked example shown in the README and the agent explainer:
    Q=0.20, r=+1, max Q(s',a')=0.80, alpha=0.1, gamma=0.9 -> 0.35."""
    agent = QLearningAgent(rows=4, cols=4, alpha=0.1, gamma=0.9, seed=0)
    state = [[-1] * 4 for _ in range(4)]
    next_state = [[0] + [-1] * 3] + [[-1] * 4 for _ in range(3)]

    agent._q_values(agent.encode_state(state))[0] = 0.20
    next_values = agent._q_values(agent.encode_state(next_state))
    next_values[:] = 0.0
    # Only hidden cells count toward the next-state max, so put 0.80 on one.
    next_values[1] = 0.80

    agent.update_q_value(state, 0, reward=1.0, next_state=next_state, done=False)

    assert agent.get_q_value(state, 0) == pytest.approx(0.352)


def test_terminal_update_drops_the_bootstrap_term():
    # On a terminal step the target is the reward alone.
    agent = QLearningAgent(rows=4, cols=4, alpha=0.1, gamma=0.9, seed=0)
    state = [[-1] * 4 for _ in range(4)]
    agent._q_values(agent.encode_state(state))[0] = 0.20

    agent.update_q_value(state, 0, reward=-10.0, next_state=state, done=True)

    # 0.20 + 0.1 * (-10 - 0.20) = -0.82, with no gamma term involved.
    assert agent.get_q_value(state, 0) == pytest.approx(-0.82)


def test_next_state_max_ignores_already_revealed_cells():
    # A high value parked on a revealed cell must not be bootstrapped from.
    agent = QLearningAgent(rows=4, cols=4, alpha=1.0, gamma=1.0, seed=0)
    state = [[-1] * 4 for _ in range(4)]
    next_state = [[0] + [-1] * 3] + [[-1] * 4 for _ in range(3)]

    next_values = agent._q_values(agent.encode_state(next_state))
    next_values[0] = 99.0  # index 0 is revealed in next_state
    next_values[1] = 1.0   # index 1 is still hidden

    agent.update_q_value(state, 5, reward=0.0, next_state=next_state, done=False)

    # alpha=1, gamma=1 -> Q becomes the target exactly: 0 + 1*1.0, not 99.
    assert agent.get_q_value(state, 5) == pytest.approx(1.0)
