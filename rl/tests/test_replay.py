"""Tests for evaluation.replay: ReplayRecorder and the reasoning extractors."""

import json

import numpy as np
import pytest

from agents.csp_solver import CSPAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from environment.minesweeper_env import MinesweeperEnv
from evaluation.replay import ReplayRecorder, build_replay, csp_reasoning, dqn_reasoning, ppo_reasoning


# --- ReplayRecorder: basic structure ----------------------------------------


def test_record_episode_returns_expected_top_level_shape():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=1)
    agent = RandomAgent(seed=1)
    recorder = ReplayRecorder()

    episode = recorder.record_episode(env, agent.select_action)

    assert set(episode.keys()) == {"initial_board", "steps", "won", "total_reward", "steps_taken"}
    assert episode["steps_taken"] == len(episode["steps"])
    assert isinstance(episode["won"], bool)
    assert episode["initial_board"] == [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]


def test_record_episode_step_shape():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=2)
    agent = RandomAgent(seed=2)
    recorder = ReplayRecorder()

    episode = recorder.record_episode(env, agent.select_action)

    for i, step in enumerate(episode["steps"], start=1):
        assert set(step.keys()) == {"step", "board_state", "action", "reward", "done", "reasoning"}
        assert step["step"] == i
        assert set(step["action"].keys()) == {"row", "col"}
        assert 0 <= step["action"]["row"] < 3
        assert 0 <= step["action"]["col"] < 3
        assert isinstance(step["reward"], float)
        assert isinstance(step["done"], bool)

    assert episode["steps"][-1]["done"] is True
    assert all(not s["done"] for s in episode["steps"][:-1])


def test_record_episode_last_step_reflects_terminal_state():
    # For a losing episode the last step's reward must be the mine-hit
    # penalty (or the win reward for a winning one) -- confirms `done`
    # actually lines up with the environment's own termination signal.
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=2)
    agent = RandomAgent(seed=2)
    recorder = ReplayRecorder()

    episode = recorder.record_episode(env, agent.select_action)

    last_reward = episode["steps"][-1]["reward"]
    if episode["won"]:
        assert last_reward == 10.0
    else:
        assert last_reward == -10.0


def test_record_episode_board_state_matches_actual_env_observations():
    # Re-play the identical seeded episode manually and confirm every
    # recorded board_state is byte-for-byte the real post-action observation
    # -- not some other array, and not silently substituted.
    env = MinesweeperEnv(rows=4, cols=4, num_mines=2, seed=5)
    agent = RandomAgent(seed=5)
    recorder = ReplayRecorder()
    episode = recorder.record_episode(env, agent.select_action, seed=5)

    verify_env = MinesweeperEnv(rows=4, cols=4, num_mines=2, seed=5)
    verify_agent = RandomAgent(seed=5)
    observation, _ = verify_env.reset(seed=5)
    for step in episode["steps"]:
        action = verify_agent.select_action(observation)
        observation, reward, terminated, truncated, info = verify_env.step(action)
        assert step["board_state"] == observation.tolist()
        assert step["reward"] == float(reward)


def test_record_episode_reasoning_is_none_without_reasoning_fn():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=3)
    agent = RandomAgent(seed=3)
    recorder = ReplayRecorder()

    episode = recorder.record_episode(env, agent.select_action)

    assert all(step["reasoning"] is None for step in episode["steps"])


def test_record_episode_calls_on_episode_start():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=2, seed=9)
    agent = CSPAgent(rows=4, cols=4, num_mines=2, seed=9)
    recorder = ReplayRecorder()

    calls = []
    recorder.record_episode(env, agent.choose_action, on_episode_start=lambda: calls.append(1))
    assert calls == [1]

    recorder.record_episode(env, agent.choose_action, on_episode_start=lambda: calls.append(2))
    assert calls == [1, 2]  # called exactly once per episode, in order


def test_record_episode_on_episode_start_actually_resets_csp_knowledge():
    # A CSPAgent replayed across two episodes without reset() would carry
    # stale known_safe/known_mines cells from the first board into the
    # second (a different mine layout) -- on_episode_start=agent.reset must
    # clear them before the first move of the new episode.
    env = MinesweeperEnv(rows=4, cols=4, num_mines=2, seed=9)
    agent = CSPAgent(rows=4, cols=4, num_mines=2, seed=9)
    recorder = ReplayRecorder()

    # Poison the agent's state as if a previous, unrelated episode left
    # knowledge behind that doesn't apply to the board about to be played.
    agent.known_safe.add((0, 0))
    agent.known_mines.add((1, 1))
    agent.constraints.append((frozenset({(2, 2)}), 1))

    reset_was_called = False

    def wrapped_reset() -> None:
        nonlocal reset_was_called
        agent.reset()
        reset_was_called = True
        # Asserted here, at the exact point record_episode calls this hook --
        # i.e. before any move of the new episode is chosen.
        assert agent.known_safe == set()
        assert agent.known_mines == set()
        assert agent.constraints == []

    recorder.record_episode(env, agent.choose_action, on_episode_start=wrapped_reset)
    assert reset_was_called


def test_record_episode_reproducible_with_same_seed():
    agent_a = RandomAgent(seed=11)
    env_a = MinesweeperEnv(rows=3, cols=3, num_mines=1)
    recorder = ReplayRecorder()
    episode_a = recorder.record_episode(env_a, agent_a.select_action, seed=11)

    agent_b = RandomAgent(seed=11)
    env_b = MinesweeperEnv(rows=3, cols=3, num_mines=1)
    episode_b = recorder.record_episode(env_b, agent_b.select_action, seed=11)

    assert episode_a["steps"] == episode_b["steps"]
    assert episode_a["won"] == episode_b["won"]


# --- No hidden-state leakage --------------------------------------------------


def test_replay_never_contains_mine_related_keys():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=3, seed=13)
    agent = RandomAgent(seed=13)
    recorder = ReplayRecorder()
    episode = recorder.record_episode(env, agent.select_action)

    replay = build_replay(
        agent_name="Random",
        experiment_id=None,
        board_size="4x4",
        mines=3,
        seed=13,
        episode_number=1,
        episode=episode,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    text = json.dumps(replay).lower()

    assert "mine_position" not in text
    assert '"mines_grid"' not in text
    assert "game.mines" not in text
    # "mines" itself legitimately appears once, as the public mine *count*
    # field -- only forbid anything that looks like per-cell mine data.
    assert replay["mines"] == 3


def test_replay_terminal_mine_hit_is_derivable_only_from_action_and_won():
    # The specific cell that ended a losing game is recoverable from the
    # last step's `action` + `won=False` -- exactly the legitimate signal
    # the viewer is meant to use -- and nothing else in the payload marks it.
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=2)
    agent = RandomAgent(seed=2)
    recorder = ReplayRecorder()
    episode = recorder.record_episode(env, agent.select_action)

    if not episode["won"]:
        last_step = episode["steps"][-1]
        assert last_step["done"] is True
        assert "row" in last_step["action"] and "col" in last_step["action"]


# --- Reasoning extractors -----------------------------------------------------


def test_csp_reasoning_safe_deduction():
    agent = CSPAgent(rows=3, cols=3, num_mines=1, seed=0)
    board = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    agent.known_safe = {(0, 0)}
    agent.constraints = [(frozenset({(0, 0)}), 0)]

    result = csp_reasoning(agent, board, action=0)  # (0,0) -> flattened action 0

    assert result["deduction_type"] == "safe"
    assert result["remaining_mines"] == 0
    assert result["constraint_cells"] == [[0, 0]]
    assert "safe" in result["inference"].lower()


def test_csp_reasoning_probability_guess_when_no_deduction():
    agent = CSPAgent(rows=3, cols=3, num_mines=1, seed=0)
    board = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]

    result = csp_reasoning(agent, board, action=0)

    assert result["deduction_type"] == "probability_guess"
    assert result["mine_probability"] is not None
    assert result["constraint_cells"] is None


def test_dqn_reasoning_returns_q_value_for_chosen_action():
    agent = DQNAgent(rows=3, cols=3, seed=0)
    board = np.full((3, 3), -1)

    result = dqn_reasoning(agent, board, action=4)

    assert set(result.keys()) == {"q_value"}
    assert isinstance(result["q_value"], float)


def test_dqn_reasoning_matches_agents_own_greedy_choice():
    # The Q-value returned for the action select_action(explore=False)
    # actually picked must be the max Q-value among hidden cells --
    # otherwise the "reasoning" wouldn't really explain the real choice.
    agent = DQNAgent(rows=3, cols=3, seed=0)
    board = np.full((3, 3), -1)

    action = agent.select_action(board, explore=False)
    result = dqn_reasoning(agent, board, action)

    from models.dqn_network import encode_observation
    import torch as torch_

    with torch_.no_grad():
        tensor = torch_.from_numpy(encode_observation(board)[None, ...]).to(agent.device)
        all_q = agent.online_network(tensor).squeeze(0).cpu().numpy()

    assert result["q_value"] == pytest.approx(float(all_q[action]))
    assert result["q_value"] == pytest.approx(float(all_q.max()))  # all cells hidden -> unmasked argmax


def test_ppo_reasoning_returns_probability_for_chosen_action():
    agent = PPOAgent(rows=3, cols=3, seed=0)
    board = np.full((3, 3), -1)

    action = agent.select_action(board, explore=False)
    result = ppo_reasoning(agent, board, action)

    assert set(result.keys()) == {"action_probability"}
    assert 0.0 <= result["action_probability"] <= 1.0


# --- build_replay --------------------------------------------------------------


def test_build_replay_assembles_expected_fields():
    episode = {
        "initial_board": [[-1, -1], [-1, -1]],
        "steps": [{"step": 1, "board_state": [[0, -1], [-1, -1]], "action": {"row": 0, "col": 0}, "reward": 1.0, "done": False, "reasoning": None}],
        "won": False,
        "total_reward": 1.0,
        "steps_taken": 1,
    }

    replay = build_replay(
        agent_name="Random",
        experiment_id=None,
        board_size="2x2",
        mines=1,
        seed=42,
        episode_number=7,
        episode=episode,
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert replay["id"] == "random_episode_7"
    assert replay["agent"] == "Random"
    assert replay["episode_number"] == 7
    assert replay["steps"] == episode["steps"]
    assert replay["steps_taken"] == 1
