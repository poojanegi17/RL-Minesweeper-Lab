"""Tests for evaluation.replay.build_race and the shared-seed determinism it relies on."""

from agents.csp_solver import CSPAgent
from agents.random_agent import RandomAgent
from environment.minesweeper_env import MinesweeperEnv
from evaluation.replay import ReplayRecorder, build_race


def test_build_race_assembles_expected_fields():
    episode_a = {
        "initial_board": [[-1, -1], [-1, -1]],
        "steps": [{"step": 1, "board_state": [[0, -1], [-1, -1]], "action": {"row": 0, "col": 0}, "reward": 1.0, "done": False, "reasoning": None}],
        "won": False,
        "total_reward": 1.0,
        "steps_taken": 1,
    }
    episode_b = {
        "initial_board": [[-1, -1], [-1, -1]],
        "steps": [{"step": 1, "board_state": [[-1, -1], [-1, 0]], "action": {"row": 1, "col": 1}, "reward": 10.0, "done": True, "reasoning": None}],
        "won": True,
        "total_reward": 10.0,
        "steps_taken": 1,
    }

    race = build_race(
        seed=7,
        board_size="2x2",
        mines=1,
        generated_at="2026-01-01T00:00:00+00:00",
        agent_episodes={"Random": (None, episode_a), "CSP": (None, episode_b)},
    )

    assert race["id"] == "race_7"
    assert race["seed"] == 7
    assert race["initial_board"] == [[-1, -1], [-1, -1]]
    assert set(race["agents"].keys()) == {"Random", "CSP"}
    assert race["agents"]["Random"]["won"] is False
    assert race["agents"]["CSP"]["won"] is True
    assert race["agents"]["CSP"]["steps"] == episode_b["steps"]
    # Per-agent entries don't duplicate what's already at the top level.
    assert "initial_board" not in race["agents"]["Random"]


def test_build_race_carries_experiment_id_per_agent():
    episode = {
        "initial_board": [[-1]],
        "steps": [],
        "won": False,
        "total_reward": 0.0,
        "steps_taken": 0,
    }

    race = build_race(
        seed=1,
        board_size="1x1",
        mines=0,
        generated_at="2026-01-01T00:00:00+00:00",
        agent_episodes={"DQN": ("exp_C_lr_decay", episode), "Random": (None, episode)},
    )

    assert race["agents"]["DQN"]["experiment_id"] == "exp_C_lr_decay"
    assert race["agents"]["Random"]["experiment_id"] is None


def test_same_seed_gives_every_agent_an_identical_initial_board():
    # The whole premise of a "race" is a fair, shared board -- confirm two
    # structurally different agents (Random vs. CSP) recorded on the same
    # seed really do see the identical mine layout, not just the same RNG
    # draw coincidentally producing similar-looking boards.
    recorder = ReplayRecorder()

    env_a = MinesweeperEnv(rows=5, cols=5, num_mines=5)
    random_episode = recorder.record_episode(env_a, RandomAgent(seed=99).select_action, seed=123)

    csp_agent = CSPAgent(rows=5, cols=5, num_mines=5, seed=99)
    env_b = MinesweeperEnv(rows=5, cols=5, num_mines=5)
    csp_episode = recorder.record_episode(env_b, csp_agent.choose_action, on_episode_start=csp_agent.reset, seed=123)

    assert random_episode["initial_board"] == csp_episode["initial_board"]
