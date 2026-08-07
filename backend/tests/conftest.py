"""Shared pytest fixtures: a crafted fake results directory + a TestClient wired to it.

Tests never touch the real rl/results/ -- that data changes as experiments
are run and isn't guaranteed to contain any particular shape, so every test
here builds its own small, controlled fixture directory instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.race_loader import RaceLoader, get_race_loader
from app.services.replay_loader import ReplayLoader, get_replay_loader
from app.services.results_loader import ResultsLoader, get_results_loader


def _write_dqn_experiment(
    base: Path, name: str, episodes: int = 100, *, win_rate: float = 0.1, with_checkpoints: bool = True
) -> None:
    """A well-formed DQN experiment directory, mirroring dqn_experiment.py's output."""
    exp_dir = base / name
    exp_dir.mkdir(parents=True)
    if with_checkpoints:
        checkpoint_dir = exp_dir / f"checkpoints_{episodes}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "best_model.pt").write_bytes(b"")
        (checkpoint_dir / "final_model.pt").write_bytes(b"")
    history = [
        {
            "episode": i + 1,
            "total_reward": -1.0 if i % 2 == 0 else 1.0,
            "steps": 3,
            "won": i % 5 == 0,
            "epsilon": max(0.05, 1.0 - i * 0.01),
            "lr": 1e-4,
            "loss": None if i < 2 else 1.5,
            "avg_q": None if i < 2 else 0.3,
            "max_q": None if i < 2 else 0.5,
            "grad_norm": None if i < 2 else 2.0,
            "td_error_mean": None if i < 2 else 0.2,
            "td_error_max": None if i < 2 else 0.4,
            "reward_rolling_mean": 0.0,
            "win_rate_rolling": 0.1,
        }
        for i in range(episodes)
    ]
    (exp_dir / f"dqn_history_{episodes}.json").write_text(json.dumps(history))
    summary = {
        "episodes": episodes,
        "lr": 1e-4,
        "batch_size": 64,
        "target_update_every": 25,
        "network_size": "default",
        "checkpoint_every": 50,
        "used_checkpoint": "best_model.pt",
        "best_checkpoint_metadata": {"episode": 50, "win_rate": 0.2, "average_reward": 1.0, "timestamp": "2026-01-01T00:00:00+00:00"},
        "train_seconds": 12.3,
        "eval_episodes": 200,
        "win_rate": win_rate,
        "avg_episode_length": 4.2,
        "avg_reward": -2.0,
        "failures": 180,
    }
    (exp_dir / f"dqn_history_{episodes}_summary.json").write_text(json.dumps(summary))


def _write_ppo_experiment(base: Path, name: str, episodes: int = 100) -> None:
    """A well-formed PPO experiment directory, mirroring ppo_experiment.py's output."""
    exp_dir = base / name
    exp_dir.mkdir(parents=True)
    history = [
        {
            "episode": i + 1,
            "total_reward": -1.0 if i % 2 == 0 else 1.0,
            "steps": 3,
            "won": i % 6 == 0,
            "lr": 3e-4,
            "policy_loss": None if i < 2 else -0.01,
            "value_loss": None if i < 2 else 5.0,
            "entropy": None if i < 2 else 1.2,
            "approx_kl": None if i < 2 else 0.001,
            "clip_fraction": None if i < 2 else 0.05,
            "total_loss": None if i < 2 else 2.5,
            "explained_variance": None if i < 2 else 0.08,
        }
        for i in range(episodes)
    ]
    (exp_dir / f"ppo_history_{episodes}.json").write_text(json.dumps(history))
    summary = {
        "episodes": episodes,
        "reward_mode": "shaped",
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "rollout_length": 256,
        "ppo_epochs": 4,
        "batch_size": 64,
        "checkpoint_every": 50,
        "used_checkpoint": "best_policy.pt",
        "best_checkpoint_metadata": None,
        "train_seconds": 5.0,
        "eval_episodes": 200,
        "win_rate": 0.05,
        "avg_episode_length": 3.8,
        "avg_reward": -3.0,
        "failures": 190,
    }
    (exp_dir / f"ppo_history_{episodes}_summary.json").write_text(json.dumps(summary))


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """A fixture results directory with: one clean DQN experiment, one clean PPO
    experiment, a loose history-only file (no summary), a malformed summary, an
    unrelated empty subdirectory, and a two-member DQN ablation family --
    covering every case the service needs to handle."""
    _write_dqn_experiment(tmp_path, "exp_test_dqn", episodes=100)
    _write_ppo_experiment(tmp_path, "ppo_exp_test", episodes=100)

    # Loose top-level history file with no summary (mirrors evaluate_agents.py's output).
    loose_history = [{"episode": 1, "total_reward": 1.0, "steps": 2, "won": True}]
    (tmp_path / "dqn_evaluate_agents_history.json").write_text(json.dumps(loose_history))

    # Malformed summary paired with an otherwise-valid history file.
    (tmp_path / "dqn_history_broken.json").write_text(json.dumps(loose_history))
    (tmp_path / "dqn_history_broken_summary.json").write_text("{not valid json")

    # Unrelated empty directory (e.g. a stray checkpoints_* dir) must be ignored, not error.
    (tmp_path / "checkpoints_something").mkdir()

    # A two-member ablation family ("exp_A_baseline"/"exp_B_variant" both parse
    # to group "exp") for testing GET /api/experiments/{id}/ablation and grouping.
    _write_dqn_experiment(tmp_path, "exp_A_baseline", episodes=50, win_rate=0.02, with_checkpoints=False)
    _write_dqn_experiment(tmp_path, "exp_B_variant", episodes=50, win_rate=0.05, with_checkpoints=False)

    # An artifact whose stem matches neither the "dqn"/"ppo" naming convention
    # nor the ablation id pattern -- the "unnamed experiment" case from the bug
    # report: no derivable agent/algorithm identity beyond its raw filename.
    unnamed_history = [{"episode": 1, "total_reward": 1.0, "steps": 2, "won": False}]
    (tmp_path / "misc_run_history.json").write_text(json.dumps(unnamed_history))

    _write_replays(tmp_path / "replays")
    _write_races(tmp_path / "races")
    _write_level_data(tmp_path / "levels" / "intermediate" / "standard")

    return tmp_path


def _write_replay(path: Path, *, replay_id: str, agent: str, won: bool, reasoning: dict | None) -> None:
    """A well-formed replay, mirroring evaluation.replay.build_replay's output shape."""
    replay = {
        "id": replay_id,
        "agent": agent,
        "experiment_id": None,
        "board_size": "3x3",
        "mines": 1,
        "seed": 42,
        "episode_number": 1,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "initial_board": [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        "steps": [
            {
                "step": 1,
                "board_state": [[0, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                "action": {"row": 0, "col": 0},
                "reward": 1.0,
                "done": False,
                "reasoning": reasoning,
            },
            {
                "step": 2,
                "board_state": [[0, 1, -1], [-1, -1, -1], [-1, -1, -1]] if won else [[0, 0, -1], [-1, -1, -1], [-1, -1, -1]],
                "action": {"row": 0, "col": 1},
                "reward": 10.0 if won else -10.0,
                "done": True,
                "reasoning": reasoning,
            },
        ],
        "won": won,
        "total_reward": 11.0 if won else -9.0,
        "steps_taken": 2,
    }
    path.write_text(json.dumps(replay))


def _write_replays(replays_dir: Path) -> None:
    replays_dir.mkdir(parents=True)
    _write_replay(replays_dir / "dqn_episode_1.json", replay_id="dqn_episode_1", agent="DQN", won=False, reasoning={"q_value": 0.82})
    _write_replay(replays_dir / "csp_episode_1.json", replay_id="csp_episode_1", agent="CSP", won=True, reasoning={
        "deduction_type": "safe", "constraint_cells": [[0, 0]], "remaining_mines": 0, "mine_probability": None,
        "inference": "Cell (0, 0) is safe.",
    })
    _write_replay(replays_dir / "random_episode_1.json", replay_id="random_episode_1", agent="Random", won=False, reasoning=None)
    # Malformed: invalid JSON.
    (replays_dir / "broken_episode.json").write_text("{not valid json")
    # Malformed: valid JSON, but missing required fields (e.g. no "steps").
    (replays_dir / "incomplete_episode.json").write_text(json.dumps({"id": "incomplete_episode", "agent": "DQN"}))


def _write_races(races_dir: Path) -> None:
    races_dir.mkdir(parents=True)
    race = {
        "id": "race_1",
        "seed": 1,
        "board_size": "3x3",
        "mines": 1,
        "turn_order": ["Random", "CSP", "DQN"],
        "generated_at": "2026-01-01T00:00:00+00:00",
        "initial_board": [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        "turns": [
            {
                "turn": 1,
                "agent": "Random",
                "action": {"row": 0, "col": 0},
                "board_state": [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                "eliminated": True,
                "reasoning": None,
            },
            {
                "turn": 2,
                "agent": "CSP",
                "action": {"row": 1, "col": 1},
                "board_state": [[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]],
                "eliminated": False,
                "reasoning": {"deduction_type": "probability_guess", "constraint_cells": None, "remaining_mines": None, "mine_probability": 0.2, "inference": "No safe cell deduced; guessing the lowest mine-probability cell (1, 1)."},
            },
            {
                "turn": 3,
                "agent": "DQN",
                "action": {"row": 2, "col": 2},
                "board_state": [[-1, -1, -1], [-1, 2, -1], [-1, -1, 1]],
                "eliminated": False,
                "reasoning": {"q_value": 0.5},
            },
        ],
        "won": False,
        "total_turns": 3,
        "surviving_agents": ["CSP", "DQN"],
        "eliminated_agents": {"Random": 1},
    }
    (races_dir / "race_1.json").write_text(json.dumps(race))

    # Malformed: missing required fields (e.g. no "turns").
    (races_dir / "incomplete_race.json").write_text(json.dumps({"id": "incomplete_race", "seed": 2}))
    # Malformed: invalid JSON.
    (races_dir / "broken_race.json").write_text("{not valid json")


def _write_level_data(level_density_dir: Path) -> None:
    """A non-default (level, density) directory -- CSP has a board result
    (mirroring evaluate_board_config.py's output), DQN deliberately doesn't
    (the "not trained yet at this level" case), plus one replay and one race
    so level-scoped replay/race listing has something real to find."""
    level_density_dir.mkdir(parents=True)
    csp_result = {
        "agent": "CSP",
        "level": "intermediate",
        "density": "standard",
        "rows": 9,
        "cols": 9,
        "mines": 12,
        "eval_episodes": 200,
        "win_rate": 0.625,
        "avg_episode_length": 16.8,
        "avg_reward": -1.2,
        "failures": 75,
        "checkpoint_source": None,
    }
    (level_density_dir / "csp_board_result.json").write_text(json.dumps(csp_result))

    (level_density_dir / "replays").mkdir()
    _write_replay(
        level_density_dir / "replays" / "csp_episode_1.json",
        replay_id="csp_episode_1",
        agent="CSP",
        won=True,
        reasoning={"deduction_type": "safe", "constraint_cells": [[0, 0]], "remaining_mines": 0, "mine_probability": None, "inference": "Cell (0, 0) is safe."},
    )


@pytest.fixture()
def client(results_dir: Path) -> TestClient:
    """A TestClient wired to `results_dir` (and its `replays/`/`races/` subfolders) instead of the real rl/results/."""
    app.dependency_overrides[get_results_loader] = lambda: ResultsLoader(results_dir)
    app.dependency_overrides[get_replay_loader] = lambda: ReplayLoader(results_dir / "replays")
    app.dependency_overrides[get_race_loader] = lambda: RaceLoader(results_dir / "races")
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def empty_client(tmp_path: Path) -> TestClient:
    """A TestClient wired to a results directory (and replays/races subfolders) that don't exist on disk."""
    missing_dir = tmp_path / "does_not_exist"
    app.dependency_overrides[get_results_loader] = lambda: ResultsLoader(missing_dir)
    app.dependency_overrides[get_replay_loader] = lambda: ReplayLoader(missing_dir / "replays")
    app.dependency_overrides[get_race_loader] = lambda: RaceLoader(missing_dir / "races")
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
