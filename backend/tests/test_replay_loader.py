"""Direct unit tests for the ReplayLoader service (below the API layer)."""

from pathlib import Path

import pytest

from app.services.replay_loader import MalformedReplayError, ReplayLoader, ReplayNotFoundError


def test_list_replays_on_missing_directory_returns_empty_list(tmp_path: Path) -> None:
    loader = ReplayLoader(tmp_path / "does_not_exist")

    assert loader.list_replays() == []


def test_list_replays_on_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    loader = ReplayLoader(tmp_path)

    assert loader.list_replays() == []


def test_get_replay_on_missing_directory_raises_not_found(tmp_path: Path) -> None:
    loader = ReplayLoader(tmp_path / "does_not_exist")

    with pytest.raises(ReplayNotFoundError):
        loader.get_replay("anything")


def test_list_replays_skips_malformed_files(results_dir: Path) -> None:
    loader = ReplayLoader(results_dir / "replays")

    records = loader.list_replays()  # must not raise despite broken_episode.json / incomplete_episode.json

    ids = {r["id"] for r in records}
    assert "broken_episode" not in ids
    assert "incomplete_episode" not in ids
    assert "dqn_episode_1" in ids


def test_get_replay_on_invalid_json_raises_malformed(results_dir: Path) -> None:
    loader = ReplayLoader(results_dir / "replays")

    with pytest.raises(MalformedReplayError):
        loader.get_replay("broken_episode")


def test_get_replay_on_missing_required_fields_raises_malformed(results_dir: Path) -> None:
    loader = ReplayLoader(results_dir / "replays")

    with pytest.raises(MalformedReplayError):
        loader.get_replay("incomplete_episode")


def test_get_replay_returns_full_raw_dict(results_dir: Path) -> None:
    loader = ReplayLoader(results_dir / "replays")

    replay = loader.get_replay("dqn_episode_1")

    assert replay["agent"] == "DQN"
    assert len(replay["steps"]) == 2


def test_get_replay_on_unknown_id_raises_not_found(results_dir: Path) -> None:
    loader = ReplayLoader(results_dir / "replays")

    with pytest.raises(ReplayNotFoundError):
        loader.get_replay("nope")
