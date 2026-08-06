"""Tests for backend configuration (app/config.py) -- the results-directory
default and its `MINESWEEPER_RESULTS_DIR` environment-variable override.

`Settings()` re-reads environment variables every time it's constructed, so
the override test below builds a fresh instance rather than relying on the
app's already-imported `settings` singleton (which was constructed once, at
import time, before any test could set an env var -- monkeypatching after
that point wouldn't change it).
"""

from pathlib import Path

import pytest

from app.config import Settings, settings
from app.services.replay_loader import get_replay_loader
from app.services.results_loader import get_results_loader

# backend/tests/test_config.py -> backend/tests -> backend -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_DEFAULT = _REPO_ROOT / "rl" / "results_public"


def test_results_dir_defaults_to_results_public() -> None:
    # The committed, deployment-safe subset -- not the gitignored,
    # never-deployed rl/results/.
    assert Settings().results_dir == _EXPECTED_DEFAULT


def test_results_dir_default_is_absolute_and_not_cwd_dependent() -> None:
    # Resolved from this file's own location, not the process's working
    # directory -- must work right after cloning, launched from anywhere.
    assert Settings().results_dir.is_absolute()


def test_results_dir_can_be_overridden_via_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override = tmp_path / "some_other_results_dir"
    monkeypatch.setenv("MINESWEEPER_RESULTS_DIR", str(override))

    assert Settings().results_dir == override


def test_app_settings_singleton_points_at_results_public() -> None:
    # The actual object get_results_loader()/get_replay_loader() close over.
    assert settings.results_dir == _EXPECTED_DEFAULT


def test_get_results_loader_reads_from_configured_default() -> None:
    loader = get_results_loader()

    assert loader.results_dir == _EXPECTED_DEFAULT


def test_get_replay_loader_reads_from_results_dir_slash_replays() -> None:
    loader = get_replay_loader()

    assert loader.replays_dir == _EXPECTED_DEFAULT / "replays"
