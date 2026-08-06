"""Integration checks against the real, committed `rl/results_public/` --
distinct from every other test file, which builds its own synthetic fixture
directory and never touches disk-real data. This is the one place that
proves the actual deployment artifact set (not a crafted stand-in) is
structurally valid and serves correctly through the full API, end to end,
using the app's real (non-overridden) dependency wiring.

If `rl/results_public/` is ever accidentally deleted, renamed, or stops
containing what the loaders expect, these fail loudly here instead of
surfacing as a silently-empty production site.
"""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.replay_loader import ReplayLoader
from app.services.results_loader import ResultsLoader


def test_results_public_directory_exists() -> None:
    assert settings.results_dir.is_dir()


def test_results_loader_discovers_real_experiments() -> None:
    loader = ResultsLoader(settings.results_dir)

    experiments = loader.list_experiments()
    ids = {e.id for e in experiments}

    assert "exp_A_baseline" in ids
    assert "ppo_exp_C_shaped" in ids
    assert len(experiments) >= 14  # matches the project-scale figure in the README


def test_replay_loader_discovers_real_replays() -> None:
    loader = ReplayLoader(settings.results_dir / "replays")

    replays = loader.list_replays()

    assert len(replays) == 40
    assert {r["agent"] for r in replays} == {"Random", "CSP", "DQN", "PPO"}


def test_no_checkpoint_or_csv_files_shipped() -> None:
    # The whole point of results_public/: chart-ready JSON only, never the
    # large .pt checkpoints or the CSVs results_loader.py never reads anyway.
    leaked = list(settings.results_dir.rglob("*.pt")) + list(settings.results_dir.rglob("*.csv"))

    assert leaked == []


def test_full_api_serves_real_data_end_to_end() -> None:
    # No dependency_overrides here, unlike every other test file -- this
    # exercises the exact wiring a deployed instance would actually use.
    client = TestClient(app)

    leaderboard = client.get("/api/leaderboard")
    assert leaderboard.status_code == 200
    assert len(leaderboard.json()) == 5

    experiments = client.get("/api/experiments")
    assert experiments.status_code == 200
    assert len(experiments.json()) > 0

    dqn_family = client.get("/api/experiments/exp")
    assert dqn_family.status_code == 200
    assert dqn_family.json()["run_count"] == 5

    metrics = client.get("/api/experiments/exp_C_lr_decay/metrics")
    assert metrics.status_code == 200
    assert len(metrics.json()["episodes"]) == 25000

    replays = client.get("/api/replays")
    assert replays.status_code == 200
    assert len(replays.json()) == 40
