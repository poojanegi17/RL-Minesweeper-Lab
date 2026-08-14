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
    # The PPO Beginner family's *shipped* members are E/F/G -- the matched-budget
    # re-runs. A-D (including the previously asserted `ppo_exp_C_shaped`) trained
    # against ~7,000 gradient updates versus the 123,000 every current PPO run
    # uses, so they measured a starved configuration rather than the agent and
    # are no longer part of `results_public/`.
    assert "ppo_exp_F_shaped_matched" in ids
    assert len(experiments) >= 14  # matches the project-scale figure in the README

    # `races/` holds JSON but is not an experiment directory -- it has its own
    # loader and endpoint. Discovered as experiments its six files became
    # phantom rows with `agent: "Unknown"`; see `NON_EXPERIMENT_DIRS`.
    assert not any(experiment_id.startswith("race") for experiment_id in ids)


def test_replay_loader_discovers_real_replays() -> None:
    loader = ReplayLoader(settings.results_dir / "replays")

    replays = loader.list_replays()

    # Every agent that can be replayed is present. The exact per-agent count is
    # deliberately not asserted: `curate_replays.py` keeps up to N wins plus a
    # spread of losses, so an agent that wins rarely at this board keeps fewer
    # files than one that wins often. Pinning a total would make an honest
    # curation result look like a regression.
    assert {r["agent"] for r in replays} == {"Random", "CSP", "DQN", "PPO"}
    assert len(replays) >= 20
    # Q-Learning is intentionally absent -- it persists no checkpoint anywhere
    # in the project, so there is nothing to replay against.
    assert "Q-Learning" not in {r["agent"] for r in replays}


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
    # The DQN Beginner ablation: A-E (the original stability variants) plus F
    # (the masked-bootstrap bug fix). Asserted as a lower bound with the known
    # members checked explicitly, so adding a variant doesn't break this test
    # while a *missing* one still does.
    family = dqn_family.json()
    assert family["run_count"] >= 6
    run_ids = {run["id"] for run in family["runs"]}
    assert {
        "exp_A_baseline",
        "exp_B_checkpoint",
        "exp_C_lr_decay",
        "exp_D_small_net",
        "exp_E_combined",
        "exp_F_masked_target",
    } <= run_ids

    metrics = client.get("/api/experiments/exp_C_lr_decay/metrics")
    assert metrics.status_code == 200
    body = metrics.json()
    # `total_episodes` is the length of the *run*, read from its summary --
    # committed histories are subsampled for size (see
    # `rl/evaluation/compact_public_histories.py`), so the number of rows served
    # is the chart's resolution and is deliberately smaller.
    assert body["total_episodes"] == 25000
    assert 0 < len(body["episodes"]) <= 25000
    # Subsampling keeps real rows with their real episode numbers, and pins the
    # endpoints -- so the curve still spans the whole run.
    assert body["episodes"][0]["episode"] == 1
    assert body["episodes"][-1]["episode"] == 25000

    replays = client.get("/api/replays")
    assert replays.status_code == 200
    assert len(replays.json()) >= 20  # see test_replay_loader_discovers_real_replays
