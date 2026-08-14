"""Tests for evaluation.reevaluate_checkpoints.

Covers the statistics helpers and run discovery directly. `evaluate_checkpoint`
itself needs a real `.pt` file, so it's exercised through a tiny DQN agent
whose weights this test writes itself rather than depending on any committed
checkpoint (`rl/results/` is gitignored, so a test that required one would
pass locally and fail on a fresh clone).
"""

import json

import pytest

from agents.dqn_agent import DQNAgent
from evaluation.apply_reevaluation import build_evaluation_block, resolve_deployed
from evaluation.reevaluate_checkpoints import (
    DEFAULT_BOARD,
    evaluate_checkpoint,
    find_loose_checkpoint_dirs,
    find_runs,
    two_proportion_test,
    wilson_interval,
)


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(40, 2000)
    assert low < 2.0 < high


def test_wilson_interval_lower_bound_never_negative_for_rare_events():
    # The normal approximation would put this below zero, which isn't a
    # meaningful bound for a count -- this is why Wilson is used.
    low, _ = wilson_interval(1, 2000)
    assert low >= 0.0


def test_wilson_interval_narrows_as_sample_size_grows():
    narrow = wilson_interval(20, 2000)
    wide = wilson_interval(2, 200)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_handles_zero_wins():
    low, high = wilson_interval(0, 2000)
    assert low == 0.0
    assert 0.0 < high < 1.0


@pytest.mark.parametrize("n", [20, 50, 200, 2000])
def test_wilson_interval_never_reports_a_negative_lower_bound(n):
    # At wins == 0 the two terms cancel only up to floating point, so small n
    # used to yield a tiny negative bound that rendered as "-0.00%".
    low, high = wilson_interval(0, n)
    assert low == 0.0
    assert high <= 100.0


def test_wilson_interval_upper_bound_capped_at_100():
    _, high = wilson_interval(5, 5)
    assert high <= 100.0


def test_wilson_interval_handles_zero_episodes():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_two_proportion_test_finds_no_difference_between_equal_counts():
    z, p = two_proportion_test(30, 30, 2000)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_two_proportion_test_detects_a_large_difference():
    # 11/2000 vs 58/2000 -- the baseline-vs-6000-episode comparison.
    _, p = two_proportion_test(11, 58, 2000)
    assert p < 0.001


def test_two_proportion_test_reports_small_differences_as_insignificant():
    # 4/200 vs 7/200 -- the original underpowered comparison this script exists
    # to replace. Must not read as significant.
    _, p = two_proportion_test(4, 7, 200)
    assert p > 0.05


def test_two_proportion_test_sign_follows_the_second_argument():
    z_up, _ = two_proportion_test(10, 40, 2000)
    z_down, _ = two_proportion_test(40, 10, 2000)
    assert z_up > 0 > z_down


def test_two_proportion_test_handles_two_zero_counts():
    z, p = two_proportion_test(0, 0, 2000)
    assert (z, p) == (0.0, 1.0)


def _write_run(root, name, agent_prefix, checkpoint_name, summary_extra=None):
    """Create a minimal experiment directory with a summary and a checkpoint."""
    run_dir = root / name
    checkpoint_dir = run_dir / "checkpoints_10"
    checkpoint_dir.mkdir(parents=True)

    summary = {"episodes": 10, "win_rate": 0.02, "eval_episodes": 200}
    summary.update(summary_extra or {})
    (run_dir / f"{agent_prefix}_history_10_summary.json").write_text(json.dumps(summary))

    agent = DQNAgent(rows=5, cols=5, seed=0)
    agent.save_checkpoint(checkpoint_dir / checkpoint_name, episode=10, win_rate=0.02, average_reward=-6.0)
    return run_dir


def test_find_runs_discovers_a_run_and_defaults_the_board(tmp_path):
    _write_run(tmp_path, "exp_A_baseline", "dqn", "final_model.pt")

    runs = find_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0]["name"] == "exp_A_baseline"
    assert runs[0]["agent"] == "dqn"
    # Earliest Beginner summaries record no rows/cols/mines at all.
    assert (runs[0]["rows"], runs[0]["cols"], runs[0]["mines"]) == DEFAULT_BOARD


def test_find_runs_reads_a_recorded_board_size_when_present(tmp_path):
    _write_run(tmp_path, "dqn_intermediate_A_baseline", "dqn", "final_model.pt",
               summary_extra={"rows": 9, "cols": 9, "mines": 12})

    runs = find_runs(tmp_path)

    assert (runs[0]["rows"], runs[0]["cols"], runs[0]["mines"]) == (9, 9, 12)


def test_find_runs_classifies_ppo_from_the_summary_filename(tmp_path):
    _write_run(tmp_path, "ppo_exp_A_baseline", "ppo", "final_policy.pt")

    assert find_runs(tmp_path)[0]["agent"] == "ppo"


def test_find_runs_skips_a_run_with_no_checkpoint_directory(tmp_path):
    run_dir = tmp_path / "exp_no_checkpoints"
    run_dir.mkdir()
    (run_dir / "dqn_history_10_summary.json").write_text(json.dumps({"episodes": 10}))

    assert find_runs(tmp_path) == []


def test_find_runs_honours_the_only_filter(tmp_path):
    _write_run(tmp_path, "exp_A_baseline", "dqn", "final_model.pt")
    _write_run(tmp_path, "dqn_expert_A_baseline", "dqn", "final_model.pt",
               summary_extra={"rows": 16, "cols": 16, "mines": 40})

    names = [run["name"] for run in find_runs(tmp_path, only="expert")]

    assert names == ["dqn_expert_A_baseline"]


def test_evaluate_checkpoint_returns_a_scored_result(tmp_path):
    run_dir = _write_run(tmp_path, "exp_A_baseline", "dqn", "final_model.pt")
    run = find_runs(tmp_path)[0]

    result = evaluate_checkpoint(run, run_dir / "checkpoints_10" / "final_model.pt", episodes=5)

    assert result["episodes"] == 5
    assert 0 <= result["wins"] <= 5
    assert result["win_rate"] == result["wins"] / 5
    assert result["win_rate_ci95"][0] <= result["win_rate"] * 100 <= result["win_rate_ci95"][1]
    assert result["avg_episode_length"] > 0
    assert result["checkpoint_metadata"]["episode"] == 10


def test_find_loose_checkpoint_dirs_picks_up_evaluate_agents_output(tmp_path):
    # evaluate_agents.py writes weights with no summary JSON at all, so
    # find_runs can't see them -- but they produce the README's matched-budget
    # figures and must still be scoreable.
    (tmp_path / "checkpoints_evaluate_agents").mkdir()
    (tmp_path / "checkpoints_ppo_evaluate_agents").mkdir()

    found = {run["name"]: run for run in find_loose_checkpoint_dirs(tmp_path)}

    assert set(found) == {"checkpoints_evaluate_agents", "checkpoints_ppo_evaluate_agents"}
    assert found["checkpoints_evaluate_agents"]["agent"] == "dqn"
    assert found["checkpoints_ppo_evaluate_agents"]["agent"] == "ppo"
    # No summary means no recorded deployment choice to preserve.
    assert found["checkpoints_evaluate_agents"]["reported_used_checkpoint"] is None


def test_find_loose_checkpoint_dirs_ignores_experiment_directories(tmp_path):
    _write_run(tmp_path, "exp_A_baseline", "dqn", "final_model.pt")

    # exp_A_baseline is a run directory, not a top-level checkpoints_* dir --
    # find_runs owns it, so this must not double-report it.
    assert find_loose_checkpoint_dirs(tmp_path) == []


def test_find_loose_checkpoint_dirs_returns_empty_for_missing_directory(tmp_path):
    assert find_loose_checkpoint_dirs(tmp_path / "nope") == []


@pytest.mark.parametrize(
    "used_checkpoint,expected",
    [
        ("best_model.pt", "best"),
        ("best_policy.pt", "best"),
        ("final_in_memory_weights", "final"),
    ],
)
def test_resolve_deployed_maps_used_checkpoint_to_the_scored_label(used_checkpoint, expected):
    assert resolve_deployed(used_checkpoint, {"final": {}, "best": {}}) == expected


def test_resolve_deployed_returns_none_for_an_unknown_value():
    # Must be skipped loudly rather than silently defaulting to wrong weights.
    assert resolve_deployed("something_else.pt", {"final": {}, "best": {}}) is None
    assert resolve_deployed(None, {"final": {}, "best": {}}) is None


def test_resolve_deployed_returns_none_when_that_checkpoint_was_not_scored():
    assert resolve_deployed("best_model.pt", {"final": {}}) is None


def test_build_evaluation_block_derives_failures_from_wins():
    scored = {
        "wins": 11,
        "win_rate": 0.0055,
        "win_rate_ci95": [0.307, 0.982],
        "avg_reward": -6.87654321,
        "avg_episode_length": 4.0135,
    }

    block = build_evaluation_block(scored, episodes=2000)

    assert block["failures"] == 1989
    assert block["eval_episodes"] == 2000
    assert block["win_rate"] == 0.0055
    assert block["avg_reward"] == pytest.approx(-6.8765)
    assert block["evaluation_source"] == "reevaluate_checkpoints.py"


def test_evaluate_checkpoint_is_deterministic_for_the_same_weights(tmp_path):
    run_dir = _write_run(tmp_path, "exp_A_baseline", "dqn", "final_model.pt")
    run = find_runs(tmp_path)[0]
    path = run_dir / "checkpoints_10" / "final_model.pt"

    first = evaluate_checkpoint(run, path, episodes=10)
    second = evaluate_checkpoint(run, path, episodes=10)

    # Greedy policy plus a fixed EVAL_SEED means every model faces an identical
    # board sequence -- the property that makes rows of one report comparable.
    assert first["wins"] == second["wins"]
    assert first["avg_episode_length"] == second["avg_episode_length"]
