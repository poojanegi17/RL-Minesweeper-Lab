"""Tests for evaluation.evaluate_board_config -- uses random/csp only, since
neither needs a trained checkpoint, so these run without any .pt files present."""

from pathlib import Path

from evaluation.evaluate_board_config import checkpoint_experiment_id, run_evaluation


def test_checkpoint_experiment_id_is_none_at_beginner():
    # Beginner reuses the project's original checkpoints -- no per-level
    # training run to point at.
    assert checkpoint_experiment_id("beginner") is None


def test_checkpoint_experiment_id_points_at_that_levels_standard_run():
    assert checkpoint_experiment_id("intermediate") == "levels/intermediate/standard"
    assert checkpoint_experiment_id("expert") == "levels/expert/standard"


def test_run_evaluation_result_shape_for_random():
    result = run_evaluation("random", "beginner", "sparse", eval_episodes=5, results_dir=Path("results"))

    assert result["agent"] == "Random"
    assert result["level"] == "beginner"
    assert result["density"] == "sparse"
    assert (result["rows"], result["cols"], result["mines"]) == (5, 5, 3)
    assert result["eval_episodes"] == 5
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["checkpoint_source"] is None


def test_run_evaluation_uses_the_right_board_size_per_level():
    result = run_evaluation("csp", "intermediate", "dense", eval_episodes=3, results_dir=Path("results"))

    assert (result["rows"], result["cols"], result["mines"]) == (9, 9, 18)
    assert result["checkpoint_source"] == "levels/intermediate/standard"


def test_run_evaluation_csp_outperforms_random_on_average():
    # A light sanity check that the right agent logic is actually being
    # exercised (CSP deduces, Random doesn't) -- not a strict statistical
    # test, just confirms these aren't secretly running the same policy.
    random_result = run_evaluation("random", "beginner", "standard", eval_episodes=100, seed=1, results_dir=Path("results"))
    csp_result = run_evaluation("csp", "beginner", "standard", eval_episodes=100, seed=1, results_dir=Path("results"))

    assert csp_result["win_rate"] > random_result["win_rate"]
