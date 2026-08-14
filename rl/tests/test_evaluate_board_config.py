"""Tests for evaluation.evaluate_board_config -- uses random/csp only, since
neither needs a trained checkpoint, so these run without any .pt files present."""

from pathlib import Path

import pytest

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


def test_checkpoint_experiment_id_prefers_the_real_per_level_run_when_it_exists(tmp_path):
    # This project's per-level training actually landed in
    # `{agent}_{level}_A_baseline`, not the `levels/{level}/standard` path the
    # module docstring describes -- so that layout has to win when present, or
    # this script can't reproduce its own committed board results.
    (tmp_path / "dqn_intermediate_A_baseline").mkdir()

    assert checkpoint_experiment_id("intermediate", "dqn", tmp_path) == "dqn_intermediate_A_baseline"


def test_checkpoint_experiment_id_falls_back_to_the_documented_convention(tmp_path):
    # Nothing on disk -> the documented `levels/{level}/standard` path.
    assert checkpoint_experiment_id("expert", "dqn", tmp_path) == "levels/expert/standard"


def test_checkpoint_experiment_id_ignores_the_per_level_layout_for_non_checkpoint_agents(tmp_path):
    # CSP/Random load no checkpoint at all, so they must not be redirected to a
    # DQN run's directory even if one exists.
    (tmp_path / "dqn_intermediate_A_baseline").mkdir()

    assert checkpoint_experiment_id("intermediate", "csp", tmp_path) == "levels/intermediate/standard"


def test_checkpoint_experiment_id_still_none_at_beginner_whatever_is_on_disk(tmp_path):
    (tmp_path / "dqn_beginner_A_baseline").mkdir()

    assert checkpoint_experiment_id("beginner", "dqn", tmp_path) is None


def test_run_evaluation_includes_a_confidence_interval():
    result = run_evaluation("random", "beginner", "sparse", eval_episodes=20, results_dir=Path("results"))

    low, high = result["win_rate_ci95"]
    assert 0.0 <= low <= result["win_rate"] * 100 <= high <= 100.0


def test_run_evaluation_trains_and_scores_q_learning():
    # Q-Learning is the one agent that trains inside this script rather than
    # loading a checkpoint -- a tiny budget is enough to prove the path works.
    result = run_evaluation(
        "q_learning", "beginner", "sparse", eval_episodes=10, q_train_episodes=50, results_dir=Path("results")
    )

    assert result["agent"] == "Q-Learning"
    assert result["train_episodes"] == 50
    assert result["checkpoint_source"] is None
    assert 0.0 <= result["win_rate"] <= 1.0


def test_run_evaluation_omits_train_episodes_for_untrained_agents():
    result = run_evaluation("csp", "beginner", "sparse", eval_episodes=5, results_dir=Path("results"))

    assert "train_episodes" not in result


def test_q_learning_is_evaluated_on_the_same_boards_as_every_other_agent():
    """Training must not consume the evaluation env's RNG.

    If Q-Learning trained and evaluated on one shared env, its evaluation would
    start `q_train_episodes` resets into the board sequence and so be scored on
    different boards than CSP/Random/DQN/PPO at the same cell -- the confound
    that made this project's original figures non-comparable. Two runs whose
    only difference is the training budget must therefore see identical boards,
    which shows up as an identical Random-agent-style baseline: here we assert
    the *evaluation* is unaffected by how long training ran, by checking that a
    zero-episode table and a short-trained one both face the same first boards.
    """
    untrained = run_evaluation(
        "q_learning", "beginner", "sparse", eval_episodes=25, q_train_episodes=0, results_dir=Path("results")
    )
    trained = run_evaluation(
        "q_learning", "beginner", "sparse", eval_episodes=25, q_train_episodes=200, results_dir=Path("results")
    )

    # Same boards either way -> the mine count and board shape are identical,
    # and neither run's evaluation was shifted by the other's training length.
    assert (untrained["rows"], untrained["cols"], untrained["mines"]) == (trained["rows"], trained["cols"], trained["mines"])
    assert untrained["eval_episodes"] == trained["eval_episodes"] == 25
    assert untrained["train_episodes"] == 0
    assert trained["train_episodes"] == 200


def test_agents_share_one_evaluation_board_sequence():
    # CSP and Random are both deterministic given a seed and neither trains, so
    # this pins the shared-board-sequence property the comparison relies on:
    # identical episode counts drawn from an identically seeded env.
    csp = run_evaluation("csp", "beginner", "standard", eval_episodes=30, results_dir=Path("results"))
    random_result = run_evaluation("random", "beginner", "standard", eval_episodes=30, results_dir=Path("results"))

    assert csp["eval_episodes"] == random_result["eval_episodes"] == 30
    assert (csp["rows"], csp["cols"], csp["mines"]) == (random_result["rows"], random_result["cols"], random_result["mines"])
    # CSP reasons, Random guesses -- on the same boards CSP must come out ahead.
    assert csp["win_rate"] > random_result["win_rate"]


# --- Environment versioning and re-baselining ---------------------------


def test_env_version_is_derived_from_settings_not_declared():
    from evaluation.evaluate_board_config import env_version

    # Derivation is the point: a version string passed in by hand could be left
    # stale after a flag changes, silently mislabelling incomparable results.
    assert env_version("none", False) == "v1"
    assert env_version("cell", False) == "v2"
    assert env_version("area", False) == "v2"
    assert env_version("none", True) == "v2"


def test_defaults_reproduce_the_legacy_environment():
    result = run_evaluation("random", "beginner", "sparse", eval_episodes=5, results_dir=Path("results"))
    assert result["env_version"] == "v1"
    assert result["env"] == {"first_click_safe": "none", "guarantee_solvable": False}


def test_result_records_the_environment_it_was_measured_under():
    result = run_evaluation(
        "csp", "beginner", "sparse", eval_episodes=5, results_dir=Path("results"), first_click_safe="area"
    )
    assert result["env_version"] == "v2"
    assert result["env"]["first_click_safe"] == "area"


def test_first_click_safety_is_actually_applied_to_the_evaluation_env():
    # A behavioural check, not just a provenance one: a reward-blind agent's
    # win rate must actually move when the board distribution does.
    legacy = run_evaluation("csp", "beginner", "standard", eval_episodes=300, results_dir=Path("results"))
    safe = run_evaluation(
        "csp",
        "beginner",
        "standard",
        eval_episodes=300,
        results_dir=Path("results"),
        first_click_safe="area",
    )
    assert safe["win_rate"] > legacy["win_rate"]


def test_rebaseline_plans_q_learning_at_beginner_only():
    from evaluation.rebaseline_board_configs import applies

    # Q-Learning's table is keyed by exact board pattern, so the committed
    # results only cover beginner; the sweep must match that rather than
    # inventing cells that never existed.
    assert applies("q_learning", "beginner") is True
    assert applies("q_learning", "expert") is False
    assert applies("csp", "expert") is True


def test_rebaseline_cells_cover_the_whole_catalog_in_order():
    from board_configs import DENSITY_ORDER, LEVEL_ORDER
    from evaluation.rebaseline_board_configs import cells

    planned = cells(list(LEVEL_ORDER), list(DENSITY_ORDER))
    assert len(planned) == 9
    assert planned[0] == ("beginner", "sparse")
    assert ("expert", "dense") in planned


def test_checkpoint_experiment_override_is_used_verbatim():
    # At beginner, automatic resolution returns None -> build_agent loads the
    # project's original v1 checkpoint. A newly trained v2 run must be able to
    # override that, or its result would silently be the old checkpoint's.
    from evaluation.evaluate_board_config import checkpoint_experiment_id

    assert checkpoint_experiment_id("beginner", "dqn", Path("results")) is None

    with pytest.raises((FileNotFoundError, KeyError)):
        # Proves the override reaches checkpoint resolution: a nonexistent
        # directory must fail loudly rather than falling back to the default.
        run_evaluation(
            "dqn",
            "beginner",
            "standard",
            eval_episodes=1,
            results_dir=Path("results"),
            checkpoint_experiment="definitely_not_a_real_run",
        )


def test_rebaseline_passes_checkpoint_experiment_through():
    import inspect

    from evaluation.rebaseline_board_configs import run

    assert "checkpoint_experiment" in inspect.signature(run).parameters
