"""Tests for GET /api/experiments and GET /api/experiments/{id}.

`GET /api/experiments` returns one row per *family* of related runs (2+
sharing an ablation-style id prefix, e.g. "exp") or per standalone run
otherwise -- see `results_loader.group_experiments`. `GET /api/experiments/{id}`
resolves `{id}` as a family id first, falling back to an individual run id
(the pre-grouping behavior, preserved for backward compatibility) when no
family matches.
"""

import pytest
from fastapi.testclient import TestClient


# --- list: discovery & grouping -----------------------------------------------------


def test_list_experiments_discovers_all_fixture_experiments(client: TestClient) -> None:
    response = client.get("/api/experiments")

    assert response.status_code == 200
    ids = {e["id"] for e in response.json()}
    # exp_A_baseline/exp_B_variant are grouped under "exp", not listed individually.
    assert "exp" in ids
    assert "exp_A_baseline" not in ids
    assert "exp_B_variant" not in ids
    # Standalone runs and the loose/unnamed artifacts are listed under their own id.
    assert "exp_test_dqn" in ids
    assert "ppo_exp_test" in ids
    assert "dqn_evaluate_agents_history" in ids
    assert "misc_run_history" in ids
    # dqn_history_broken has a malformed summary and must be silently skipped.
    assert "dqn_history_broken" not in ids


def test_list_experiments_empty_when_results_dir_missing(empty_client: TestClient) -> None:
    response = empty_client.get("/api/experiments")

    assert response.status_code == 200
    assert response.json() == []


# --- list: grouped family summary -----------------------------------------------------


def test_grouped_family_has_both_runs_with_variant_aware_titles(client: TestClient) -> None:
    response = client.get("/api/experiments")
    exp = next(e for e in response.json() if e["id"] == "exp")

    assert exp["run_count"] == 2
    assert exp["agent"] == "DQN"
    assert exp["algorithm"] == "Double DQN"
    assert exp["title"] == "DQN - 2 runs"
    run_ids = {r["id"] for r in exp["runs"]}
    assert run_ids == {"exp_A_baseline", "exp_B_variant"}

    runs_by_id = {r["id"]: r for r in exp["runs"]}
    # Variant-derived titles, not the identical "DQN - 50 episodes" both
    # siblings would render under the pre-grouping episode-count formula.
    assert runs_by_id["exp_A_baseline"]["title"] == "DQN - Baseline"
    assert runs_by_id["exp_A_baseline"]["variant"] == "baseline"
    assert runs_by_id["exp_B_variant"]["title"] == "DQN - Variant"
    assert runs_by_id["exp_B_variant"]["variant"] == "variant"


def test_grouped_family_episodes_range_and_description(client: TestClient) -> None:
    response = client.get("/api/experiments")
    exp = next(e for e in response.json() if e["id"] == "exp")

    assert exp["episodes_range"] == [50, 50]  # both siblings trained for 50 episodes
    assert exp["board"] == "5x5"
    assert exp["mines"] == 5
    assert "Baseline" in exp["description"]
    assert "Variant" in exp["description"]


def test_grouped_family_metrics_summary_aggregates_across_runs(client: TestClient) -> None:
    response = client.get("/api/experiments")
    exp = next(e for e in response.json() if e["id"] == "exp")
    metrics = exp["metrics_summary"]

    assert metrics["runs_with_metrics"] == 2
    assert metrics["best_run_id"] == "exp_B_variant"  # win_rate 0.05 > 0.02
    assert metrics["best_win_rate"] == pytest.approx(0.05)
    assert metrics["avg_win_rate"] == pytest.approx(0.035)


def test_grouped_family_techniques_are_unioned_across_runs(client: TestClient) -> None:
    response = client.get("/api/experiments")
    exp = next(e for e in response.json() if e["id"] == "exp")

    # Both siblings are DQN (always "Double DQN") with used_checkpoint set,
    # so "Best-checkpoint deployment" fires for both -- union, not duplicated.
    assert exp["techniques"] == ["Double DQN", "Best-checkpoint deployment"]


def test_standalone_entry_has_run_count_one_and_no_variant(client: TestClient) -> None:
    response = client.get("/api/experiments")
    dqn = next(e for e in response.json() if e["id"] == "exp_test_dqn")

    assert dqn["run_count"] == 1
    assert dqn["episodes_range"] == [100, 100]
    assert dqn["title"] == "DQN - 100 episodes"
    assert dqn["runs"][0]["variant"] is None
    assert dqn["runs"][0]["title"] == "DQN - 100 episodes"
    assert "Double DQN" in dqn["techniques"]
    assert "Best-checkpoint deployment" in dqn["techniques"]


def test_standalone_entry_ppo_summary_fields(client: TestClient) -> None:
    response = client.get("/api/experiments")
    ppo = next(e for e in response.json() if e["id"] == "ppo_exp_test")

    assert ppo["agent"] == "PPO"
    assert ppo["algorithm"].startswith("PPO")
    assert "Clipped-surrogate PPO" in ppo["techniques"]
    assert "Reward shaping" in ppo["techniques"]  # reward_mode == "shaped" in the fixture


def test_loose_history_file_still_gets_derived_fields_without_a_summary(client: TestClient) -> None:
    response = client.get("/api/experiments")
    loose = next(e for e in response.json() if e["id"] == "dqn_evaluate_agents_history")

    assert loose["title"] == "DQN - 1 episodes"  # episode count derived from row count, no summary needed
    assert loose["runs"][0]["metrics_available"] is True
    assert loose["techniques"] == ["Double DQN"]  # no summary -> none of the conditional flags can fire
    assert loose["metrics_summary"]["runs_with_metrics"] == 0  # no win_rate to read


# --- list: unnamed artifacts -----------------------------------------------------


def test_unnamed_artifact_gets_unknown_agent_and_still_appears(client: TestClient) -> None:
    response = client.get("/api/experiments")
    unnamed = next(e for e in response.json() if e["id"] == "misc_run_history")

    assert unnamed["agent"] == "Unknown"
    assert unnamed["algorithm"] == "Unknown"
    assert unnamed["title"] == "Unknown - 1 episodes"
    assert unnamed["techniques"] == []  # no technique rules registered for an unrecognized agent
    assert unnamed["run_count"] == 1
    assert unnamed["runs"][0]["id"] == "misc_run_history"


# --- detail: family id resolves to grouped summary -----------------------------------------------------


def test_get_experiment_by_family_id_returns_grouped_summary(client: TestClient) -> None:
    response = client.get("/api/experiments/exp")

    assert response.status_code == 200
    body = response.json()
    assert body["run_count"] == 2
    assert {r["id"] for r in body["runs"]} == {"exp_A_baseline", "exp_B_variant"}
    # This is the summary shape, not the individual-run detail shape.
    assert "artifacts" not in body
    assert "hyperparameters" not in body


# --- detail: individual run id (backward compatibility) -----------------------------------------------------


def test_get_experiment_detail_dqn(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_test_dqn")

    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == "Double DQN"
    assert "CNN" in body["architecture"]
    assert body["hyperparameters"]["lr"] == 1e-4
    assert body["hyperparameters"]["network_size"] == "default"
    assert body["training_configuration"]["episodes"] == 100
    assert body["training_configuration"]["used_checkpoint"] == "best_model.pt"
    assert body["evaluation_metrics"]["win_rate"] == 0.1
    assert body["best_checkpoint"]["episode"] == 50
    # Evaluation/training-config keys must not leak into hyperparameters.
    assert "win_rate" not in body["hyperparameters"]
    assert "episodes" not in body["hyperparameters"]
    # Standalone run, not part of any family.
    assert body["family_id"] is None


def test_get_experiment_detail_ppo(client: TestClient) -> None:
    response = client.get("/api/experiments/ppo_exp_test")

    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"].startswith("PPO")
    assert body["hyperparameters"]["gamma"] == 0.99
    assert body["hyperparameters"]["clip_epsilon"] == 0.2
    assert body["training_configuration"]["reward_mode"] == "shaped"
    assert body["evaluation_metrics"]["avg_reward"] == -3.0
    assert body["best_checkpoint"] is None
    assert body["family_id"] is None


def test_get_experiment_detail_family_member_includes_family_id_and_variant_title(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_A_baseline")

    assert response.status_code == 200
    body = response.json()
    assert body["family_id"] == "exp"
    assert body["title"] == "DQN - Baseline"


def test_get_experiment_detail_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/experiments/does_not_exist")

    assert response.status_code == 404
    assert "does_not_exist" in response.json()["detail"]


def test_get_experiment_detail_malformed_summary_returns_422(client: TestClient) -> None:
    response = client.get("/api/experiments/dqn_history_broken")

    assert response.status_code == 422


def test_get_experiment_detail_missing_results_dir_returns_404(empty_client: TestClient) -> None:
    response = empty_client.get("/api/experiments/anything")

    assert response.status_code == 404


def test_unnamed_artifact_detail_still_fetchable(client: TestClient) -> None:
    response = client.get("/api/experiments/misc_run_history")

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "Unknown"
    assert body["algorithm"] == "Unknown"
    assert body["family_id"] is None


# --- detail: derived metadata (description, artifacts) ------


def test_get_experiment_detail_includes_description(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_test_dqn")
    body = response.json()

    assert body["description"] == "Double DQN trained for 100 episodes on a 5x5 board (5 mines)."


def test_get_experiment_detail_artifacts_with_checkpoints(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_test_dqn")
    artifacts = response.json()["artifacts"]

    assert artifacts["history_json"] is True
    assert artifacts["history_csv"] is False  # the fixture never writes a .csv sibling
    assert artifacts["summary_json"] is True
    assert artifacts["checkpoint_dir"] == "checkpoints_100"
    assert artifacts["best_checkpoint_file"] == "best_model.pt"
    assert artifacts["final_checkpoint_file"] == "final_model.pt"


def test_get_experiment_detail_artifacts_without_checkpoints(client: TestClient) -> None:
    # ppo_exp_test's fixture never creates a checkpoints_* directory.
    response = client.get("/api/experiments/ppo_exp_test")
    artifacts = response.json()["artifacts"]

    assert artifacts["checkpoint_dir"] is None
    assert artifacts["best_checkpoint_file"] is None
    assert artifacts["final_checkpoint_file"] is None


def test_loose_history_file_artifacts_do_not_guess_unrelated_checkpoint_dir_names(client: TestClient) -> None:
    # dqn_evaluate_agents_history has no "checkpoints_1" directory (1 == its
    # row-count-derived episode count) -- must resolve to null, not error.
    response = client.get("/api/experiments/dqn_evaluate_agents_history")
    artifacts = response.json()["artifacts"]

    assert artifacts["checkpoint_dir"] is None


# --- ablation grouping (unaffected by the family-list refactor) -------------------------------------------------------------------


def test_ablation_groups_sibling_experiments(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_A_baseline/ablation")

    assert response.status_code == 200
    body = response.json()
    assert body["group"] == "exp"
    ids = [m["id"] for m in body["members"]]
    assert ids == ["exp_A_baseline", "exp_B_variant"]  # id-sorted, includes the queried experiment itself


def test_ablation_members_include_parsed_variant_and_win_rate(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_A_baseline/ablation")
    members = {m["id"]: m for m in response.json()["members"]}

    assert members["exp_A_baseline"]["variant_label"] == "A"
    assert members["exp_A_baseline"]["variant"] == "baseline"
    assert members["exp_A_baseline"]["win_rate"] == 0.02
    assert members["exp_B_variant"]["variant_label"] == "B"
    assert members["exp_B_variant"]["win_rate"] == 0.05


def test_ablation_is_symmetric_regardless_of_which_member_is_queried(client: TestClient) -> None:
    from_a = client.get("/api/experiments/exp_A_baseline/ablation").json()
    from_b = client.get("/api/experiments/exp_B_variant/ablation").json()

    assert from_a == from_b


def test_ablation_returns_empty_members_for_an_id_that_does_not_match_the_pattern(client: TestClient) -> None:
    # "exp_test_dqn" has no single-uppercase-letter segment, so it doesn't match.
    response = client.get("/api/experiments/exp_test_dqn/ablation")

    assert response.status_code == 200
    assert response.json() == {"group": None, "members": []}


def test_ablation_grouping_is_not_dqn_specific(client: TestClient) -> None:
    # The same generic pattern must also group the PPO fixture family if one
    # exists; here we confirm it does NOT cross-contaminate DQN and PPO groups
    # by checking exp_A_baseline's group never includes a ppo_* id.
    response = client.get("/api/experiments/exp_A_baseline/ablation")
    ids = [m["id"] for m in response.json()["members"]]

    assert all(not i.startswith("ppo_") for i in ids)


def test_ablation_unknown_experiment_returns_404(client: TestClient) -> None:
    response = client.get("/api/experiments/does_not_exist/ablation")

    assert response.status_code == 404


def test_ablation_malformed_experiment_returns_422(client: TestClient) -> None:
    response = client.get("/api/experiments/dqn_history_broken/ablation")

    assert response.status_code == 422
