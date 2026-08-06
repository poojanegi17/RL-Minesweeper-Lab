"""Direct unit tests for the ResultsLoader service (below the API layer)."""

from pathlib import Path

import pytest

from app.services.results_loader import (
    ExperimentNotFoundError,
    MalformedArtifactError,
    ResultsLoader,
    algorithm_info,
    build_artifact_manifest,
    derive_description,
    derive_techniques,
    derive_title,
    parse_ablation_id,
    split_summary_fields,
)


def test_list_experiments_on_missing_directory_returns_empty_list(tmp_path: Path) -> None:
    loader = ResultsLoader(tmp_path / "does_not_exist")

    assert loader.list_experiments() == []


def test_list_experiments_on_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    loader = ResultsLoader(tmp_path)

    assert loader.list_experiments() == []


def test_get_experiment_on_missing_directory_raises_not_found(tmp_path: Path) -> None:
    loader = ResultsLoader(tmp_path / "does_not_exist")

    with pytest.raises(ExperimentNotFoundError):
        loader.get_experiment("anything")


def test_list_experiments_skips_malformed_summary_without_raising(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)

    records = loader.list_experiments()  # must not raise despite dqn_history_broken's bad summary

    ids = {r.id for r in records}
    assert "dqn_history_broken" not in ids
    assert "exp_test_dqn" in ids


def test_get_experiment_on_malformed_summary_raises(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)

    with pytest.raises(MalformedArtifactError):
        loader.get_experiment("dqn_history_broken")


def test_get_history_returns_full_episode_list(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)

    history = loader.get_history("exp_test_dqn")

    assert len(history) == 100
    assert history[0]["episode"] == 1


def test_get_history_on_unknown_id_raises_not_found(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)

    with pytest.raises(ExperimentNotFoundError):
        loader.get_history("nope")


def test_split_summary_fields_buckets_correctly() -> None:
    summary = {
        "episodes": 100,
        "checkpoint_every": 50,
        "win_rate": 0.1,
        "avg_reward": -2.0,
        "lr": 1e-4,
        "gamma": 0.99,
        "best_checkpoint_metadata": {"episode": 50},
    }

    result = split_summary_fields(summary)

    assert result["training_configuration"] == {"episodes": 100, "checkpoint_every": 50}
    assert result["evaluation_metrics"] == {"win_rate": 0.1, "avg_reward": -2.0}
    assert result["hyperparameters"] == {"lr": 1e-4, "gamma": 0.99}
    assert result["best_checkpoint"] == {"episode": 50}


def test_split_summary_fields_handles_missing_best_checkpoint() -> None:
    result = split_summary_fields({"episodes": 5})

    assert result["best_checkpoint"] is None


def test_algorithm_info_known_and_unknown_agents() -> None:
    assert algorithm_info("DQN")["algorithm"] == "Double DQN"
    assert algorithm_info("PPO")["algorithm"].startswith("PPO")
    assert algorithm_info("Nonsense")["algorithm"] == "Unknown"


# --- derived metadata --------------------------------------------------------------------


def test_derive_title_is_mechanical_not_authored() -> None:
    assert derive_title("DQN", 25000) == "DQN - 25,000 episodes"
    assert derive_title("PPO", 100) == "PPO - 100 episodes"


def test_derive_description_is_mechanical() -> None:
    text = derive_description("Double DQN", 25000, "5x5", 5)
    assert text == "Double DQN trained for 25,000 episodes on a 5x5 board (5 mines)."


def test_derive_techniques_only_flags_recorded_config() -> None:
    minimal_summary = {"used_checkpoint": "final_in_memory_weights"}
    assert derive_techniques("DQN", minimal_summary) == ["Double DQN"]

    full_summary = {
        "lr_schedule": [[0, 1e-4]],
        "network_size": "small",
        "used_checkpoint": "best_model.pt",
    }
    techniques = derive_techniques("DQN", full_summary)
    assert set(techniques) == {"Double DQN", "LR decay", "Reduced network capacity", "Best-checkpoint deployment"}


def test_derive_techniques_unknown_agent_returns_empty_list() -> None:
    assert derive_techniques("Random", {"anything": True}) == []


def test_parse_ablation_id_matches_expected_patterns() -> None:
    assert parse_ablation_id("exp_A_baseline") == {"group": "exp", "variant_label": "A", "variant": "baseline"}
    assert parse_ablation_id("ppo_exp_C_shaped") == {"group": "ppo_exp", "variant_label": "C", "variant": "shaped"}


def test_parse_ablation_id_returns_none_for_non_matching_ids() -> None:
    assert parse_ablation_id("dqn_history_25000") is None
    assert parse_ablation_id("dqn_evaluate_agents_history") is None


def test_list_ablation_group_returns_sorted_siblings(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)

    members = loader.list_ablation_group("exp")

    assert [m.id for m in members] == ["exp_A_baseline", "exp_B_variant"]


def test_list_ablation_group_empty_for_unknown_group(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)

    assert loader.list_ablation_group("no_such_group") == []


def test_build_artifact_manifest_detects_checkpoint_files(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)
    record = loader.get_experiment("exp_test_dqn")

    manifest = build_artifact_manifest(record)

    assert manifest["checkpoint_dir"] == "checkpoints_100"
    assert manifest["best_checkpoint_file"] == "best_model.pt"
    assert manifest["final_checkpoint_file"] == "final_model.pt"
    assert manifest["history_json"] is True
    assert manifest["summary_json"] is True


def test_build_artifact_manifest_none_when_no_checkpoint_dir(results_dir: Path) -> None:
    loader = ResultsLoader(results_dir)
    record = loader.get_experiment("ppo_exp_test")

    manifest = build_artifact_manifest(record)

    assert manifest["checkpoint_dir"] is None
    assert manifest["best_checkpoint_file"] is None
    assert manifest["final_checkpoint_file"] is None
