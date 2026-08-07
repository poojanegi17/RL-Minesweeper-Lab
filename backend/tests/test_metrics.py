"""Tests for GET /api/experiments/{id}/metrics and GET /api/leaderboard."""

from fastapi.testclient import TestClient


# --- per-experiment metrics ---------------------------------------------------


def test_get_metrics_dqn_series_and_row_count(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_test_dqn/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"] == "exp_test_dqn"
    assert body["agent"] == "DQN"
    assert body["total_episodes"] == 100
    assert len(body["episodes"]) == 100
    assert set(body["series"]) == {
        "episode", "total_reward", "steps", "won", "epsilon", "lr", "loss",
        "avg_q", "max_q", "grad_norm", "td_error_mean", "td_error_max",
        "reward_rolling_mean", "win_rate_rolling",
    }


def test_get_metrics_ppo_series_differs_from_dqn(client: TestClient) -> None:
    response = client.get("/api/experiments/ppo_exp_test/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "PPO"
    assert "policy_loss" in body["series"]
    assert "value_loss" in body["series"]
    assert "explained_variance" in body["series"]
    assert "epsilon" not in body["series"]  # DQN-only field must not appear for PPO


def test_get_metrics_preserves_null_values_for_pre_training_episodes(client: TestClient) -> None:
    response = client.get("/api/experiments/exp_test_dqn/metrics")

    first_episode = response.json()["episodes"][0]
    assert first_episode["loss"] is None  # matches the fixture: loss is null before episode index 2


def test_get_metrics_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/experiments/does_not_exist/metrics")

    assert response.status_code == 404


def test_get_metrics_malformed_summary_returns_422(client: TestClient) -> None:
    # dqn_history_broken's history JSON is actually valid -- it's the summary
    # that's malformed -- but get_experiment() (called first, to resolve the
    # agent name) fails on that before history is ever read.
    response = client.get("/api/experiments/dqn_history_broken/metrics")

    assert response.status_code == 422


# --- leaderboard ---------------------------------------------------------------


def test_leaderboard_includes_full_catalog(client: TestClient) -> None:
    response = client.get("/api/leaderboard")

    assert response.status_code == 200
    agents = {e["agent"] for e in response.json()}
    assert agents == {"Random", "CSP", "Q-Learning", "DQN", "PPO"}


def test_leaderboard_sorted_by_win_rate_descending(client: TestClient) -> None:
    response = client.get("/api/leaderboard")
    win_rates = [e["win_rate"] for e in response.json()]

    assert win_rates == sorted(win_rates, reverse=True)


def test_leaderboard_tags_source_correctly(client: TestClient) -> None:
    response = client.get("/api/leaderboard")
    by_agent = {e["agent"]: e for e in response.json()}

    assert by_agent["DQN"]["source"] == "experiment_artifact"
    assert by_agent["DQN"]["experiment_id"] == "exp_test_dqn"
    assert by_agent["PPO"]["source"] == "experiment_artifact"
    assert by_agent["CSP"]["source"] == "static_reference"
    assert by_agent["CSP"]["experiment_id"] is None


def test_leaderboard_artifact_entries_use_fixture_win_rates(client: TestClient) -> None:
    response = client.get("/api/leaderboard")
    by_agent = {e["agent"]: e for e in response.json()}

    assert by_agent["DQN"]["win_rate"] == 0.1  # from exp_test_dqn's summary
    assert by_agent["PPO"]["win_rate"] == 0.05  # from ppo_exp_test's summary


def test_leaderboard_works_when_results_dir_missing(empty_client: TestClient) -> None:
    # No DQN/PPO artifacts to read -- only the static reference entries should appear.
    response = empty_client.get("/api/leaderboard")

    assert response.status_code == 200
    agents = {e["agent"] for e in response.json()}
    assert agents == {"Random", "CSP", "Q-Learning"}


# --- level/density scoping -------------------------------------------------------


def test_leaderboard_default_level_density_matches_omitted_params(client: TestClient) -> None:
    # Explicitly passing the default must be byte-for-byte the same response
    # as omitting the params entirely -- the whole point of the alias.
    default_response = client.get("/api/leaderboard")
    explicit_response = client.get("/api/leaderboard?level=beginner&density=standard")

    assert default_response.json() == explicit_response.json()


def test_leaderboard_scoped_to_non_default_level_reads_board_results(client: TestClient) -> None:
    response = client.get("/api/leaderboard?level=intermediate&density=standard")

    assert response.status_code == 200
    by_agent = {e["agent"]: e for e in response.json()}
    assert by_agent["CSP"]["win_rate"] == 0.625
    assert by_agent["CSP"]["source"] == "board_result"


def test_leaderboard_scoped_agent_with_no_data_is_not_trained_not_missing(client: TestClient) -> None:
    response = client.get("/api/leaderboard?level=intermediate&density=standard")
    by_agent = {e["agent"]: e for e in response.json()}

    # Every cataloged agent still gets a row -- DQN just has no board result
    # at this level yet (it hasn't been trained there).
    assert set(by_agent.keys()) == {"Random", "CSP", "Q-Learning", "DQN", "PPO"}
    assert by_agent["DQN"]["source"] == "not_trained"
    assert by_agent["DQN"]["win_rate"] is None


def test_leaderboard_unknown_level_returns_400(client: TestClient) -> None:
    response = client.get("/api/leaderboard?level=nightmare&density=standard")

    assert response.status_code == 400


def test_leaderboard_unknown_density_returns_400(client: TestClient) -> None:
    response = client.get("/api/leaderboard?level=beginner&density=nightmare")

    assert response.status_code == 400
