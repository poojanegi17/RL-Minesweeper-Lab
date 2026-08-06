"""Tests for GET /api/replays and GET /api/replays/{id}."""

from fastapi.testclient import TestClient


# --- list ----------------------------------------------------------------------


def test_list_replays_discovers_well_formed_replays(client: TestClient) -> None:
    response = client.get("/api/replays")

    assert response.status_code == 200
    ids = {r["id"] for r in response.json()}
    assert "dqn_episode_1" in ids
    assert "csp_episode_1" in ids
    assert "random_episode_1" in ids
    # Malformed files must be silently skipped from the list, not crash it.
    assert "broken_episode" not in ids
    assert "incomplete_episode" not in ids


def test_list_replays_summary_fields(client: TestClient) -> None:
    response = client.get("/api/replays")
    by_id = {r["id"]: r for r in response.json()}

    dqn = by_id["dqn_episode_1"]
    assert dqn["agent"] == "DQN"
    assert dqn["experiment_id"] is None
    assert dqn["won"] is False
    assert dqn["steps"] == 2

    csp = by_id["csp_episode_1"]
    assert csp["won"] is True


def test_list_replays_summary_never_includes_board_or_reasoning(client: TestClient) -> None:
    # The list view is deliberately lightweight -- board states and
    # reasoning belong to the detail endpoint only.
    response = client.get("/api/replays")
    for replay in response.json():
        assert "timeline" not in replay
        assert "initial_board" not in replay


def test_list_replays_empty_when_replays_dir_missing(empty_client: TestClient) -> None:
    response = empty_client.get("/api/replays")

    assert response.status_code == 200
    assert response.json() == []


# --- detail ----------------------------------------------------------------------


def test_get_replay_detail_dqn(client: TestClient) -> None:
    response = client.get("/api/replays/dqn_episode_1")

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "DQN"
    assert body["board_size"] == "3x3"
    assert body["mines"] == 1
    assert body["won"] is False
    assert body["steps"] == 2
    assert len(body["timeline"]) == 2
    assert body["timeline"][0]["reasoning"] == {"q_value": 0.82}


def test_get_replay_detail_csp_reasoning_shape(client: TestClient) -> None:
    response = client.get("/api/replays/csp_episode_1")
    body = response.json()

    reasoning = body["timeline"][0]["reasoning"]
    assert reasoning["deduction_type"] == "safe"
    assert reasoning["constraint_cells"] == [[0, 0]]
    assert reasoning["remaining_mines"] == 0


def test_get_replay_detail_random_has_no_reasoning(client: TestClient) -> None:
    response = client.get("/api/replays/random_episode_1")
    body = response.json()

    assert all(step["reasoning"] is None for step in body["timeline"])


def test_get_replay_detail_initial_board_is_all_hidden(client: TestClient) -> None:
    response = client.get("/api/replays/dqn_episode_1")
    body = response.json()

    assert body["initial_board"] == [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]


def test_get_replay_detail_step_action_shape(client: TestClient) -> None:
    response = client.get("/api/replays/dqn_episode_1")
    body = response.json()

    first_action = body["timeline"][0]["action"]
    assert set(first_action.keys()) == {"row", "col"}


def test_get_replay_detail_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/replays/does_not_exist")

    assert response.status_code == 404
    assert "does_not_exist" in response.json()["detail"]


def test_get_replay_detail_malformed_json_returns_422(client: TestClient) -> None:
    response = client.get("/api/replays/broken_episode")

    assert response.status_code == 422


def test_get_replay_detail_missing_required_fields_returns_422(client: TestClient) -> None:
    response = client.get("/api/replays/incomplete_episode")

    assert response.status_code == 422


def test_get_replay_detail_missing_results_dir_returns_404(empty_client: TestClient) -> None:
    response = empty_client.get("/api/replays/anything")

    assert response.status_code == 404


# --- security: no hidden-state leakage -------------------------------------------


def test_replay_response_never_contains_undeclared_fields(client: TestClient) -> None:
    # Even if a replay file somehow had an extra top-level key (e.g. a bug or
    # a hand-edited file), FastAPI's response_model only serializes declared
    # schema fields -- this pins that behavior down explicitly.
    response = client.get("/api/replays/dqn_episode_1")
    body = response.json()

    expected_keys = {
        "id", "agent", "experiment_id", "board_size", "mines", "seed",
        "episode_number", "generated_at", "initial_board", "timeline",
        "won", "total_reward", "steps",
    }
    assert set(body.keys()) == expected_keys


def test_replay_step_response_never_contains_undeclared_fields(client: TestClient) -> None:
    response = client.get("/api/replays/dqn_episode_1")
    body = response.json()

    for step in body["timeline"]:
        assert set(step.keys()) == {"step", "board_state", "action", "reward", "done", "reasoning"}
