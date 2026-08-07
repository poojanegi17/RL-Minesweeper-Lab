"""Tests for GET /api/races and GET /api/races/{id}."""

from fastapi.testclient import TestClient


# --- list ----------------------------------------------------------------------


def test_list_races_discovers_well_formed_races(client: TestClient) -> None:
    response = client.get("/api/races")

    assert response.status_code == 200
    ids = {r["id"] for r in response.json()}
    assert "race_1" in ids
    # Malformed files must be silently skipped from the list, not crash it.
    assert "incomplete_race" not in ids
    assert "broken_race" not in ids


def test_list_races_summary_fields(client: TestClient) -> None:
    response = client.get("/api/races")
    by_id = {r["id"]: r for r in response.json()}

    race = by_id["race_1"]
    assert race["seed"] == 1
    assert race["board_size"] == "3x3"
    assert race["mines"] == 1
    assert set(race["agents"]) == {"Random", "CSP", "DQN"}


def test_list_races_summary_never_includes_timelines(client: TestClient) -> None:
    # The list view is deliberately lightweight -- per-agent timelines and
    # the initial board belong to the detail endpoint only.
    response = client.get("/api/races")
    for race in response.json():
        assert "initial_board" not in race
        for agent_name in race["agents"]:
            assert isinstance(agent_name, str)  # agents is List[str] here, not the detail's Dict[str, RaceAgentResult]


def test_list_races_empty_when_races_dir_missing(empty_client: TestClient) -> None:
    response = empty_client.get("/api/races")

    assert response.status_code == 200
    assert response.json() == []


# --- detail ----------------------------------------------------------------------


def test_get_race_detail_shape(client: TestClient) -> None:
    response = client.get("/api/races/race_1")

    assert response.status_code == 200
    body = response.json()
    assert body["seed"] == 1
    assert body["initial_board"] == [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    assert set(body["agents"].keys()) == {"Random", "CSP", "DQN"}


def test_get_race_detail_per_agent_results(client: TestClient) -> None:
    response = client.get("/api/races/race_1")
    agents = response.json()["agents"]

    assert agents["CSP"]["won"] is True
    assert agents["Random"]["won"] is False
    assert agents["DQN"]["experiment_id"] == "exp_test_dqn"
    assert agents["CSP"]["experiment_id"] is None


def test_get_race_detail_per_agent_reasoning_shape(client: TestClient) -> None:
    response = client.get("/api/races/race_1")
    agents = response.json()["agents"]

    assert agents["CSP"]["steps"][0]["reasoning"]["deduction_type"] == "safe"
    assert agents["DQN"]["steps"][0]["reasoning"] == {"q_value": 0.5}
    assert agents["Random"]["steps"][0]["reasoning"] is None


def test_get_race_detail_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/races/does_not_exist")

    assert response.status_code == 404
    assert "does_not_exist" in response.json()["detail"]


def test_get_race_detail_malformed_json_returns_422(client: TestClient) -> None:
    response = client.get("/api/races/broken_race")

    assert response.status_code == 422


def test_get_race_detail_missing_required_fields_returns_422(client: TestClient) -> None:
    response = client.get("/api/races/incomplete_race")

    assert response.status_code == 422


def test_get_race_detail_missing_results_dir_returns_404(empty_client: TestClient) -> None:
    response = empty_client.get("/api/races/anything")

    assert response.status_code == 404
