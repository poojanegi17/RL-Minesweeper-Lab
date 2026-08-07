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
    assert race["turn_order"] == ["Random", "CSP", "DQN"]
    assert race["won"] is False
    assert race["total_turns"] == 3


def test_list_races_summary_never_includes_turns_or_board(client: TestClient) -> None:
    # The list view is deliberately lightweight -- the turn-by-turn timeline
    # and the initial board belong to the detail endpoint only.
    response = client.get("/api/races")
    for race in response.json():
        assert "turns" not in race
        assert "initial_board" not in race


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
    assert len(body["turns"]) == 3
    assert body["surviving_agents"] == ["CSP", "DQN"]
    assert body["eliminated_agents"] == {"Random": 1}


def test_get_race_detail_turn_shape_and_order(client: TestClient) -> None:
    response = client.get("/api/races/race_1")
    turns = response.json()["turns"]

    assert [t["agent"] for t in turns] == ["Random", "CSP", "DQN"]
    assert turns[0]["eliminated"] is True
    assert turns[1]["eliminated"] is False
    assert turns[0]["action"] == {"row": 0, "col": 0}


def test_get_race_detail_eliminated_agents_fatal_cell_never_shown_revealed(client: TestClient) -> None:
    response = client.get("/api/races/race_1")
    body = response.json()

    # Random was eliminated at (0, 0), turn 1 -- confirm that cell stays
    # hidden (-1) in every subsequent turn's board_state, per the fixture.
    for turn in body["turns"]:
        assert turn["board_state"][0][0] == -1


def test_get_race_detail_reasoning_shape(client: TestClient) -> None:
    response = client.get("/api/races/race_1")
    turns = response.json()["turns"]

    assert turns[1]["reasoning"]["deduction_type"] == "probability_guess"
    assert turns[2]["reasoning"] == {"q_value": 0.5}
    assert turns[0]["reasoning"] is None


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
