"""Tests for GET /api/board-configs."""

from fastapi.testclient import TestClient


def test_list_board_configs_returns_all_three_levels(client: TestClient) -> None:
    response = client.get("/api/board-configs")

    assert response.status_code == 200
    levels = [c["level"] for c in response.json()]
    assert levels == ["beginner", "intermediate", "expert"]


def test_board_config_shape(client: TestClient) -> None:
    response = client.get("/api/board-configs")
    by_level = {c["level"]: c for c in response.json()}

    beginner = by_level["beginner"]
    assert beginner["rows"] == 5
    assert beginner["cols"] == 5
    assert beginner["densities"] == {"sparse": 3, "standard": 5, "dense": 8}


def test_board_configs_available_even_with_no_results_dir(empty_client: TestClient) -> None:
    # The catalog is static config, not filesystem-derived -- it must be
    # available even before any level/density data has been generated.
    response = empty_client.get("/api/board-configs")

    assert response.status_code == 200
    assert len(response.json()) == 3
