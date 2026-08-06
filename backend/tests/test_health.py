"""Tests for the health endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_does_not_depend_on_results_directory(empty_client: TestClient) -> None:
    # Health must stay up even when the results directory is entirely missing.
    response = empty_client.get("/health")

    assert response.status_code == 200
