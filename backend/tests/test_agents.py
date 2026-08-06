"""Tests for GET /api/agents."""

from fastapi.testclient import TestClient


def test_list_agents_returns_full_catalog(client: TestClient) -> None:
    response = client.get("/api/agents")

    assert response.status_code == 200
    agents = response.json()
    names = {a["name"] for a in agents}
    assert names == {"Random", "CSP", "Q-Learning", "DQN", "PPO"}


def test_agent_entries_have_required_fields(client: TestClient) -> None:
    response = client.get("/api/agents")

    for agent in response.json():
        assert isinstance(agent["name"], str) and agent["name"]
        assert isinstance(agent["type"], str) and agent["type"]
        assert isinstance(agent["description"], str) and agent["description"]
        assert isinstance(agent["has_experiment_artifacts"], bool)


def test_only_dqn_and_ppo_have_experiment_artifacts(client: TestClient) -> None:
    response = client.get("/api/agents")
    agents = {a["name"]: a for a in response.json()}

    assert agents["DQN"]["has_experiment_artifacts"] is True
    assert agents["PPO"]["has_experiment_artifacts"] is True
    assert agents["Random"]["has_experiment_artifacts"] is False
    assert agents["CSP"]["has_experiment_artifacts"] is False
    assert agents["Q-Learning"]["has_experiment_artifacts"] is False


def test_agents_endpoint_does_not_depend_on_results_directory(empty_client: TestClient) -> None:
    # The catalog is static, so it must not care whether results/ exists.
    response = empty_client.get("/api/agents")

    assert response.status_code == 200
    assert len(response.json()) == 5
