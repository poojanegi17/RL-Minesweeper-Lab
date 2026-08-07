"""Shared agent-construction logic for replay/race generation scripts.

Both `generate_replays.py` and `generate_race.py` need to build a runnable
agent (plus its action_fn/reasoning_fn/on_episode_start) from a CLI-selected
agent kind, loading a trained checkpoint for DQN/PPO -- this is the one place
that logic lives, so the two scripts can't drift out of sync on how
checkpoints are resolved or how reasoning is wired up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch

from agents.csp_solver import CSPAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from evaluation.replay import ReasoningFn, csp_reasoning, dqn_reasoning, ppo_reasoning

AGENT_DISPLAY_NAMES = {"csp": "CSP", "dqn": "DQN", "ppo": "PPO", "random": "Random"}


def default_checkpoint_path(agent: str, results_dir: Path) -> Path:
    if agent == "dqn":
        return results_dir / "checkpoints_evaluate_agents" / "best_model.pt"
    return results_dir / "checkpoints_ppo_evaluate_agents" / "best_policy.pt"


def experiment_checkpoint_path(agent: str, experiment_id: str, results_dir: Path) -> Path:
    experiment_dir = results_dir / experiment_id
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"No experiment directory found at {experiment_dir}")

    checkpoint_dirs = sorted(experiment_dir.glob("checkpoints_*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints_* directory found under {experiment_dir}")
    checkpoint_dir = checkpoint_dirs[0]

    best_name = "best_model.pt" if agent == "dqn" else "best_policy.pt"
    final_name = "final_model.pt" if agent == "dqn" else "final_policy.pt"
    for name in (best_name, final_name):
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {best_name} or {final_name} found under {checkpoint_dir}")


def _load_dqn_agent(checkpoint_path: Path, seed: int) -> DQNAgent:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = DQNAgent(rows=raw["rows"], cols=raw["cols"], network_size=raw.get("network_size", "default"), seed=seed)
    agent.load_checkpoint(checkpoint_path)
    return agent


def _load_ppo_agent(checkpoint_path: Path, seed: int) -> PPOAgent:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = PPOAgent(rows=raw["rows"], cols=raw["cols"], seed=seed)
    agent.load_checkpoint(checkpoint_path)
    return agent


def build_agent(
    agent_kind: str,
    seed: int,
    experiment_id: Optional[str],
    results_dir: Path,
    *,
    rows: int,
    cols: int,
    num_mines: int,
) -> Tuple[Any, Callable[[Any], int], Optional[ReasoningFn], Optional[Callable[[], None]]]:
    """Construct the agent plus its action_fn/reasoning_fn/on_episode_start, mirroring
    evaluate_agents.py's per-agent wiring against the same generic recorder -- see
    evaluation.replay's module docstring. `rows`/`cols`/`num_mines` only matter for CSP
    (DQN/PPO's board size is whatever their loaded checkpoint was trained for)."""
    if agent_kind == "random":
        agent = RandomAgent(seed=seed)
        return agent, agent.select_action, None, None

    if agent_kind == "csp":
        agent = CSPAgent(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
        return agent, agent.choose_action, (lambda board, action: csp_reasoning(agent, board, action)), agent.reset

    checkpoint_path = (
        experiment_checkpoint_path(agent_kind, experiment_id, results_dir)
        if experiment_id is not None
        else default_checkpoint_path(agent_kind, results_dir)
    )
    if agent_kind == "dqn":
        agent = _load_dqn_agent(checkpoint_path, seed)
        return (
            agent,
            lambda obs: agent.select_action(obs, explore=False),
            (lambda board, action: dqn_reasoning(agent, board, action)),
            None,
        )

    agent = _load_ppo_agent(checkpoint_path, seed)
    return (
        agent,
        lambda obs: agent.select_action(obs, explore=False),
        (lambda board, action: ppo_reasoning(agent, board, action)),
        None,
    )
