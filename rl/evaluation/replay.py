"""Records a single episode's full timeline for interactive playback.

`ReplayRecorder` is agent-agnostic, the same way `evaluation.metrics.run_episode`
is: it takes an `action_fn(observation) -> action` callable, not an agent
instance, so it works for any agent in this project without needing a shared
base class. Per-agent wiring (which `action_fn`, and which optional
`reasoning_fn` to attach) lives in `evaluation.generate_replays`, mirroring
exactly how `evaluation.evaluate_agents` wires four different agent types
against the same generic `evaluate_agent()`.

Reasoning extraction (`csp_reasoning`, `dqn_reasoning`, `ppo_reasoning`) reads
each agent's already-public attributes (`CSPAgent.known_safe`/`known_mines`/
`constraints`, `DQNAgent.online_network`, `PPOAgent.network`) from the
*outside* -- nothing in `agents/*.py` is modified to support this. Q-values
and action probabilities are recomputed by replaying the exact same forward
pass `select_action` already performs internally (same network, same
encoding, same masking); since evaluation is deterministic (`explore=False`),
recomputing from the same observation reproduces exactly what the agent used
to decide, not an approximation.

Security note (see also the module-level docstring in `generate_replays.py`
and the README's Replay Visualization section): a recorded replay **never**
contains mine positions, anywhere, including in metadata. This is a
deliberate, maximal choice -- rather than storing hidden state in an
optional "debug" section and trusting every downstream consumer (this
module, the API, the frontend) to never surface it, the data simply isn't
written down in the first place. The one thing that might look like a leak
-- highlighting a hit mine -- is derived safely at render time from
information the episode already legitimately reported: `done and not won`
on the step whose `action` was the fatal click. That's the outcome the
agent's own move caused, not a preview of anything else on the board.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from environment.utils import action_to_coords

Board = Sequence[Sequence[int]]
ActionFn = Callable[[Any], int]
ReasoningFn = Callable[[Board, int], Optional[Dict[str, Any]]]


def _to_plain_board(observation: Any) -> List[List[int]]:
    """Convert a numpy observation (or already-plain board) into a JSON-serializable nested list of ints."""
    array = np.asarray(observation)
    return [[int(v) for v in row] for row in array]


class ReplayRecorder:
    """Plays one episode and records its full timeline for later playback."""

    def record_episode(
        self,
        env: Any,
        action_fn: ActionFn,
        reasoning_fn: Optional[ReasoningFn] = None,
        on_episode_start: Optional[Callable[[], None]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Play one episode with `action_fn`, recording every step's board/action/reward/reasoning.

        Args:
            env: A MinesweeperEnv (or compatible) instance.
            action_fn: Maps the current observation to a discrete action --
                the same contract `evaluation.metrics.run_episode` uses.
            reasoning_fn: Optional. Given the observation `action_fn` just
                acted on and the action it chose, returns agent-specific
                explanation metadata (or None if nothing is available for
                this agent type). Never fabricated -- see module docstring.
            on_episode_start: Optional hook called after `env.reset()`, e.g.
                to clear a CSPAgent's deduced knowledge from a previous episode.
            seed: Optional per-episode seed, forwarded to `env.reset(seed=...)`
                -- lets `generate_replays.py` produce a distinct board per
                replay rather than replaying the same layout repeatedly.

        Returns:
            A dict with `initial_board`, `steps` (the timeline), `won`,
            `total_reward`, and `steps_taken`.
        """
        observation, info = env.reset(seed=seed) if seed is not None else env.reset()
        if on_episode_start is not None:
            on_episode_start()

        initial_board = _to_plain_board(observation)
        steps: List[Dict[str, Any]] = []
        total_reward = 0.0
        terminated = truncated = False
        step_number = 0

        while not (terminated or truncated):
            step_number += 1
            action = action_fn(observation)
            row, col = action_to_coords(action, env.cols)
            reasoning = reasoning_fn(observation, action) if reasoning_fn is not None else None

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            steps.append(
                {
                    "step": step_number,
                    "board_state": _to_plain_board(observation),
                    "action": {"row": row, "col": col},
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                    "reasoning": reasoning,
                }
            )

        return {
            "initial_board": initial_board,
            "steps": steps,
            "won": bool(info.get("won", False)),
            "total_reward": total_reward,
            "steps_taken": step_number,
        }


# --- Reasoning extractors ---------------------------------------------------
#
# Each function is given the observation the agent acted on and the action it
# chose, and returns a small dict of real, derived facts -- or None if no
# explanation is available for that agent (e.g. RandomAgent). Never guesses
# or invents a plausible-sounding reason; when the actual deduction can't be
# cleanly attributed (e.g. CSP's subset rule spans two constraints, not one),
# the returned fields are left `None` rather than approximated.


def csp_reasoning(agent: Any, board: Board, action: int) -> Dict[str, Any]:
    """Explain a CSPAgent's choice using its own already-computed deduction state.

    `agent.known_safe`/`known_mines`/`constraints` were already updated by
    `agent.choose_action()` (which calls `update_constraints` first) before
    this is called with the same `board` -- so this reads state the agent
    itself just derived, not a fresh recomputation.
    """
    row, col = action_to_coords(action, agent.cols)
    cell = (row, col)

    if cell in agent.known_mines:
        # CSP never intentionally clicks a known mine; guard defensively anyway.
        return {
            "deduction_type": "mine",
            "constraint_cells": None,
            "remaining_mines": None,
            "mine_probability": None,
            "inference": f"Cell ({row}, {col}) was deduced to be a mine.",
        }

    if cell in agent.known_safe:
        # The single-constraint rule: a constraint with count<=0 means every
        # cell in it is safe. Find one that contains the chosen cell, if any
        # -- the deduction may instead have come from the subset rule (two
        # constraints compared against each other), in which case no single
        # constraint attributes it and constraint_cells stays None.
        match = next(
            ((sorted(cells), count) for cells, count in agent.constraints if cell in cells and count <= 0),
            None,
        )
        return {
            "deduction_type": "safe",
            "constraint_cells": [[r, c] for r, c in match[0]] if match else None,
            "remaining_mines": match[1] if match else 0,
            "mine_probability": None,
            "inference": f"Cell ({row}, {col}) is safe.",
        }

    probabilities = agent.calculate_probabilities(board)
    return {
        "deduction_type": "probability_guess",
        "constraint_cells": None,
        "remaining_mines": None,
        "mine_probability": probabilities.get(cell),
        "inference": f"No safe cell deduced; guessing the lowest mine-probability cell ({row}, {col}).",
    }


def dqn_reasoning(agent: Any, board: Board, action: int) -> Dict[str, Any]:
    """The chosen action's Q-value, recomputed via the same forward pass `select_action` used."""
    from models.dqn_network import encode_observation

    encoded = encode_observation(np.asarray(board))
    tensor = torch.from_numpy(encoded[None, ...]).to(agent.device)
    with torch.no_grad():
        q_values = agent.online_network(tensor).squeeze(0).cpu().numpy()
    return {"q_value": float(q_values[action])}


def ppo_reasoning(agent: Any, board: Board, action: int) -> Dict[str, Any]:
    """The chosen action's probability under the policy, recomputed the same way `select_action` did.

    Reuses `PPOAgent._apply_action_mask` (a stateless `@staticmethod`) rather
    than re-deriving the masking logic here, so this can't silently drift
    out of sync with the agent's own masking if that ever changes.
    """
    from models.dqn_network import encode_observation
    from torch.distributions import Categorical

    encoded = encode_observation(np.asarray(board))
    tensor = torch.from_numpy(encoded[None, ...]).to(agent.device)
    with torch.no_grad():
        logits, _ = agent.network(tensor)
        masked_logits = type(agent)._apply_action_mask(logits, tensor)
        probabilities = Categorical(logits=masked_logits).probs.squeeze(0).cpu().numpy()
    return {"action_probability": float(probabilities[action])}


def build_replay(
    *,
    agent_name: str,
    experiment_id: Optional[str],
    board_size: str,
    mines: int,
    seed: Optional[int],
    episode_number: int,
    episode: Dict[str, Any],
    generated_at: str,
) -> Dict[str, Any]:
    """Assemble episode metadata + a recorded episode dict into one replay artifact."""
    return {
        "id": f"{agent_name.lower()}_episode_{episode_number}",
        "agent": agent_name,
        "experiment_id": experiment_id,
        "board_size": board_size,
        "mines": mines,
        "seed": seed,
        "episode_number": episode_number,
        "generated_at": generated_at,
        "initial_board": episode["initial_board"],
        "steps": episode["steps"],
        "won": episode["won"],
        "total_reward": episode["total_reward"],
        "steps_taken": episode["steps_taken"],
    }


def build_race(
    *,
    seed: int,
    board_size: str,
    mines: int,
    generated_at: str,
    agent_episodes: Dict[str, Tuple[Optional[str], Dict[str, Any]]],
) -> Dict[str, Any]:
    """Bundle several agents' episodes -- recorded on the *same* seed, so they share an
    identical mine layout (see `environment.minesweeper.Minesweeper`: placement depends
    only on the seed) -- into one race artifact.

    Args:
        seed: The shared seed every episode in `agent_episodes` was recorded with.
        agent_episodes: Maps display agent name (e.g. "DQN") to
            `(experiment_id, episode)`, where `episode` is a `ReplayRecorder.record_episode`
            result. `experiment_id` is `None` for agents with no checkpoint (Random, CSP).

    `initial_board` is stored once at the top level (every agent's episode has the
    identical board by construction) rather than duplicated per agent.
    """
    first_episode = next(iter(agent_episodes.values()))[1]
    return {
        "id": f"race_{seed}",
        "seed": seed,
        "board_size": board_size,
        "mines": mines,
        "generated_at": generated_at,
        "initial_board": first_episode["initial_board"],
        "agents": {
            agent_name: {
                "experiment_id": experiment_id,
                "steps": episode["steps"],
                "won": episode["won"],
                "total_reward": episode["total_reward"],
                "steps_taken": episode["steps_taken"],
            }
            for agent_name, (experiment_id, episode) in agent_episodes.items()
        },
    }
