"""Shared agent-construction logic for replay/race generation scripts.

Both `generate_replays.py` and `generate_race.py` need to build a runnable
agent (plus its action_fn/reasoning_fn/on_episode_start) from a CLI-selected
agent kind, loading a trained checkpoint for DQN/PPO -- this is the one place
that logic lives, so the two scripts can't drift out of sync on how
checkpoints are resolved or how reasoning is wired up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from agents.csp_solver import CSPAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from models.dqn_network import NETWORK_PRESETS as DQN_PRESETS
from models.ppo_network import NETWORK_PRESETS as PPO_PRESETS
from evaluation.replay import ReasoningFn, csp_reasoning, dqn_reasoning, ppo_reasoning

AGENT_DISPLAY_NAMES = {"csp": "CSP", "dqn": "DQN", "ppo": "PPO", "random": "Random"}


def default_checkpoint_path(agent: str, results_dir: Path) -> Path:
    if agent == "dqn":
        return results_dir / "checkpoints_evaluate_agents" / "best_model.pt"
    return results_dir / "checkpoints_ppo_evaluate_agents" / "best_policy.pt"


# `used_checkpoint` values written by dqn_experiment.py / ppo_experiment.py,
# mapped to which of the two saved files that run actually deployed. Kept
# byte-identical in meaning to `apply_reevaluation._DEPLOYED`, which reads the
# same field to decide which checkpoint's score belongs in a summary -- the two
# must agree, or the number published for a run and the weights replayed for it
# describe different models.
_DEPLOYED_KIND = {
    "best_model.pt": "best",
    "best_policy.pt": "best",
    "final_in_memory_weights": "final",
}


def deployed_checkpoint_kind(experiment_dir: Path) -> Optional[str]:
    """"best" or "final" -- which weights this run actually deployed, read from
    its own summary's `used_checkpoint`, or None when there is no summary or it
    records a value this mapping doesn't recognize.

    Every run saves *both* a best_* and a final_* file, so the filename alone
    cannot say which one the run's published win rate describes. 31 of this
    project's 47 runs deployed `final`, and the two differ by more than most of
    the training variables under test (see `reevaluate_checkpoints.py`'s
    docstring on best-checkpoint selection ranking candidates on 50-episode
    noise) -- so picking by filename precedence silently replays weights that
    never produced the number shown next to them.
    """
    summaries = sorted(experiment_dir.glob("*_summary.json"))
    if not summaries:
        return None
    try:
        summary = json.loads(summaries[0].read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return _DEPLOYED_KIND.get(summary.get("used_checkpoint"))


def experiment_checkpoint_path(
    agent: str, experiment_id: str, results_dir: Path, *, prefer: str = "auto"
) -> Path:
    """Path to the checkpoint to load for `experiment_id`.

    `prefer="auto"` (the default) resolves to whichever file the run itself
    recorded deploying, so a replay or a re-scored board result is played by the
    same weights whose win rate the site publishes for that run. Pass "best" or
    "final" to force one explicitly. When provenance can't be determined, falls
    back to the historical best-then-final precedence and says so, rather than
    failing a regeneration that has no summary to consult.
    """
    experiment_dir = results_dir / experiment_id
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"No experiment directory found at {experiment_dir}")

    checkpoint_dirs = sorted(experiment_dir.glob("checkpoints_*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints_* directory found under {experiment_dir}")
    checkpoint_dir = checkpoint_dirs[0]

    best_name = "best_model.pt" if agent == "dqn" else "best_policy.pt"
    final_name = "final_model.pt" if agent == "dqn" else "final_policy.pt"

    kind = prefer if prefer in ("best", "final") else deployed_checkpoint_kind(experiment_dir)
    if kind is None:
        print(
            f"  ! {experiment_id}: no recognized `used_checkpoint` in its summary -- "
            f"falling back to {best_name} then {final_name}. The weights loaded may not be the ones "
            f"whose published win rate this run reports; pass --checkpoint-file to be explicit."
        )
        order = (best_name, final_name)
    else:
        order = (best_name, final_name) if kind == "best" else (final_name, best_name)

    for name in order:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {best_name} or {final_name} found under {checkpoint_dir}")


# Presets whose weights are independent of board size, and so can be loaded at a
# board the checkpoint never trained on. Read from the preset table rather than
# hardcoded, so adding a future conv-head preset does not need a change here.
#
# Every other preset ends in `Flatten -> Linear(conv_channels[-1] * rows * cols,
# ...)`, whose weight matrix has `rows * cols` baked into its shape. Loading one
# at a different board size fails in `load_state_dict` -- correctly, but with a
# raw shape-mismatch traceback that says nothing about *why*. `_board_override`
# raises before that point instead, because "this architecture cannot transfer"
# is the actual answer and is worth stating as one.
def _size_independent(presets: Dict[str, Dict[str, Any]], network_size: str) -> bool:
    return presets.get(network_size, {}).get("head_type") == "conv"


def _board_override(
    raw: Dict[str, Any],
    presets: Dict[str, Dict[str, Any]],
    board_override: Optional[Tuple[int, int]],
    agent: str,
) -> Tuple[int, int]:
    """The (rows, cols) to build the agent at: the checkpoint's own, or an
    explicit override when the architecture actually supports one."""
    trained = (raw["rows"], raw["cols"])
    if board_override is None or board_override == trained:
        return trained

    network_size = raw.get("network_size", "default")
    if not _size_independent(presets, network_size):
        raise ValueError(
            f"Cannot evaluate this {agent.upper()} checkpoint at {board_override[0]}x{board_override[1]}: it was "
            f"trained at {trained[0]}x{trained[1]} on the {network_size!r} preset, whose Linear head is "
            f"board-size-specific. Only presets with a 1x1-convolution head (e.g. 'fully_conv') transfer "
            f"across board sizes."
        )
    return board_override


def _load_dqn_agent(checkpoint_path: Path, seed: int, board_override: Optional[Tuple[int, int]] = None) -> DQNAgent:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network_size = raw.get("network_size", "default")
    rows, cols = _board_override(raw, DQN_PRESETS, board_override, "dqn")
    agent = DQNAgent(rows=rows, cols=cols, network_size=network_size, seed=seed)
    agent.load_checkpoint(checkpoint_path)
    return agent


def _load_ppo_agent(checkpoint_path: Path, seed: int, board_override: Optional[Tuple[int, int]] = None) -> PPOAgent:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network_size = raw.get("network_size", "default")
    rows, cols = _board_override(raw, PPO_PRESETS, board_override, "ppo")
    agent = PPOAgent(rows=rows, cols=cols, network_size=network_size, seed=seed)
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
    checkpoint_file: str = "auto",
    transfer_board: bool = False,
) -> Tuple[Any, Callable[[Any], int], Optional[ReasoningFn], Optional[Callable[[], None]]]:
    """Construct the agent plus its action_fn/reasoning_fn/on_episode_start, mirroring
    evaluate_agents.py's per-agent wiring against the same generic recorder -- see
    evaluation.replay's module docstring. `rows`/`cols`/`num_mines` only matter for CSP
    (DQN/PPO's board size is whatever their loaded checkpoint was trained for), unless
    `transfer_board` is set.

    `checkpoint_file` picks which of a run's two saved checkpoints to load --
    "auto" follows what the run recorded deploying (see
    `experiment_checkpoint_path`), and only applies when `experiment_id` is set.

    `transfer_board` builds a DQN/PPO agent at the caller's `rows`/`cols` instead
    of the checkpoint's own, which is what a zero-shot cross-board-size
    evaluation is. It defaults to False so every existing caller keeps loading a
    checkpoint at exactly the board it was trained on -- the safe behaviour,
    since silently evaluating a 5x5 policy on 9x9 boards would report transfer as
    though it were a matched result. Only `fully_conv`-style conv-head presets
    can honour it; anything else raises rather than failing later on a shape
    mismatch."""
    if agent_kind == "random":
        agent = RandomAgent(seed=seed)
        return agent, agent.select_action, None, None

    if agent_kind == "csp":
        agent = CSPAgent(rows=rows, cols=cols, num_mines=num_mines, seed=seed)
        return agent, agent.choose_action, (lambda board, action: csp_reasoning(agent, board, action)), agent.reset

    checkpoint_path = (
        experiment_checkpoint_path(agent_kind, experiment_id, results_dir, prefer=checkpoint_file)
        if experiment_id is not None
        else default_checkpoint_path(agent_kind, results_dir)
    )
    override = (rows, cols) if transfer_board else None
    if agent_kind == "dqn":
        agent = _load_dqn_agent(checkpoint_path, seed, override)
        return (
            agent,
            lambda obs: agent.select_action(obs, explore=False),
            (lambda board, action: dqn_reasoning(agent, board, action)),
            None,
        )

    agent = _load_ppo_agent(checkpoint_path, seed, override)
    return (
        agent,
        lambda obs: agent.select_action(obs, explore=False),
        (lambda board, action: ppo_reasoning(agent, board, action)),
        None,
    )
