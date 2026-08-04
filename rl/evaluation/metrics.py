"""Episode-running and metric-aggregation helpers for agent evaluation.

Works with any agent that exposes a callable `action_fn(observation) -> action`,
so it applies equally to `RandomAgent.select_action` and `CSPAgent.choose_action`
without either agent needing a shared base class.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

ActionFn = Callable[[np.ndarray], int]


def run_episode(
    env,
    action_fn: ActionFn,
    on_episode_start: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Play one full episode with `action_fn` until termination.

    Args:
        env: A MinesweeperEnv (or compatible Gymnasium environment) instance.
        action_fn: Maps the current observation to a discrete action.
        on_episode_start: Optional hook called after `env.reset()`, e.g. to
            clear a CSPAgent's deduced knowledge from the previous episode.

    Returns:
        A dict with `total_reward`, `steps`, and `won`.
    """
    observation, info = env.reset()
    if on_episode_start is not None:
        on_episode_start()

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = action_fn(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    return {
        "total_reward": total_reward,
        "steps": steps,
        "won": info.get("won", False),
    }


def evaluate_agent(
    env,
    action_fn: ActionFn,
    num_episodes: int = 100,
    on_episode_start: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Run `num_episodes` episodes and aggregate win rate / length / failures.

    Returns:
        A dict with `games_played`, `wins`, `failures`, `win_rate`, and
        `avg_episode_length`.
    """
    wins = 0
    failures = 0
    total_steps = 0

    for _ in range(num_episodes):
        result = run_episode(env, action_fn, on_episode_start)
        total_steps += result["steps"]
        if result["won"]:
            wins += 1
        else:
            failures += 1

    return {
        "games_played": num_episodes,
        "wins": wins,
        "failures": failures,
        "win_rate": wins / num_episodes,
        "avg_episode_length": total_steps / num_episodes,
    }
