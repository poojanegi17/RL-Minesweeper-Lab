"""Deep Q-Network agent for Minesweeper.

Replaces `QLearningAgent`'s dictionary lookup with a small CNN
(`models.dqn_network.DQNNetwork`) that maps a board pattern to Q-values for
every cell. Because the network generalizes across similar-looking board
patterns instead of memorizing exact ones, it can in principle learn useful
values for states it has never visited, which is precisely the property
tabular Q-learning was found to lack on boards larger than a small grid.

Three standard DQN stabilization tricks are used:
    - Experience replay (`training.replay_buffer.ReplayBuffer`): transitions
      are stored and sampled in random minibatches, breaking the strong
      correlation between consecutive steps of the same episode.
    - A separate target network: bootstrapped targets are computed from a
      periodically-synced copy of the online network rather than the network
      being updated, which keeps the regression target from shifting under
      every single gradient step.
    - Huber loss + gradient clipping: an early version of this agent used
      plain MSE loss, which diverged (loss grew from hundreds into the
      millions within a few thousand episodes) because a handful of noisy,
      large TD-errors early in training produced squared gradients large
      enough to push Q-values further from convergence rather than towards
      it. Huber loss (`smooth_l1_loss`) grows linearly instead of
      quadratically for large errors, and clipping the gradient norm caps
      the damage any single batch can do -- together they keep training
      stable.

Switching the input from one normalized scalar channel to the 11-channel
one-hot encoding (see `models.dqn_network.encode_observation`) let the
network fit the data more aggressively, which made the above divergence
reappear even with Huber loss and clipping -- a sharper input signal means
sharper (and more compounding) errors through the max-based bootstrap. A
lower learning rate (1e-4 instead of 5e-4) and less frequent target-network
syncs (every 25 episodes instead of 10, so the bootstrap target moves more
slowly) were enough to keep loss bounded across a full 6000-episode run
instead of growing without bound.

Double DQN: the bootstrap target now uses the online network to pick the
best next action (`argmax_a' Q_online(s', a')`) but the target network to
evaluate it (`Q_target(s', that_action)`), instead of one network's `max`
doing both jobs (see `_double_dqn_targets`). Plain DQN's `max` operator picks
out whichever action currently looks best *according to the same network
being evaluated*, which is disproportionately likely to be one whose value is
overestimated -- and that bias compounds every time it's bootstrapped from.
Decoupling "which action" from "how good is it" across two networks that
rarely agree on which action is overestimated removes most of that bias.

Production-quality training additions (this module's current milestone):
    - Best-checkpoint selection (`save_checkpoint`/`load_checkpoint`, and
      `train(..., checkpoint_dir=...)`): even with Double DQN, a 50,000-episode
      run showed loss recovering to single digits and then periodically
      spiking into the thousands past ~episode 10,000, and the *final*
      snapshot of weights landed during one of those unstable stretches --
      scoring 0.0% despite in-training windows as high as 2.8%. Periodically
      evaluating the greedy policy during training and keeping only the
      best-scoring snapshot (`best_model.pt`, alongside an unconditional
      `final_model.pt`) means a bad patch late in training can no longer
      erase a good policy found earlier.
    - Extended per-episode metrics (avg/max Q-value, gradient norm, TD-error
      mean/max, rolling reward and win rate; see `train`): turns "loss
      spiked" into a diagnosable signal instead of an unexplained number.
    - Optional learning-rate decay schedule (`lr_schedule`, see
      `resolve_scheduled_lr`): off by default -- a constant `lr` is used
      unless a schedule is explicitly given, so it's an isolated experiment
      rather than a silent behavior change.
    - Configurable network capacity (`network_size`, see
      `models.dqn_network.NETWORK_PRESETS`): tests whether a smaller network
      is inherently more stable over long runs, independent of the above.

Like the other agents, this one only ever reads the observation the
environment returns; it never inspects `env.game.mines` or any other hidden
environment state.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from models.dqn_network import NETWORK_PRESETS, DQNNetwork, encode_observation
from training.replay_buffer import ReplayBuffer

PathLike = Union[str, Path]
LRSchedule = List[Tuple[int, float]]


def resolve_scheduled_lr(schedule: LRSchedule, episode: int) -> float:
    """Return the learning rate in effect for `episode` under a milestone schedule.

    `schedule` is a list of `(episode_threshold, lr)` pairs, e.g.
    `[(0, 1e-4), (20000, 5e-5), (40000, 1e-5)]` meaning "1e-4 from episode 0,
    5e-5 from episode 20000, 1e-5 from episode 40000 onward." `episode` is
    0-indexed, matching `train()`'s loop variable. Pairs need not be
    pre-sorted; they're sorted here by threshold before resolving.
    """
    ordered = sorted(schedule, key=lambda pair: pair[0])
    current_lr = ordered[0][1]
    for threshold, lr in ordered:
        if episode >= threshold:
            current_lr = lr
        else:
            break
    return current_lr


def _average(step_metrics: List[Dict[str, float]], key: str) -> Optional[float]:
    """Mean of `key` across a list of per-step metric dicts, or None if empty."""
    if not step_metrics:
        return None
    return sum(m[key] for m in step_metrics) / len(step_metrics)


def _maximum(step_metrics: List[Dict[str, float]], key: str) -> Optional[float]:
    """Max of `key` across a list of per-step metric dicts, or None if empty."""
    if not step_metrics:
        return None
    return max(m[key] for m in step_metrics)


class DQNAgent:
    """A CNN-based Double DQN agent trained with experience replay and a target network."""

    def __init__(
        self,
        rows: int,
        cols: int,
        lr: float = 1e-4,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        target_update_every: int = 25,
        min_replay_size: int = 200,
        network_size: str = "default",
        lr_schedule: Optional[LRSchedule] = None,
        rolling_window: int = 100,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Create a DQN agent for a board of the given size.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            lr: Adam learning rate for the online network. Ignored (except as
                a fallback) if `lr_schedule` is given.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration probability.
            epsilon_min: Floor that epsilon decays towards.
            epsilon_decay: Multiplicative decay applied to epsilon after each
                training episode.
            buffer_capacity: Maximum number of transitions kept in replay.
            batch_size: Minibatch size sampled per gradient step.
            target_update_every: Episodes between target-network syncs.
            min_replay_size: Minimum transitions collected before training
                starts, so early batches aren't sampled from a handful of
                highly correlated transitions.
            network_size: Key into `models.dqn_network.NETWORK_PRESETS`
                controlling conv-filter counts and hidden-layer width.
            lr_schedule: Optional list of `(episode_threshold, lr)` pairs (see
                `resolve_scheduled_lr`). When given, `train()` updates the
                optimizer's learning rate at the start of each episode
                according to the schedule instead of holding `lr` constant.
            rolling_window: Number of trailing episodes used to compute the
                rolling mean reward and rolling win rate in training history.
            device: Torch device string, e.g. "cpu" or "cuda". Defaults to CPU.
            seed: Optional seed for reproducible exploration, replay sampling,
                and network initialization.
        """
        if network_size not in NETWORK_PRESETS:
            raise ValueError(f"Unknown network_size {network_size!r}; choose from {list(NETWORK_PRESETS)}")

        self.rows = rows
        self.cols = cols
        self.n_actions = rows * cols

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.min_replay_size = min_replay_size
        self.network_size = network_size
        self.lr_schedule: Optional[LRSchedule] = sorted(lr_schedule, key=lambda p: p[0]) if lr_schedule else None
        self.rolling_window = rolling_window

        self.device = torch.device(device or "cpu")
        self._rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)

        preset = NETWORK_PRESETS[network_size]
        self.online_network = DQNNetwork(rows, cols, **preset).to(self.device)
        self.target_network = DQNNetwork(rows, cols, **preset).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        initial_lr = resolve_scheduled_lr(self.lr_schedule, 0) if self.lr_schedule else lr
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=initial_lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, seed=seed)

    def select_action(self, state: Sequence[Sequence[int]], explore: bool = True) -> int:
        """Pick an action for `state` using epsilon-greedy selection over hidden cells.

        Only hidden cells are considered, since revealing an already-revealed
        cell is always a wasted move (reward 0, no state change). When
        `explore` is False the agent acts greedily with respect to the current
        Q-values, which is what evaluation should use.
        """
        board = np.asarray(state)
        hidden_actions = np.flatnonzero(board.flatten() == -1)
        if hidden_actions.size == 0:
            return self._rng.randrange(self.n_actions)

        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.choice(hidden_actions))

        with torch.no_grad():
            tensor = self._to_tensor(encode_observation(board)[None, ...])
            q_values = self.online_network(tensor).squeeze(0).cpu().numpy()

        masked_q = np.full(self.n_actions, -np.inf, dtype=np.float32)
        masked_q[hidden_actions] = q_values[hidden_actions]
        return int(np.argmax(masked_q))

    def remember(
        self,
        state: Sequence[Sequence[int]],
        action: int,
        reward: float,
        next_state: Sequence[Sequence[int]],
        done: bool,
    ) -> None:
        """Store one transition in the replay buffer, normalizing both states."""
        self.replay_buffer.push(
            encode_observation(state), action, float(reward), encode_observation(next_state), done
        )

    def _double_dqn_targets(self, next_states_t: torch.Tensor) -> torch.Tensor:
        """Compute the Double DQN bootstrap value for a batch of next-states.

        Action *selection* uses the online network
        (`argmax_a' Q_online(s', a')`); action *evaluation* uses the target
        network (`Q_target(s', that_action)`). See the module docstring for
        why splitting these two roles across networks reduces overestimation
        bias relative to plain DQN's single-network `max`.
        """
        next_actions = self.online_network(next_states_t).argmax(dim=1)
        return self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)

    def train_step(self) -> Optional[Dict[str, float]]:
        """Sample a batch and perform one gradient update on the online network.

        The target for each transition is `reward` alone when `done`, or
        `reward + gamma * Q_target(next_state, argmax_a' Q_online(next_state, a'))`
        otherwise (the Double DQN target; see `_double_dqn_targets`). The
        online network is fit to that target with Huber loss (see module
        docstring for why MSE was replaced).

        Returns:
            A dict of scalar diagnostics for this gradient step -- `loss`,
            `avg_q`/`max_q` (over the batch's taken-action Q-values),
            `grad_norm` (pre-clipping), and `td_error_mean`/`td_error_max`
            (|target - prediction|) -- or None if the replay buffer doesn't
            yet hold enough transitions to train.
        """
        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = self._to_tensor(states)
        next_states_t = self._to_tensor(next_states)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_values = self.online_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self._double_dqn_targets(next_states_t)
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)
            td_errors = targets - q_values

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Q-learning's max-based bootstrap target can produce occasional large
        # gradients that compound through the target network and diverge
        # (loss growing without bound); clipping the gradient norm is the
        # standard DQN fix and keeps training stable. clip_grad_norm_ returns
        # the pre-clipping total norm, which we surface as a diagnostic.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "avg_q": float(q_values.mean().item()),
            "max_q": float(q_values.max().item()),
            "grad_norm": float(grad_norm),
            "td_error_mean": float(td_errors.abs().mean().item()),
            "td_error_max": float(td_errors.abs().max().item()),
        }

    def update_target_network(self) -> None:
        """Copy the online network's weights into the target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay epsilon towards `epsilon_min` after a training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, path: PathLike, episode: int, win_rate: float, average_reward: float) -> None:
        """Save the online network's weights plus evaluation metadata to `path`.

        Metadata (`episode`, `win_rate`, `average_reward`, `timestamp`) lets a
        later run report when and how well this checkpoint was performing
        without re-running evaluation just to decide whether it's worth
        loading.
        """
        checkpoint = {
            "model_state_dict": self.online_network.state_dict(),
            "rows": self.rows,
            "cols": self.cols,
            "network_size": self.network_size,
            "metadata": {
                "episode": episode,
                "win_rate": win_rate,
                "average_reward": average_reward,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: PathLike, load_into_target: bool = True) -> Dict[str, Any]:
        """Load a checkpoint saved by `save_checkpoint` into the online network.

        Args:
            path: Checkpoint file written by `save_checkpoint`.
            load_into_target: Also sync the target network to the loaded
                weights -- the natural choice when loading a checkpoint to
                resume training or to serve as the deployed evaluation policy.

        Returns:
            The checkpoint's metadata dict (`episode`, `win_rate`,
            `average_reward`, `timestamp`).
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.online_network.load_state_dict(checkpoint["model_state_dict"])
        if load_into_target:
            self.target_network.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["metadata"]

    def _evaluate_greedy(self, env: Any, episodes: int) -> Dict[str, float]:
        """Run `episodes` greedy (epsilon=0) episodes and return win rate / avg reward.

        Used internally for periodic checkpoint evaluation during training.
        Kept self-contained (rather than importing `evaluation.metrics`) so
        `DQNAgent` has no dependency on the evaluation layer.
        """
        wins = 0
        total_reward = 0.0
        for _ in range(episodes):
            observation, info = env.reset()
            terminated = truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action = self.select_action(observation, explore=False)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
            total_reward += episode_reward
            if info.get("won", False):
                wins += 1
        return {"win_rate": wins / episodes, "avg_reward": total_reward / episodes}

    def train(
        self,
        env: Any,
        episodes: int,
        checkpoint_dir: Optional[PathLike] = None,
        checkpoint_every: int = 500,
        checkpoint_eval_episodes: int = 50,
    ) -> List[Dict[str, Any]]:
        """Run `episodes` training episodes of epsilon-greedy play with replay.

        Each step: act, store the transition, and take one gradient step once
        enough experience has been collected. The target network is synced
        every `target_update_every` episodes, epsilon decays once per
        episode, and (if `lr_schedule` was given) the optimizer's learning
        rate is updated to match the schedule for the current episode.

        If `checkpoint_dir` is given, every `checkpoint_every` episodes the
        current greedy policy is evaluated for `checkpoint_eval_episodes`
        episodes; if its win rate beats every previous evaluation this run,
        the online network is saved to `checkpoint_dir/best_model.pt`. At the
        end of training, the current weights are always saved to
        `checkpoint_dir/final_model.pt`, regardless of whether they're the
        best seen. `checkpoint_dir=None` (the default) disables all of this.

        Returns:
            A per-episode history list. Each entry has: `total_reward`,
            `steps`, `won`, `epsilon`, `lr`, `loss`, `avg_q`, `max_q`,
            `grad_norm`, `td_error_mean`, `td_error_max` (all averaged over
            the episode's gradient steps, except `max_q`/`td_error_max`
            which are the max over those steps, and all None if no gradient
            step happened during the episode), plus `reward_rolling_mean`
            and `win_rate_rolling` (over the trailing `rolling_window`
            episodes, inclusive of the current one).
        """
        history: List[Dict[str, Any]] = []

        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_path is not None:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        best_win_rate = -1.0

        for episode in range(episodes):
            if self.lr_schedule is not None:
                scheduled_lr = resolve_scheduled_lr(self.lr_schedule, episode)
                for group in self.optimizer.param_groups:
                    group["lr"] = scheduled_lr

            observation, info = env.reset()
            terminated = truncated = False
            total_reward = 0.0
            steps = 0
            step_metrics: List[Dict[str, float]] = []

            while not (terminated or truncated):
                action = self.select_action(observation, explore=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                self.remember(observation, action, reward, next_observation, terminated or truncated)

                metrics = self.train_step()
                if metrics is not None:
                    step_metrics.append(metrics)

                observation = next_observation
                total_reward += reward
                steps += 1

            self.decay_epsilon()
            if (episode + 1) % self.target_update_every == 0:
                self.update_target_network()

            won = bool(info.get("won", False))
            window = history[-(self.rolling_window - 1) :] if self.rolling_window > 1 else []
            recent_rewards = [h["total_reward"] for h in window] + [total_reward]
            recent_wins = [h["won"] for h in window] + [won]

            history.append(
                {
                    "total_reward": total_reward,
                    "steps": steps,
                    "won": won,
                    "epsilon": self.epsilon,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "loss": _average(step_metrics, "loss"),
                    "avg_q": _average(step_metrics, "avg_q"),
                    "max_q": _maximum(step_metrics, "max_q"),
                    "grad_norm": _average(step_metrics, "grad_norm"),
                    "td_error_mean": _average(step_metrics, "td_error_mean"),
                    "td_error_max": _maximum(step_metrics, "td_error_max"),
                    "reward_rolling_mean": sum(recent_rewards) / len(recent_rewards),
                    "win_rate_rolling": sum(recent_wins) / len(recent_wins),
                }
            )

            if checkpoint_path is not None and (episode + 1) % checkpoint_every == 0:
                eval_result = self._evaluate_greedy(env, checkpoint_eval_episodes)
                if eval_result["win_rate"] > best_win_rate:
                    best_win_rate = eval_result["win_rate"]
                    self.save_checkpoint(
                        checkpoint_path / "best_model.pt",
                        episode=episode + 1,
                        win_rate=eval_result["win_rate"],
                        average_reward=eval_result["avg_reward"],
                    )

        if checkpoint_path is not None:
            final_eval = self._evaluate_greedy(env, checkpoint_eval_episodes)
            self.save_checkpoint(
                checkpoint_path / "final_model.pt",
                episode=episodes,
                win_rate=final_eval["win_rate"],
                average_reward=final_eval["avg_reward"],
            )

        return history

    def _to_tensor(self, states: np.ndarray) -> torch.Tensor:
        """Convert a (N, channels, rows, cols) float array into a tensor on `self.device`."""
        return torch.from_numpy(states).to(self.device)
