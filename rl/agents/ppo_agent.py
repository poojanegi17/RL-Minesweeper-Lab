"""Proximal Policy Optimization (PPO) agent for Minesweeper.

Where `DQNAgent` learns Q(s, a) and acts by taking `argmax_a Q(s, a)` over an
off-policy replay buffer, `PPOAgent` directly parameterizes a policy
pi(a | s) (the actor) and learns it on-policy: every update uses only
trajectories collected under the current policy, then discards them --
unlike DQN, old data can't be replayed once the policy that generated it is
gone, because PPO's clipped objective is a correction for how far a *new*
policy has drifted from the one that collected the data, and that
correction stops being meaningful once the drift is large. Consequently
this agent has no replay buffer and no target network; instead it has:

    - `models.ppo_network.PPONetwork`: a shared CNN trunk (reusing
      `models.dqn_network.encode_observation`'s 11-channel input, for the
      same reason DQN uses it -- Minesweeper's clues are categorical, not
      continuous) with two heads -- an actor producing one logit per cell,
      and a critic producing a scalar V(s), since policy-gradient methods
      need "what should I do" and "how good is this state" as separate
      quantities.
    - `training.rollout_buffer.RolloutBuffer`: an on-policy buffer holding
      exactly one rollout, discarded after the update it feeds.
    - Generalized Advantage Estimation (GAE): blends the low-variance,
      high-bias one-step TD advantage with the high-variance, low-bias
      full-Monte-Carlo advantage via `gae_lambda`, giving a tunable
      bias/variance tradeoff that plain REINFORCE-style returns don't offer.
    - The clipped surrogate objective: policy-gradient updates without a
      trust-region constraint can take a single bad step that collapses the
      policy (there's no equivalent of DQN's target network to keep the
      bootstrap from moving too fast). Clipping the probability ratio
      `pi_new(a|s) / pi_old(a|s)` to `[1 - eps, 1 + eps]` inside the
      objective's `min` term caps how much a single update can exploit an
      advantage estimate for one action, which is PPO's substitute for a
      hard trust-region constraint.

Like every other agent in this project, `PPOAgent` only ever reads the
observation the environment returns; it never inspects `env.game.mines` or
any other hidden environment state. Actions are restricted to hidden cells
by masking the actor's logits before sampling, mirroring
`DQNAgent.select_action` and `QLearningAgent.choose_action`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models.dqn_network import encode_observation
from models.ppo_network import PPONetwork
from training.rollout_buffer import RolloutBuffer

# Gradient-norm cap for the PPO update, mirroring DQNAgent's
# clip_grad_norm_ use for the same reason (bound the damage a single noisy
# batch's gradient can do). 0.5 is the standard PPO default.
MAX_GRAD_NORM = 0.5


class PPOAgent:
    """A CNN Actor-Critic agent trained with clipped-surrogate PPO."""

    def __init__(
        self,
        rows: int,
        cols: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        rollout_length: int = 256,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Create a PPO agent for a board of the given size.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            lr: Adam learning rate for the shared actor-critic network.
            gamma: Discount factor for future rewards.
            gae_lambda: GAE's bias/variance tradeoff parameter (see module
                docstring); 0 reduces to one-step TD advantage, 1 to
                full-Monte-Carlo advantage.
            clip_epsilon: PPO's probability-ratio clip range `[1-eps, 1+eps]`.
            entropy_coef: Weight of the entropy bonus in the total loss,
                encouraging continued exploration by penalizing an overly
                peaked policy.
            value_coef: Weight of the critic's value loss in the total loss.
            rollout_length: Number of environment steps collected per
                rollout before each PPO update. A rollout may span multiple
                episodes (Minesweeper episodes here are short) or end
                mid-episode, in which case the value function bootstraps the
                remainder.
            ppo_epochs: Number of passes over each rollout's data per update.
            batch_size: Minibatch size within each PPO epoch.
            device: Torch device string, e.g. "cpu" or "cuda". Defaults to CPU.
            seed: Optional seed for reproducible action sampling, minibatch
                shuffling, and network initialization.
        """
        self.rows = rows
        self.cols = cols
        self.n_actions = rows * cols

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_length = rollout_length
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.device = torch.device(device or "cpu")
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.network = PPONetwork(rows, cols).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert a (N, channels, rows, cols) float array into a tensor on `self.device`."""
        return torch.from_numpy(array).to(self.device)

    @staticmethod
    def _apply_action_mask(logits: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        """Mask actor logits to only hidden cells, using the encoding's own hidden-mask channel.

        `encoded` channel 0 is exactly `encode_observation`'s hidden-cell
        mask, so it doubles as the action mask without needing a separately
        stored mask array. Revealed cells get `-inf` logits, i.e. zero
        probability after softmax. If a row has no hidden cells at all (not
        expected mid-episode, since the game only terminates once every
        mine-free cell is revealed or a mine is hit), masking everything to
        `-inf` would make the categorical distribution undefined, so that
        row is left unmasked instead of producing NaNs.
        """
        batch = encoded.shape[0]
        hidden_mask = encoded[:, 0, :, :].reshape(batch, -1) > 0.5
        masked = logits.masked_fill(~hidden_mask, float("-inf"))
        no_hidden_cells = ~hidden_mask.any(dim=1)
        if bool(no_hidden_cells.any()):
            masked = torch.where(no_hidden_cells.unsqueeze(1), logits, masked)
        return masked

    def select_action(self, observation: Sequence[Sequence[int]], explore: bool = True) -> int:
        """Pick an action for `observation` under the current policy.

        When `explore` is True (the default, used during rollout collection
        as well as ad-hoc play), an action is sampled from the masked
        categorical policy distribution -- PPO's exploration is inherent to
        its stochastic policy, unlike DQN's separate epsilon-greedy
        schedule. When `explore` is False (evaluation), the action is the
        mode of that distribution (`argmax` over masked logits), matching
        the "greedy at evaluation time" convention every other agent in this
        project uses.
        """
        board = np.asarray(observation)
        encoded = encode_observation(board)
        encoded_t = self._to_tensor(encoded[None, ...])

        with torch.no_grad():
            logits, _ = self.network(encoded_t)
            masked_logits = self._apply_action_mask(logits, encoded_t)
            if explore:
                action = Categorical(logits=masked_logits).sample()
            else:
                action = masked_logits.argmax(dim=1)

        return int(action.item())

    def act(self, observation: Sequence[Sequence[int]]) -> Tuple[int, float, float]:
        """Sample an action and return it with its log-probability and value estimate.

        Used internally during rollout collection, where the log-probability
        under the *acting* policy and the critic's value estimate both need
        to be stored alongside the transition (see `training.rollout_buffer`).
        """
        board = np.asarray(observation)
        encoded = encode_observation(board)
        encoded_t = self._to_tensor(encoded[None, ...])

        with torch.no_grad():
            logits, value = self.network(encoded_t)
            masked_logits = self._apply_action_mask(logits, encoded_t)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def _update(self) -> Dict[str, float]:
        """Run `ppo_epochs` passes of minibatch clipped-surrogate updates over the current rollout.

        For each minibatch: recompute action log-probabilities and values
        under the *current* (possibly already-updated-this-call) network,
        form the probability ratio against the log-probabilities recorded at
        collection time, and take the clipped-surrogate policy loss plus a
        value-regression loss plus an entropy bonus as the total loss (see
        module docstring for why each term is there).

        Returns:
            Average `policy_loss`, `value_loss`, `entropy`, `approx_kl`
            (mean `old_log_prob - new_log_prob`, a cheap divergence
            diagnostic), and `clip_fraction` (fraction of ratios that hit the
            clip boundary) across all epochs/minibatches in this update.
        """
        data = self.buffer.get()
        observations = self._to_tensor(data["observations"])
        actions = torch.from_numpy(data["actions"]).to(self.device)
        old_log_probs = torch.from_numpy(data["log_probs"]).to(self.device)
        returns = torch.from_numpy(data["returns"]).to(self.device)

        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.from_numpy(advantages.astype(np.float32)).to(self.device)

        num_steps = len(data["actions"])
        indices = np.arange(num_steps)

        stats: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": [],
        }

        for _ in range(self.ppo_epochs):
            self._rng.shuffle(indices)
            for start in range(0, num_steps, self.batch_size):
                batch_idx = torch.from_numpy(indices[start : start + self.batch_size]).to(self.device)

                batch_obs = observations[batch_idx]
                logits, values = self.network(batch_obs)
                masked_logits = self._apply_action_mask(logits, batch_obs)
                dist = Categorical(logits=masked_logits)

                batch_actions = actions[batch_idx]
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                batch_advantages = advantages[batch_idx]
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), returns[batch_idx])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=MAX_GRAD_NORM)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs[batch_idx] - new_log_probs).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()

                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.item()))
                stats["approx_kl"].append(float(approx_kl.item()))
                stats["clip_fraction"].append(float(clip_fraction.item()))

        return {key: sum(values_) / len(values_) for key, values_ in stats.items()}

    def train(
        self,
        env: Any,
        episodes: int,
        rollout_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Train for approximately `episodes` episodes via repeated rollout-then-update cycles.

        Each cycle: collect `rollout_length` environment steps (which may
        span multiple episodes, or end mid-episode), bootstrap the value of
        whatever state the rollout stopped at (0.0 if that state is
        terminal), compute GAE advantages/returns over the whole rollout,
        and run `_update()`. This repeats until at least `episodes` episodes
        have completed -- since a rollout's length is fixed rather than
        episode-aligned, the final rollout may finish a few episodes past
        the requested budget rather than stopping exactly on it.

        Returns:
            A per-episode history list, one entry per completed episode:
            `total_reward`, `steps`, `won`, `lr`, plus the most recently
            completed update's `policy_loss`, `value_loss`, `entropy`,
            `approx_kl`, `clip_fraction` (all `None` until the first update
            finishes, then carried forward until the next one).
        """
        rollout_length = rollout_length if rollout_length is not None else self.rollout_length

        history: List[Dict[str, Any]] = []
        observation, info = env.reset()
        completed_episodes = 0
        episode_reward = 0.0
        episode_steps = 0
        done = False
        last_metrics: Dict[str, Optional[float]] = {
            "policy_loss": None, "value_loss": None, "entropy": None, "approx_kl": None, "clip_fraction": None,
        }

        while completed_episodes < episodes:
            self.buffer.reset()

            for _ in range(rollout_length):
                action, log_prob, value = self.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.buffer.add(encode_observation(observation), action, reward, done, log_prob, value)

                episode_reward += reward
                episode_steps += 1
                observation = next_observation

                if done:
                    history.append(
                        {
                            "total_reward": episode_reward,
                            "steps": episode_steps,
                            "won": bool(info.get("won", False)),
                            "lr": self.optimizer.param_groups[0]["lr"],
                            **last_metrics,
                        }
                    )
                    completed_episodes += 1
                    episode_reward = 0.0
                    episode_steps = 0
                    observation, info = env.reset()
                    if completed_episodes >= episodes:
                        break

            with torch.no_grad():
                if done:
                    bootstrap_value = 0.0
                else:
                    encoded = encode_observation(observation)
                    _, value_t = self.network(self._to_tensor(encoded[None, ...]))
                    bootstrap_value = float(value_t.item())

            self.buffer.compute_gae(bootstrap_value, self.gamma, self.gae_lambda)
            last_metrics = self._update()

        return history
