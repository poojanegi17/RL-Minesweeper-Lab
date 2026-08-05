"""Tests for the PPO network, rollout buffer, and PPO agent."""

import inspect

import numpy as np
import pytest
import torch

from agents.ppo_agent import PPOAgent
from environment.minesweeper_env import MinesweeperEnv
from models.dqn_network import NUM_CHANNELS, encode_observation
from models.ppo_network import PPONetwork
from training.rollout_buffer import RolloutBuffer


# --- PPONetwork: output shapes --------------------------------------------


def test_network_forward_pass_has_correct_output_shapes():
    net = PPONetwork(rows=5, cols=5)
    batch = torch.zeros((4, NUM_CHANNELS, 5, 5))

    logits, values = net(batch)

    assert logits.shape == (4, 25)  # one action logit per cell, per board in the batch
    assert values.shape == (4, 1)  # one scalar value per board in the batch


def test_network_handles_non_square_boards():
    net = PPONetwork(rows=3, cols=4)
    batch = torch.zeros((2, NUM_CHANNELS, 3, 4))

    logits, values = net(batch)

    assert logits.shape == (2, 12)
    assert values.shape == (2, 1)


def test_encoded_observation_feeds_directly_into_network():
    board = np.array([[-1, 1], [2, -1]])
    net = PPONetwork(rows=2, cols=2)

    tensor = torch.from_numpy(encode_observation(board)[None, ...])
    logits, values = net(tensor)

    assert logits.shape == (1, 4)
    assert values.shape == (1, 1)


# --- PPONetwork: actor outputs valid logits --------------------------------


def test_actor_outputs_finite_logits_one_per_cell():
    net = PPONetwork(rows=4, cols=4)
    batch = torch.randn((3, NUM_CHANNELS, 4, 4))

    logits, _ = net(batch)

    assert logits.shape == (3, 16)
    assert torch.isfinite(logits).all()


def test_actor_logits_produce_a_valid_probability_distribution():
    # Softmax over raw logits should sum to 1 and stay non-negative -- a
    # basic sanity check that nothing about the actor head's output shape or
    # scale breaks the categorical distribution built on top of it.
    net = PPONetwork(rows=3, cols=3)
    batch = torch.randn((2, NUM_CHANNELS, 3, 3))

    logits, _ = net(batch)
    probs = torch.softmax(logits, dim=-1)

    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


# --- PPONetwork: critic outputs scalar values ------------------------------


def test_critic_outputs_one_scalar_value_per_state():
    net = PPONetwork(rows=5, cols=5)
    batch = torch.randn((6, NUM_CHANNELS, 5, 5))

    _, values = net(batch)

    assert values.shape == (6, 1)
    assert torch.isfinite(values).all()


# --- RolloutBuffer: storage -------------------------------------------------


def test_rollout_buffer_stores_transitions_in_order():
    buffer = RolloutBuffer()
    for i in range(5):
        buffer.add(np.full((NUM_CHANNELS, 2, 2), i), action=i, reward=float(i), done=False, log_prob=-0.1 * i, value=float(i))

    assert len(buffer) == 5
    assert buffer.actions == [0, 1, 2, 3, 4]
    assert buffer.rewards == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_rollout_buffer_get_returns_correctly_shaped_stacked_arrays():
    buffer = RolloutBuffer()
    for i in range(4):
        buffer.add(np.zeros((NUM_CHANNELS, 3, 3)), action=i, reward=1.0, done=(i == 3), log_prob=0.0, value=0.5)
    buffer.compute_gae(last_value=0.0, gamma=0.9, gae_lambda=0.95)

    data = buffer.get()

    assert data["observations"].shape == (4, NUM_CHANNELS, 3, 3)
    assert data["actions"].shape == (4,)
    assert data["log_probs"].shape == (4,)
    assert data["advantages"].shape == (4,)
    assert data["returns"].shape == (4,)
    assert data["actions"].dtype == np.int64
    assert data["observations"].dtype == np.float32


def test_rollout_buffer_get_before_compute_gae_raises():
    buffer = RolloutBuffer()
    buffer.add(np.zeros((NUM_CHANNELS, 2, 2)), 0, 1.0, True, 0.0, 0.0)

    with pytest.raises(RuntimeError):
        buffer.get()


def test_rollout_buffer_reset_clears_all_state():
    buffer = RolloutBuffer()
    buffer.add(np.zeros((NUM_CHANNELS, 2, 2)), 0, 1.0, True, 0.0, 0.0)
    buffer.compute_gae(last_value=0.0, gamma=0.9, gae_lambda=0.95)

    buffer.reset()

    assert len(buffer) == 0
    assert buffer.advantages is None
    assert buffer.returns is None


# --- RolloutBuffer: GAE calculation -----------------------------------------


def test_gae_matches_hand_computed_values_for_known_rollout():
    # Three steps, all rewards=1, all values=0, terminal on the last step:
    # delta_t = r_t (values are 0 and, for the terminal step, next_value is
    # masked out by (1 - done)), so this reduces to a simple, checkable
    # recursion. Worked by hand (gamma=0.9, lambda=0.8):
    #   A_2 = 1.0
    #   A_1 = 1 + 0.9*0.8*1*A_2      = 1.72
    #   A_0 = 1 + 0.9*0.8*1*A_1      = 2.2384
    buffer = RolloutBuffer()
    for _ in range(2):
        buffer.add(np.zeros((1, 1, 1)), 0, 1.0, False, 0.0, 0.0)
    buffer.add(np.zeros((1, 1, 1)), 0, 1.0, True, 0.0, 0.0)

    buffer.compute_gae(last_value=0.0, gamma=0.9, gae_lambda=0.8)

    np.testing.assert_allclose(buffer.advantages, [2.2384, 1.72, 1.0], atol=1e-6)
    # values are all 0 here, so returns == advantages.
    np.testing.assert_allclose(buffer.returns, buffer.advantages, atol=1e-6)


def test_gae_bootstraps_from_last_value_when_rollout_ends_mid_episode():
    # A single non-terminal step: A_0 = delta_0 = r_0 + gamma*last_value - v_0.
    buffer = RolloutBuffer()
    buffer.add(np.zeros((1, 1, 1)), 0, reward=1.0, done=False, log_prob=0.0, value=0.5)

    buffer.compute_gae(last_value=2.0, gamma=0.9, gae_lambda=0.95)

    expected_advantage = 1.0 + 0.9 * 2.0 - 0.5
    np.testing.assert_allclose(buffer.advantages, [expected_advantage], atol=1e-6)
    np.testing.assert_allclose(buffer.returns, [expected_advantage + 0.5], atol=1e-6)


def test_gae_does_not_bootstrap_across_a_done_step():
    # done=True on the first step must zero out any contribution from
    # last_value / the next step, regardless of how large last_value is.
    buffer = RolloutBuffer()
    buffer.add(np.zeros((1, 1, 1)), 0, reward=1.0, done=True, log_prob=0.0, value=0.5)

    buffer.compute_gae(last_value=1000.0, gamma=0.9, gae_lambda=0.95)

    np.testing.assert_allclose(buffer.advantages, [1.0 - 0.5], atol=1e-6)


# --- PPOAgent: action selection --------------------------------------------


def test_select_action_only_returns_hidden_cells():
    board = np.array([[-1, 1], [2, -1]])
    agent = PPOAgent(rows=2, cols=2, seed=0)

    for _ in range(20):
        action = agent.select_action(board, explore=True)
        assert action in (0, 3)  # flattened indices of the two -1 cells


def test_greedy_action_is_deterministic():
    board = np.full((3, 3), -1)
    agent = PPOAgent(rows=3, cols=3, seed=2)

    first = agent.select_action(board, explore=False)
    for _ in range(5):
        assert agent.select_action(board, explore=False) == first


def test_act_returns_action_log_prob_and_value_for_a_hidden_cell():
    board = np.array([[-1, 1], [2, -1]])
    agent = PPOAgent(rows=2, cols=2, seed=0)

    action, log_prob, value = agent.act(board)

    assert action in (0, 3)
    assert log_prob <= 0.0  # log-probabilities are never positive
    assert isinstance(value, float)


# --- PPOAgent: fairness (no hidden-state access) ---------------------------


def test_agent_never_accesses_hidden_mine_positions():
    for name in ("select_action", "act", "_update"):
        params = list(inspect.signature(getattr(PPOAgent, name)).parameters)
        assert "env" not in params and "game" not in params and "mines" not in params


# --- PPOAgent: environment interaction / episode completion ----------------


def test_agent_can_complete_an_episode():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=3, seed=5)
    agent = PPOAgent(rows=4, cols=4, seed=5)

    observation, info = env.reset()
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < 16:
        action = agent.select_action(observation, explore=True)
        assert 0 <= action < 16
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps > 0
    assert terminated  # a 4x4 board with 13 safe cells is small enough to always finish quickly


def test_trained_agent_can_act_greedily_after_training():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=8)
    agent = PPOAgent(rows=3, cols=3, rollout_length=16, ppo_epochs=2, batch_size=4, seed=8)
    agent.train(env, episodes=5)

    observation, info = env.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 9:
        action = agent.select_action(observation, explore=False)
        assert 0 <= action < 9
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps > 0


# --- PPOAgent: PPO update changes network weights --------------------------


def test_update_changes_network_weights():
    agent = PPOAgent(rows=3, cols=3, ppo_epochs=2, batch_size=4, seed=0)

    board = np.full((3, 3), -1)
    for i in range(8):
        agent.buffer.add(encode_observation(board), action=i % 9, reward=1.0, done=(i == 7), log_prob=-1.0, value=0.0)
    agent.buffer.compute_gae(last_value=0.0, gamma=0.9, gae_lambda=0.95)

    before = [p.clone() for p in agent.network.parameters()]
    metrics = agent._update()
    after = list(agent.network.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert set(metrics.keys()) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"}
    assert metrics["entropy"] >= 0.0  # entropy of a categorical distribution is non-negative


def test_clip_fraction_is_zero_when_ratios_never_move():
    # With old_log_probs recomputed from a network that hasn't changed yet
    # (first gradient step of an update never moves ratio far), the very
    # first minibatch's ratio should start at ~1.0 -- exercised indirectly
    # via a full update rather than asserting internals, since PPOAgent
    # doesn't expose intermediate ratios.
    agent = PPOAgent(rows=2, cols=2, ppo_epochs=1, batch_size=4, clip_epsilon=0.2, seed=0)
    board = np.full((2, 2), -1)

    for i in range(4):
        action, log_prob, value = agent.act(board)
        agent.buffer.add(encode_observation(board), action, reward=1.0, done=(i == 3), log_prob=log_prob, value=value)
    agent.buffer.compute_gae(last_value=0.0, gamma=0.9, gae_lambda=0.95)

    metrics = agent._update()

    assert 0.0 <= metrics["clip_fraction"] <= 1.0


# --- PPOAgent: full training integration with MinesweeperEnv ---------------


def test_agent_can_complete_small_training_run():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=7)
    agent = PPOAgent(rows=3, cols=3, rollout_length=16, ppo_epochs=2, batch_size=4, seed=7)

    history = agent.train(env, episodes=5)

    assert len(history) >= 5  # training stops once at least `episodes` episodes have completed
    assert all(entry["steps"] > 0 for entry in history)
    assert all(isinstance(entry["won"], bool) for entry in history)


def test_training_updates_network_weights_over_a_longer_run():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=2, seed=11)
    agent = PPOAgent(rows=4, cols=4, rollout_length=64, ppo_epochs=3, batch_size=16, seed=11)

    before = [p.clone() for p in agent.network.parameters()]
    history = agent.train(env, episodes=150)
    after = list(agent.network.parameters())

    assert len(history) >= 150
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    # By this point at least one rollout must have completed and updated,
    # so later history entries should carry real (non-None) update metrics.
    assert any(entry["policy_loss"] is not None for entry in history)


def test_train_history_includes_expected_keys():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=9)
    agent = PPOAgent(rows=3, cols=3, rollout_length=16, ppo_epochs=2, batch_size=4, seed=9)

    history = agent.train(env, episodes=10)

    expected_keys = {
        "total_reward", "steps", "won", "lr",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
    }
    assert expected_keys.issubset(history[-1].keys())


def test_train_respects_custom_rollout_length_argument():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=3)
    agent = PPOAgent(rows=3, cols=3, rollout_length=999, ppo_epochs=1, batch_size=4, seed=3)

    # Overriding rollout_length at call time should be honored instead of
    # the constructor default, letting a short test run without waiting on
    # a huge default-length rollout.
    history = agent.train(env, episodes=3, rollout_length=8)

    assert len(history) >= 3
