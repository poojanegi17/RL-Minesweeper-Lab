"""Tests for the DQN network, replay buffer, and DQN agent."""

import inspect

import numpy as np
import pytest
import torch

from agents.dqn_agent import DQNAgent, resolve_scheduled_lr
from environment.minesweeper_env import MinesweeperEnv
from models.dqn_network import NETWORK_PRESETS, NUM_CHANNELS, DQNNetwork, encode_observation
from training.replay_buffer import ReplayBuffer


# --- DQNNetwork ---------------------------------------------------------


def test_encode_observation_produces_correct_channel_shape_and_dtype():
    board = np.array([[-1, 0, 3], [2, -1, 1]])
    encoded = encode_observation(board)

    assert encoded.shape == (NUM_CHANNELS, 2, 3)
    assert encoded.dtype == np.float32


def test_encode_observation_hidden_cells_activate_hidden_channel_only():
    board = np.array([[-1, 0, 3], [2, -1, 1]])
    encoded = encode_observation(board)

    for r, c in np.argwhere(board == -1):
        assert encoded[0, r, c] == 1.0  # hidden mask
        assert encoded[1, r, c] == 0.0  # revealed mask
        assert np.all(encoded[2:, r, c] == 0.0)  # no one-hot value channel lit


def test_encode_observation_revealed_cells_activate_revealed_and_value_channels():
    board = np.array([[-1, 0, 3], [2, -1, 1]])
    encoded = encode_observation(board)

    for r, c in np.argwhere(board != -1):
        value = board[r, c]
        assert encoded[0, r, c] == 0.0  # hidden mask
        assert encoded[1, r, c] == 1.0  # revealed mask
        # Exactly the channel for this value is lit among the one-hot channels.
        one_hot = encoded[2:, r, c]
        assert one_hot[value] == 1.0
        assert one_hot.sum() == 1.0


def test_encode_observation_matches_worked_example():
    # From the design spec: hidden -> [1,0,...,0]; revealed value 3 -> revealed
    # mask set plus the channel for value 3 (index 2+3=5) set.
    board = np.array([[-1, 3]])
    encoded = encode_observation(board)

    hidden_vector = encoded[:, 0, 0]
    expected_hidden = np.zeros(NUM_CHANNELS, dtype=np.float32)
    expected_hidden[0] = 1.0
    np.testing.assert_array_equal(hidden_vector, expected_hidden)

    revealed_vector = encoded[:, 0, 1]
    expected_revealed = np.zeros(NUM_CHANNELS, dtype=np.float32)
    expected_revealed[1] = 1.0
    expected_revealed[2 + 3] = 1.0
    np.testing.assert_array_equal(revealed_vector, expected_revealed)


def test_network_forward_pass_has_correct_output_shape():
    net = DQNNetwork(rows=5, cols=5)
    batch = torch.zeros((4, NUM_CHANNELS, 5, 5))

    output = net(batch)

    assert output.shape == (4, 25)  # one Q-value per cell, for every board in the batch


def test_network_handles_non_square_boards():
    net = DQNNetwork(rows=3, cols=4)
    batch = torch.zeros((2, NUM_CHANNELS, 3, 4))

    output = net(batch)

    assert output.shape == (2, 12)


def test_encoded_observation_feeds_directly_into_network():
    board = np.array([[-1, 1], [2, -1]])
    net = DQNNetwork(rows=2, cols=2)

    tensor = torch.from_numpy(encode_observation(board)[None, ...])
    output = net(tensor)

    assert output.shape == (1, 4)


# --- DQNNetwork: configurable architecture --------------------------------


def test_network_accepts_custom_conv_channels_and_hidden_dim():
    net = DQNNetwork(rows=4, cols=4, conv_channels=(8, 16), hidden_dim=64)
    batch = torch.zeros((2, NUM_CHANNELS, 4, 4))

    output = net(batch)

    assert output.shape == (2, 16)  # rows * cols, unaffected by conv/hidden width
    assert net.conv[0].out_channels == 8
    assert net.conv[2].out_channels == 16
    assert net.head[1].out_features == 64


def test_network_presets_are_registered_and_differ_in_size():
    assert "default" in NETWORK_PRESETS
    assert "small" in NETWORK_PRESETS

    default_net = DQNNetwork(rows=5, cols=5, **NETWORK_PRESETS["default"])
    small_net = DQNNetwork(rows=5, cols=5, **NETWORK_PRESETS["small"])

    default_params = sum(p.numel() for p in default_net.parameters())
    small_params = sum(p.numel() for p in small_net.parameters())
    assert small_params < default_params


def test_dqn_agent_small_network_preset_has_fewer_parameters():
    default_agent = DQNAgent(rows=4, cols=4, network_size="default", seed=0)
    small_agent = DQNAgent(rows=4, cols=4, network_size="small", seed=0)

    default_params = sum(p.numel() for p in default_agent.online_network.parameters())
    small_params = sum(p.numel() for p in small_agent.online_network.parameters())

    assert small_params < default_params


def test_dqn_agent_rejects_unknown_network_size():
    with pytest.raises(ValueError):
        DQNAgent(rows=4, cols=4, network_size="huge")


# --- ReplayBuffer --------------------------------------------------------


def test_replay_buffer_stores_and_samples_batches():
    buffer = ReplayBuffer(capacity=10, seed=0)
    for i in range(5):
        buffer.push(np.zeros((3, 3)), i, float(i), np.ones((3, 3)), False)

    assert len(buffer) == 5

    states, actions, rewards, next_states, dones = buffer.sample(3)
    assert states.shape == (3, 3, 3)
    assert next_states.shape == (3, 3, 3)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert actions.dtype == np.int64
    assert states.dtype == np.float32


def test_replay_buffer_evicts_oldest_at_capacity():
    buffer = ReplayBuffer(capacity=3, seed=0)
    for i in range(10):
        buffer.push(np.full((2, 2), i), i, 0.0, np.zeros((2, 2)), False)

    assert len(buffer) == 3
    # Only the last 3 pushed actions (7, 8, 9) should remain.
    _, actions, _, _, _ = buffer.sample(3)
    assert set(actions.tolist()) == {7, 8, 9}


# --- DQNAgent: action selection ------------------------------------------


def test_select_action_only_returns_hidden_cells():
    board = np.array([[-1, 1], [2, -1]])
    agent = DQNAgent(rows=2, cols=2, seed=0)

    for _ in range(20):
        action = agent.select_action(board, explore=True)
        assert action in (0, 3)  # flattened indices of the two -1 cells


def test_epsilon_one_always_explores_randomly():
    board = np.full((3, 3), -1)
    agent = DQNAgent(rows=3, cols=3, epsilon=1.0, seed=1)

    actions = {agent.select_action(board, explore=True) for _ in range(50)}
    assert len(actions) > 1


def test_epsilon_zero_is_deterministic_and_greedy():
    board = np.full((2, 2), -1)
    agent = DQNAgent(rows=2, cols=2, epsilon=0.0, seed=2)

    # With untouched network weights, repeated greedy calls must agree.
    first = agent.select_action(board, explore=True)
    for _ in range(5):
        assert agent.select_action(board, explore=True) == first


def test_agent_never_accesses_hidden_mine_positions():
    for name in ("select_action", "remember", "train_step"):
        params = list(inspect.signature(getattr(DQNAgent, name)).parameters)
        assert "env" not in params and "game" not in params and "mines" not in params


# --- DQNAgent: environment interaction ------------------------------------


def test_agent_interacts_correctly_with_minesweeper_env():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=3, seed=5)
    agent = DQNAgent(rows=4, cols=4, seed=5)

    observation, info = env.reset()
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < 16:
        action = agent.select_action(observation, explore=True)
        assert 0 <= action < 16
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps > 0


# --- DQNAgent: Double DQN target calculation ------------------------------


def test_double_dqn_target_uses_online_argmax_but_target_value():
    # If this were plain DQN (max over the target network alone), the target
    # value would be 10.0 (the target network's own max). Double DQN instead
    # asks the online network which action looks best, then reads *that*
    # action's value off the target network -- which here is a deliberately
    # different, lower value (3.0), so the two algorithms are distinguishable.
    agent = DQNAgent(rows=2, cols=2, seed=0)

    class StubOnline(torch.nn.Module):
        def forward(self, x):  # noqa: D102 - test stub
            batch = x.shape[0]
            return torch.tensor([[1.0, 5.0, 2.0, 0.0]] * batch)  # argmax -> action index 1

    class StubTarget(torch.nn.Module):
        def forward(self, x):  # noqa: D102 - test stub
            batch = x.shape[0]
            return torch.tensor([[10.0, 3.0, 7.0, 1.0]] * batch)  # value of action 1 is 3.0

    agent.online_network = StubOnline()
    agent.target_network = StubTarget()

    next_states = torch.zeros((2, 11, 2, 2))
    targets = agent._double_dqn_targets(next_states)

    assert torch.allclose(targets, torch.tensor([3.0, 3.0]))
    assert not torch.allclose(targets, torch.tensor([10.0, 10.0]))  # would be plain DQN's answer


def test_double_dqn_targets_used_inside_train_step():
    # End-to-end check that train_step's target actually equals reward (done=False
    # bootstrap) computed via _double_dqn_targets, not a plain max over the target net.
    agent = DQNAgent(rows=3, cols=3, batch_size=4, min_replay_size=4, gamma=0.9, seed=0)
    board = np.full((3, 3), -1)
    next_board = np.array([[1, -1, -1], [-1, -1, -1], [-1, -1, -1]])

    for _ in range(4):
        agent.remember(board, 0, 1.0, next_board, False)

    from models.dqn_network import encode_observation

    next_states_t = agent._to_tensor(
        np.stack([encode_observation(next_board)] * agent.batch_size)
    )
    with torch.no_grad():
        expected_next_q = agent._double_dqn_targets(next_states_t)
        expected_targets = 1.0 + agent.gamma * expected_next_q

        online_q_before = agent.online_network(agent._to_tensor(np.stack([encode_observation(board)] * agent.batch_size)))[:, 0]

    agent.train_step()

    # The loss should have pulled Q(board, 0) towards expected_targets, i.e.
    # closer to it than the pre-update prediction was.
    with torch.no_grad():
        online_q_after = agent.online_network(agent._to_tensor(np.stack([encode_observation(board)] * agent.batch_size)))[:, 0]

    before_gap = (online_q_before - expected_targets).abs()
    after_gap = (online_q_after - expected_targets).abs()
    assert (after_gap <= before_gap + 1e-4).all()


# --- DQNAgent: training ----------------------------------------------------


def test_train_step_returns_none_before_min_replay_size():
    agent = DQNAgent(rows=3, cols=3, batch_size=8, min_replay_size=8, seed=0)
    assert agent.train_step() is None


def test_train_step_updates_network_weights():
    agent = DQNAgent(rows=3, cols=3, batch_size=4, min_replay_size=4, seed=0)
    board = np.full((3, 3), -1)
    next_board = np.array([[1, -1, -1], [-1, -1, -1], [-1, -1, -1]])

    for _ in range(8):
        agent.remember(board, 0, 1.0, next_board, False)

    before = [p.clone() for p in agent.online_network.parameters()]
    metrics = agent.train_step()
    after = list(agent.online_network.parameters())

    assert metrics is not None
    assert metrics["loss"] >= 0.0
    assert metrics["grad_norm"] >= 0.0
    assert metrics["td_error_mean"] >= 0.0
    assert metrics["td_error_max"] >= metrics["td_error_mean"]
    assert "avg_q" in metrics and "max_q" in metrics
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_update_target_network_syncs_weights():
    agent = DQNAgent(rows=3, cols=3, seed=0)
    board = np.full((3, 3), -1)
    next_board = np.array([[1, -1, -1], [-1, -1, -1], [-1, -1, -1]])

    # Nudge the online network away from the target network.
    for _ in range(agent.min_replay_size):
        agent.remember(board, 0, 1.0, next_board, False)
    agent.batch_size = agent.min_replay_size
    agent.train_step()

    online_params = list(agent.online_network.parameters())
    target_params_before = [p.clone() for p in agent.target_network.parameters()]
    assert any(not torch.equal(o, t) for o, t in zip(online_params, target_params_before))

    agent.update_target_network()

    target_params_after = list(agent.target_network.parameters())
    assert all(torch.equal(o, t) for o, t in zip(online_params, target_params_after))


def test_agent_can_complete_small_training_run():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=7)
    agent = DQNAgent(
        rows=3, cols=3, batch_size=8, min_replay_size=8, target_update_every=2, seed=7
    )

    history = agent.train(env, episodes=5)

    assert len(history) == 5
    assert all(entry["steps"] > 0 for entry in history)
    assert all(isinstance(entry["won"], bool) for entry in history)
    assert history[-1]["epsilon"] <= history[0]["epsilon"]
    # By the last episode the buffer has exceeded min_replay_size, so training
    # must have actually happened at least once.
    assert any(entry["loss"] is not None for entry in history)


def test_training_compatible_with_multichannel_state_over_many_episodes():
    # Confirms the 11-channel encoding produces no tensor shape errors across
    # a longer run, and that weights actually move as a result.
    env = MinesweeperEnv(rows=4, cols=4, num_mines=2, seed=11)
    agent = DQNAgent(
        rows=4, cols=4, batch_size=16, min_replay_size=16, target_update_every=5, seed=11
    )

    before = [p.clone() for p in agent.online_network.parameters()]
    history = agent.train(env, episodes=150)
    after = list(agent.online_network.parameters())

    assert len(history) == 150
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert any(entry["loss"] is not None for entry in history)


def test_trained_agent_can_act_greedily_after_training():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=8)
    agent = DQNAgent(
        rows=3, cols=3, batch_size=8, min_replay_size=8, target_update_every=2, seed=8
    )
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


def test_train_history_includes_extended_debugging_metrics():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=9)
    agent = DQNAgent(rows=3, cols=3, batch_size=4, min_replay_size=4, target_update_every=2, seed=9)

    history = agent.train(env, episodes=10)

    expected_keys = {
        "total_reward", "steps", "won", "epsilon", "lr", "loss",
        "avg_q", "max_q", "grad_norm", "td_error_mean", "td_error_max",
        "reward_rolling_mean", "win_rate_rolling",
    }
    assert expected_keys.issubset(history[-1].keys())
    assert isinstance(history[-1]["reward_rolling_mean"], float)
    assert 0.0 <= history[-1]["win_rate_rolling"] <= 1.0


# --- Learning rate scheduling ------------------------------------------


def test_resolve_scheduled_lr_picks_correct_milestone():
    schedule = [(0, 1e-4), (20000, 5e-5), (40000, 1e-5)]

    assert resolve_scheduled_lr(schedule, 0) == 1e-4
    assert resolve_scheduled_lr(schedule, 19999) == 1e-4
    assert resolve_scheduled_lr(schedule, 20000) == 5e-5
    assert resolve_scheduled_lr(schedule, 39999) == 5e-5
    assert resolve_scheduled_lr(schedule, 40000) == 1e-5
    assert resolve_scheduled_lr(schedule, 100000) == 1e-5


def test_resolve_scheduled_lr_does_not_require_presorted_input():
    schedule = [(40000, 1e-5), (0, 1e-4), (20000, 5e-5)]
    assert resolve_scheduled_lr(schedule, 25000) == 5e-5


def test_constant_lr_when_no_schedule_given():
    agent = DQNAgent(rows=3, cols=3, lr=2e-4, seed=0)
    assert agent.optimizer.param_groups[0]["lr"] == 2e-4
    assert agent.lr_schedule is None


def test_lr_schedule_updates_optimizer_during_train():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=0)
    agent = DQNAgent(
        rows=3, cols=3, batch_size=4, min_replay_size=4, target_update_every=1,
        lr_schedule=[(0, 1e-3), (3, 1e-4)], seed=0,
    )

    assert agent.optimizer.param_groups[0]["lr"] == 1e-3

    history = agent.train(env, episodes=5)

    assert history[0]["lr"] == 1e-3
    assert history[2]["lr"] == 1e-3  # episode index 2 is still before threshold 3
    assert history[3]["lr"] == 1e-4  # threshold reached
    assert history[4]["lr"] == 1e-4
    assert agent.optimizer.param_groups[0]["lr"] == 1e-4


# --- Checkpointing --------------------------------------------------------


def test_save_checkpoint_writes_state_dict_and_metadata(tmp_path):
    agent = DQNAgent(rows=3, cols=3, seed=0)
    path = tmp_path / "model.pt"

    agent.save_checkpoint(path, episode=10, win_rate=0.4, average_reward=2.5)

    assert path.exists()
    checkpoint = torch.load(path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert checkpoint["metadata"]["episode"] == 10
    assert checkpoint["metadata"]["win_rate"] == 0.4
    assert checkpoint["metadata"]["average_reward"] == 2.5
    assert "timestamp" in checkpoint["metadata"]


def test_load_checkpoint_restores_weights_and_returns_metadata(tmp_path):
    agent = DQNAgent(rows=3, cols=3, seed=0)
    agent.save_checkpoint(tmp_path / "best_model.pt", episode=42, win_rate=0.55, average_reward=3.2)

    # Mutate the in-memory weights so we can detect that loading actually restores them.
    with torch.no_grad():
        for p in agent.online_network.parameters():
            p.add_(1.0)

    metadata = agent.load_checkpoint(tmp_path / "best_model.pt")

    assert metadata["episode"] == 42
    assert metadata["win_rate"] == 0.55
    assert metadata["average_reward"] == 3.2
    assert "timestamp" in metadata

    # Online (and target) networks should now match a freshly constructed
    # agent with the same seed, since that's exactly what was saved.
    reference = DQNAgent(rows=3, cols=3, seed=0)
    for loaded, ref in zip(agent.online_network.parameters(), reference.online_network.parameters()):
        assert torch.equal(loaded, ref)
    for loaded, ref in zip(agent.target_network.parameters(), reference.target_network.parameters()):
        assert torch.equal(loaded, ref)


def test_checkpoint_best_model_only_replaced_on_improvement(tmp_path, monkeypatch):
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=0)
    agent = DQNAgent(
        rows=3, cols=3, batch_size=4, min_replay_size=4, target_update_every=1, seed=0
    )

    # Script a non-monotonic sequence of "evaluation" win rates: 5 periodic
    # checkpoint evals (one per episode, since checkpoint_every=1) plus one
    # final eval at the end of training.
    scripted_win_rates = iter([0.2, 0.5, 0.3, 0.7, 0.1, 0.1])

    def fake_evaluate_greedy(env, episodes):
        return {"win_rate": next(scripted_win_rates), "avg_reward": 1.0}

    save_calls = []
    original_save = agent.save_checkpoint

    def spy_save(path, episode, win_rate, average_reward):
        save_calls.append(win_rate)
        original_save(path, episode, win_rate, average_reward)

    monkeypatch.setattr(agent, "_evaluate_greedy", fake_evaluate_greedy)
    monkeypatch.setattr(agent, "save_checkpoint", spy_save)

    agent.train(env, episodes=5, checkpoint_dir=tmp_path, checkpoint_every=1, checkpoint_eval_episodes=1)

    # Best-model saves happen only on a new running-max win rate (0.2, 0.5,
    # 0.7); the trailing 0.1 is the unconditional final-model save.
    assert save_calls == [0.2, 0.5, 0.7, 0.1]

    best_metadata = agent.load_checkpoint(tmp_path / "best_model.pt")
    assert best_metadata["win_rate"] == 0.7
    final_metadata = agent.load_checkpoint(tmp_path / "final_model.pt")
    assert final_metadata["win_rate"] == 0.1


def test_no_checkpointing_by_default(monkeypatch):
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=0)
    agent = DQNAgent(rows=3, cols=3, batch_size=4, min_replay_size=4, target_update_every=1, seed=0)

    save_calls = []
    monkeypatch.setattr(agent, "save_checkpoint", lambda *a, **k: save_calls.append((a, k)))

    agent.train(env, episodes=3)  # checkpoint_dir defaults to None

    assert save_calls == []


def test_double_dqn_target_never_bootstraps_from_a_revealed_cell():
    """The bootstrap argmax must be restricted to cells still hidden.

    `select_action` can never take a revealed cell, so those actions get no
    gradient and their Q-values drift freely. An unmasked argmax would select
    them and feed that drift back in as a target -- the feedback loop behind
    this project's Q-value divergence.
    """
    agent = DQNAgent(rows=4, cols=4, seed=0)

    # One next-state with a single hidden cell left, at flat index 5.
    board = np.zeros((4, 4), dtype=np.int8)
    board[1, 1] = -1
    encoded = encode_observation(board)[None, ...]

    # Park a huge value on every revealed action; the only hidden one is small.
    with torch.no_grad():
        agent.online_network.head[-1].bias.fill_(1000.0)
        agent.online_network.head[-1].bias[5] = -1.0
        agent.target_network.head[-1].bias.fill_(500.0)
        agent.target_network.head[-1].bias[5] = -7.0

    target = agent._double_dqn_targets(torch.from_numpy(encoded))

    # Must evaluate the hidden cell (target bias -7), not a revealed one (500).
    # Tolerance is loose because the conv/linear weights contribute a little on
    # top of the bias -- the point is which action was selected, not the exact value.
    assert target.item() == pytest.approx(-7.0, abs=0.5)


def test_double_dqn_target_handles_a_next_state_with_no_hidden_cells():
    # A winning move reveals every safe cell; masking everything would make the
    # argmax undefined, so those rows must not produce NaN/-inf.
    agent = DQNAgent(rows=4, cols=4, seed=0)
    board = np.zeros((4, 4), dtype=np.int8)  # nothing hidden
    encoded = encode_observation(board)[None, ...]

    target = agent._double_dqn_targets(torch.from_numpy(encoded))

    assert torch.isfinite(target).all()


def test_train_step_still_runs_end_to_end_with_the_mask():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=0)
    agent = DQNAgent(rows=5, cols=5, batch_size=8, min_replay_size=8, seed=0)
    observation, _ = env.reset()
    for _ in range(40):
        action = agent.select_action(observation, explore=True)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        agent.remember(observation, action, reward, next_observation, terminated or truncated)
        observation = next_observation if not (terminated or truncated) else env.reset()[0]

    metrics = agent.train_step()

    assert metrics is not None
    assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["max_q"])


def test_epsilon_decay_rate_controls_how_long_exploration_lasts():
    """The schedule's reach is the point, so pin it numerically.

    The 0.995 default hits the 0.05 floor at episode 598, which on a
    25,000-episode run means ~98% of training is near-greedy. 0.9998 stretches
    the same descent across ~15,000 episodes. Both are exercised because the
    difference between them is a real experimental variable (see exp_I).
    """
    for decay, expected_floor_episode in ((0.995, 598), (0.9998, 14977)):
        agent = DQNAgent(rows=5, cols=5, epsilon_decay=decay, epsilon_min=0.05, seed=0)
        episodes = 0
        while agent.epsilon > agent.epsilon_min:
            agent.decay_epsilon()
            episodes += 1
            assert episodes <= 20000, "epsilon never reached its floor"
        assert episodes == pytest.approx(expected_floor_episode, rel=0.01)


def test_epsilon_never_decays_below_its_floor():
    agent = DQNAgent(rows=5, cols=5, epsilon_decay=0.5, epsilon_min=0.2, seed=0)
    for _ in range(50):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.2)


def test_conv_channels_length_sets_network_depth():
    # Depth is a real experimental variable (see the "deep" preset), so the
    # mapping from conv_channels length to layer count is pinned.
    for widths, expected_layers in (((16,), 1), ((16, 32), 2), ((16, 32, 32, 32), 4)):
        net = DQNNetwork(rows=5, cols=5, conv_channels=widths)
        conv_layers = [m for m in net.conv if isinstance(m, torch.nn.Conv2d)]
        assert len(conv_layers) == expected_layers
        assert conv_layers[-1].out_channels == widths[-1]
        # The flattened head only depends on the *last* conv width, whatever
        # the depth, so the output stays one Q-value per cell.
        assert net(torch.zeros((2, NUM_CHANNELS, 5, 5))).shape == (2, 25)


def test_deep_preset_adds_depth_not_width():
    default_net = DQNNetwork(rows=5, cols=5, **NETWORK_PRESETS["default"])
    deep_net = DQNNetwork(rows=5, cols=5, **NETWORK_PRESETS["deep"])

    default_convs = [m for m in default_net.conv if isinstance(m, torch.nn.Conv2d)]
    deep_convs = [m for m in deep_net.conv if isinstance(m, torch.nn.Conv2d)]

    assert len(deep_convs) > len(default_convs)
    # The overlapping layers keep the default's widths, so depth is the variable.
    for shallow_layer, deep_layer in zip(default_convs, deep_convs):
        assert shallow_layer.out_channels == deep_layer.out_channels


def test_network_rejects_an_empty_conv_stack():
    with pytest.raises(ValueError):
        DQNNetwork(rows=5, cols=5, conv_channels=())


def test_deep_preset_is_selectable_through_the_agent():
    agent = DQNAgent(rows=5, cols=5, network_size="deep", seed=0)
    conv_layers = [m for m in agent.online_network.conv if isinstance(m, torch.nn.Conv2d)]
    assert len(conv_layers) == 4
    # Target network must be built the same way, or syncing would fail.
    assert len([m for m in agent.target_network.conv if isinstance(m, torch.nn.Conv2d)]) == 4
    agent.update_target_network()


# --- Fully-convolutional head and train_every ---------------------------


def test_linear_head_remains_the_default():
    # Existing checkpoints were all trained with the flatten-and-Linear head,
    # so the default must not move or they stop loading.
    net = DQNNetwork(rows=5, cols=5)
    assert net.head_type == "linear"
    assert NETWORK_PRESETS["default"].get("head_type", "linear") == "linear"


def test_unknown_head_type_rejected():
    with pytest.raises(ValueError, match="Unknown head_type"):
        DQNNetwork(rows=5, cols=5, head_type="bogus")


def test_conv_head_output_matches_linear_head_contract():
    board = np.random.randint(-1, 9, size=(5, 5))
    batch = torch.from_numpy(encode_observation(board))[None]
    for preset in ("default", "fully_conv"):
        out = DQNNetwork(5, 5, **NETWORK_PRESETS[preset])(batch)
        assert out.shape == (1, 25), preset


def test_conv_head_parameter_count_is_independent_of_board_size():
    counts = {
        size: sum(p.numel() for p in DQNNetwork(size, size, **NETWORK_PRESETS["fully_conv"]).parameters())
        for size in (5, 9, 16)
    }
    assert len(set(counts.values())) == 1, counts
    # And it must stay far below the linear head it replaces, which is the
    # point of the preset -- the Linear layer dominates at every board size.
    linear_16 = sum(p.numel() for p in DQNNetwork(16, 16, **NETWORK_PRESETS["default"]).parameters())
    assert counts[16] < linear_16 / 10


@pytest.mark.parametrize("rows,cols", [(5, 5), (9, 9), (16, 16), (9, 16)])
def test_conv_head_network_accepts_boards_it_was_not_built_for(rows, cols):
    # Zero-shot transfer: one set of weights, any board size.
    net = DQNNetwork(6, 6, **NETWORK_PRESETS["fully_conv"])
    board = np.random.randint(-1, 9, size=(rows, cols))
    out = net(torch.from_numpy(encode_observation(board))[None])
    assert out.shape == (1, rows * cols)


def test_agent_with_conv_head_plays_a_larger_board_than_it_was_built_for():
    agent = DQNAgent(rows=6, cols=6, network_size="fully_conv", seed=0)
    env = MinesweeperEnv(rows=9, cols=9, num_mines=10, seed=0)
    observation, _ = env.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 50:
        action = agent.select_action(observation, explore=False)
        assert 0 <= action < 81
        observation, _, terminated, truncated, _ = env.step(action)
        steps += 1
    assert steps > 0


def test_train_every_defaults_to_one_and_rejects_zero():
    assert DQNAgent(rows=5, cols=5, seed=0).train_every == 1
    with pytest.raises(ValueError, match="train_every must be at least 1"):
        DQNAgent(rows=5, cols=5, train_every=0, seed=0)


def test_train_every_reduces_gradient_step_count(monkeypatch):
    calls = {"n": 0}

    def make_agent(train_every):
        agent = DQNAgent(
            rows=5, cols=5, min_replay_size=1, batch_size=4, train_every=train_every, seed=0
        )
        real = agent.train_step

        def counted():
            calls["n"] += 1
            return real()

        agent.train_step = counted
        return agent

    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=0)

    calls["n"] = 0
    make_agent(1).train(env, episodes=30)
    every_one = calls["n"]

    calls["n"] = 0
    make_agent(4).train(env, episodes=30)
    every_four = calls["n"]

    assert every_one > every_four
    # Not exactly 4x, since episode counts vary between the two runs, but the
    # reduction should be substantial rather than incidental.
    assert every_four < every_one / 2


def test_train_every_cycle_spans_episodes():
    # The counter is global, not per-episode: with 4-step episodes and
    # train_every=4, a per-episode counter would fire on every episode's 4th
    # step, which would silently defeat the setting on short episodes.
    agent = DQNAgent(rows=5, cols=5, train_every=4, seed=0)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=0)
    agent.train(env, episodes=5)
    assert agent._steps_since_reset > 0
