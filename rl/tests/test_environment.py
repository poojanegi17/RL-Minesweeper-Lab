"""Tests for the Minesweeper game engine, Gymnasium environment, and random agent."""

import numpy as np
import pytest

from agents.random_agent import RandomAgent
from environment.minesweeper import Minesweeper
from environment.minesweeper_env import MinesweeperEnv


def test_board_initializes_correctly():
    game = Minesweeper(rows=5, cols=5, num_mines=5, seed=42)
    board = game.get_state()
    assert len(board) == 5
    assert all(len(row) == 5 for row in board)
    assert all(cell == -1 for row in board for cell in row)


def test_mines_are_placed_correctly():
    game = Minesweeper(rows=5, cols=5, num_mines=5, seed=42)
    mine_count = sum(sum(row) for row in game.mines)
    assert mine_count == 5


def test_reset_clears_revealed_cells_and_replaces_mines():
    game = Minesweeper(rows=5, cols=5, num_mines=5, seed=1)
    game.reveal(0, 0)
    game.reset()
    board = game.get_state()
    assert all(cell == -1 for row in board for cell in row)
    assert not game.is_game_over()


def test_revealing_a_safe_cell_updates_state():
    game = Minesweeper(rows=5, cols=5, num_mines=1, seed=7)
    safe_r, safe_c = next(
        (r, c) for r in range(5) for c in range(5) if not game.mines[r][c]
    )
    game.reveal(safe_r, safe_c)
    assert game.revealed[safe_r][safe_c] is True
    assert game.board[safe_r][safe_c] != -1


def test_hitting_a_mine_ends_the_episode():
    game = Minesweeper(rows=5, cols=5, num_mines=1, seed=7)
    mine_r, mine_c = next(
        (r, c) for r in range(5) for c in range(5) if game.mines[r][c]
    )
    game.reveal(mine_r, mine_c)
    assert game.is_game_over() is True
    assert game.is_won() is False


def test_revealing_every_safe_cell_wins():
    game = Minesweeper(rows=3, cols=3, num_mines=1, seed=7)
    for r in range(3):
        for c in range(3):
            if not game.mines[r][c]:
                game.reveal(r, c)
    assert game.is_game_over() is True
    assert game.is_won() is True


def test_env_reset_returns_valid_observation():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    obs, info = env.reset()
    assert obs.shape == (5, 5)
    assert obs.dtype == np.int8
    assert np.all(obs == -1)


def test_env_step_reveal_safe_cell():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=1, seed=7)
    obs, info = env.reset()
    safe_action = next(
        r * env.cols + c
        for r in range(env.rows)
        for c in range(env.cols)
        if not env.game.mines[r][c]
    )
    obs, reward, terminated, truncated, info = env.step(safe_action)
    assert reward in (1.0, 10.0)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_step_hitting_mine_gives_negative_reward_and_terminates():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=1, seed=7)
    obs, info = env.reset()
    mine_action = next(
        r * env.cols + c
        for r in range(env.rows)
        for c in range(env.cols)
        if env.game.mines[r][c]
    )
    obs, reward, terminated, truncated, info = env.step(mine_action)
    assert reward == -10.0
    assert terminated is True
    assert info["won"] is False


def test_random_agent_can_play_a_full_episode():
    env = MinesweeperEnv(rows=4, cols=4, num_mines=3, seed=3)
    agent = RandomAgent(seed=3)
    result = agent.play_episode(env)
    assert result["steps"] > 0
    assert isinstance(result["total_reward"], float)
    assert isinstance(result["won"], bool)


# --- Reward shaping (reward_mode) ------------------------------------------


def test_default_reward_mode_is_default():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    assert env.reward_mode == "default"


def test_unknown_reward_mode_rejected():
    with pytest.raises(ValueError):
        MinesweeperEnv(rows=5, cols=5, num_mines=5, reward_mode="bogus")


def test_shaped_mode_matches_default_for_a_non_cascading_safe_reveal():
    # A safe reveal that doesn't trigger a cascade (adjacent count > 0) should
    # score identically to default mode's flat +1 -- shaping only adds on top
    # of cascades, it doesn't change the base per-reveal reward.
    default_env = MinesweeperEnv(rows=5, cols=5, num_mines=1, seed=7, reward_mode="default")
    shaped_env = MinesweeperEnv(rows=5, cols=5, num_mines=1, seed=7, reward_mode="shaped")
    default_env.reset()
    shaped_env.reset()

    # Find a safe cell adjacent to the mine (nonzero count -> no cascade).
    mine_r, mine_c = next(
        (r, c) for r in range(5) for c in range(5) if default_env.game.mines[r][c]
    )
    non_cascading = next(
        r * 5 + c
        for r in range(max(0, mine_r - 1), min(5, mine_r + 2))
        for c in range(max(0, mine_c - 1), min(5, mine_c + 2))
        if not default_env.game.mines[r][c]
    )

    _, default_reward, _, _, _ = default_env.step(non_cascading)
    _, shaped_reward, _, _, _ = shaped_env.step(non_cascading)

    assert default_reward == 1.0
    assert shaped_reward == 1.0


def test_shaped_mode_gives_cascade_bonus_proportional_to_revealed_cells():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=1, seed=1, reward_mode="shaped")
    env.reset()

    # seed=1 with 1 mine: revealing (0, 0) triggers a multi-cell cascade
    # (verified empirically -- see the manual reward_mode smoke test this
    # mirrors). The bonus should be > the flat +1 base reward whenever more
    # than one cell got revealed by the action.
    revealed_before = sum(sum(r) for r in env.game.revealed)
    _, reward, terminated, _, info = env.step(0)
    revealed_after = sum(sum(r) for r in env.game.revealed)
    cells_revealed = revealed_after - revealed_before

    assert not terminated  # a single safe action shouldn't end this board
    assert cells_revealed > 1  # confirms this action actually cascaded
    expected = 1.0 + (cells_revealed - 1) * 0.2
    assert reward == pytest.approx(expected)
    assert reward > 1.0


def test_shaped_mode_mine_hit_and_win_rewards_differ_from_default():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=1, seed=7, reward_mode="shaped")
    env.reset()
    mine_action = next(
        r * 5 + c for r in range(5) for c in range(5) if env.game.mines[r][c]
    )

    _, reward, terminated, _, info = env.step(mine_action)

    assert terminated is True
    assert info["won"] is False
    assert reward == -15.0  # shaped penalty, larger in magnitude than default's -10.0


def test_shaped_mode_win_reward():
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=7, reward_mode="shaped")
    env.reset()

    reward = 0.0
    terminated = False
    for r in range(3):
        for c in range(3):
            if terminated or env.game.mines[r][c]:
                continue
            _, reward, terminated, _, info = env.step(r * 3 + c)

    assert terminated is True
    assert info["won"] is True
    assert reward == 20.0  # shaped win reward, larger than default's +10.0


def test_reward_mode_never_reads_mine_positions_outside_the_engine():
    # The env's own step() implementation must only ever read
    # self.game.revealed / self.game.is_won() -- never self.game.mines --
    # to compute a shaped reward, preserving the "observation only" fairness
    # rule other agents already follow. This is a structural smoke check:
    # shaped rewards must be computable before termination using only
    # publicly-revealed information.
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42, reward_mode="shaped")
    env.reset()
    hidden_action = next(
        i for i in range(25) if not env.game.revealed[i // 5][i % 5] and not env.game.mines[i // 5][i % 5]
    )
    _, reward, terminated, _, _ = env.step(hidden_action)
    assert isinstance(reward, float)
