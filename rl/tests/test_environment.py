"""Tests for the Minesweeper game engine, Gymnasium environment, and random agent."""

import numpy as np

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
