"""Tests for first-click safety, no-guess board generation, and reward scaling."""

import pytest

from agents.csp_solver import CSPAgent
from environment.minesweeper import BoardGenerationError, Minesweeper
from environment.minesweeper_env import MinesweeperEnv
from environment.solvability import deduce_safe_cells, is_solvable_without_guessing


# --- Defaults stay exactly as they were ------------------------------------


def test_default_policy_places_mines_before_any_click():
    # Every published benchmark in this project was measured this way, so the
    # default must keep placing mines at reset() rather than deferring.
    game = Minesweeper(rows=5, cols=5, num_mines=5, seed=42)
    assert game.first_click_safe == "none"
    assert sum(sum(row) for row in game.mines) == 5
    assert game.generation_attempts == 1


def test_default_policy_rng_draw_is_unchanged():
    # Guards reproducibility of existing seeds: the default path must consume
    # the RNG in the same order it always has.
    game = Minesweeper(rows=5, cols=5, num_mines=5, seed=42)
    reference = Minesweeper(rows=5, cols=5, num_mines=5, seed=42)
    assert game.mines == reference.mines


def test_deferred_policy_leaves_board_empty_until_first_click():
    game = Minesweeper(rows=5, cols=5, num_mines=5, seed=1, first_click_safe="cell")
    assert sum(sum(row) for row in game.mines) == 0
    assert game.generation_attempts == 0

    game.reveal(2, 2)
    assert sum(sum(row) for row in game.mines) == 5
    assert game.generation_attempts == 1


# --- First-click safety -----------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_first_click_cell_policy_never_hits_a_mine(seed):
    game = Minesweeper(rows=5, cols=5, num_mines=8, seed=seed, first_click_safe="cell")
    game.reveal(seed % 5, (seed * 3) % 5)
    assert game.is_game_over() is False or game.is_won() is True


@pytest.mark.parametrize("seed", range(25))
def test_first_click_area_policy_clears_the_surrounding_block(seed):
    game = Minesweeper(rows=6, cols=6, num_mines=6, seed=seed, first_click_safe="area")
    row, col = seed % 6, (seed * 5) % 6
    game.reveal(row, col)

    for nr, nc in [(row, col)] + game._neighbors(row, col):
        assert game.mines[nr][nc] is False
    # A mine-free 3x3 block means the clicked cell has adjacent count 0, so the
    # opening move must cascade rather than reveal a lone number.
    assert game.adjacent_counts[row][col] == 0
    assert sum(sum(r) for r in game.revealed) > 1


def test_area_policy_rejects_boards_where_the_mines_cannot_fit():
    game = Minesweeper(rows=5, cols=5, num_mines=17, seed=0, first_click_safe="area")
    with pytest.raises(ValueError, match="cannot place"):
        game.reveal(2, 2)  # centre click protects 9 cells, leaving only 16


def test_same_seed_and_first_click_reproduce_the_same_board():
    a = Minesweeper(rows=6, cols=6, num_mines=6, seed=7, first_click_safe="cell")
    b = Minesweeper(rows=6, cols=6, num_mines=6, seed=7, first_click_safe="cell")
    a.reveal(1, 1)
    b.reveal(1, 1)
    assert a.mines == b.mines


# --- No-guess generation ----------------------------------------------------


def test_guarantee_solvable_requires_a_first_click_policy():
    with pytest.raises(ValueError, match="requires a first_click_safe policy"):
        Minesweeper(rows=5, cols=5, num_mines=5, guarantee_solvable=True)


@pytest.mark.parametrize("seed", range(10))
def test_generated_no_guess_boards_are_solvable_by_the_checker(seed):
    game = Minesweeper(
        rows=6, cols=6, num_mines=6, seed=seed, first_click_safe="area", guarantee_solvable=True
    )
    game.reveal(2, 2)
    assert is_solvable_without_guessing(game.mines, game.adjacent_counts, 6, 6, (2, 2))


def test_csp_agent_clears_every_no_guess_board():
    # The load-bearing test. `solvability.py` reimplements csp_solver.py's rule
    # set, so a board it certifies as needing no guess must be one CSPAgent can
    # actually clear -- if the two ever drift apart, this fails. It is also what
    # makes CSP a 100% reference ceiling on no-guess board sets.
    episodes = 40
    env = MinesweeperEnv(
        rows=6, cols=6, num_mines=6, first_click_safe="area", guarantee_solvable=True
    )
    agent = CSPAgent(rows=6, cols=6, num_mines=6, seed=0)

    wins = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        agent.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.choose_action(observation.tolist())
            observation, _, terminated, truncated, info = env.step(action)
        wins += bool(info["won"])

    assert wins == episodes, f"CSP failed {episodes - wins}/{episodes} certified no-guess boards"


def test_generation_gives_up_loudly_on_impossible_densities():
    game = Minesweeper(
        rows=5,
        cols=5,
        num_mines=20,
        seed=0,
        first_click_safe="cell",
        guarantee_solvable=True,
        max_generation_attempts=50,
    )
    with pytest.raises(BoardGenerationError, match="no no-guess"):
        game.reveal(2, 2)


def test_generation_attempts_are_reported_through_info():
    env = MinesweeperEnv(
        rows=6, cols=6, num_mines=6, first_click_safe="area", guarantee_solvable=True
    )
    env.reset(seed=3)
    _, _, _, _, info = env.step(14)  # (2, 2) on a 6-wide board
    assert info["generation_attempts"] >= 1


# --- The deduction primitive ------------------------------------------------


def test_deduce_safe_cells_applies_the_zero_rule():
    #  0 1 ?     a revealed 0 proves all of its hidden neighbours safe
    board = [
        [0, 1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    safe = deduce_safe_cells(board, 3, 3)
    assert (1, 0) in safe and (1, 1) in safe


def test_deduce_safe_cells_applies_the_subset_rule():
    # Row 0: "1" at (0,0) constrains {(1,0),(1,1)}; "1" at (0,1) constrains
    # {(1,0),(1,1),(1,2)}. The subset leftover {(1,2)} holds 1-1=0 mines.
    board = [
        [1, 1, 0],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    assert (1, 2) in deduce_safe_cells(board, 3, 3)


def test_deduce_safe_cells_returns_nothing_on_a_blank_board():
    board = [[-1] * 4 for _ in range(4)]
    assert deduce_safe_cells(board, 4, 4) == set()


# --- Reward scaling ---------------------------------------------------------


def test_reward_scale_multiplies_every_reward():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=1, seed=7, reward_scale=0.1)
    env.reset()
    mine_action = next(
        r * 5 + c for r in range(5) for c in range(5) if env.game.mines[r][c]
    )
    _, reward, terminated, _, _ = env.step(mine_action)
    assert terminated is True
    assert reward == pytest.approx(-1.0)


def test_reward_scale_defaults_to_one():
    env = MinesweeperEnv(rows=5, cols=5, num_mines=5, seed=42)
    assert env.reward_scale == 1.0


def test_non_positive_reward_scale_rejected():
    with pytest.raises(ValueError, match="reward_scale must be positive"):
        MinesweeperEnv(rows=5, cols=5, num_mines=5, reward_scale=0.0)
