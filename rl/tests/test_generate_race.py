"""Tests for evaluation.shared_race.simulate_shared_race and evaluation.replay.build_shared_race.

Uses a 2x2/1-mine board throughout: every safe cell is a neighbor of the one
mine cell (all cells are mutual neighbors on a 2x2 grid), so every safe
reveal has adjacent_count >= 1 -- no flood-reveal cascades, which keeps each
scripted agent's turns landing exactly where the test expects.
"""

from environment.minesweeper import Minesweeper
from environment.utils import coords_to_action
from evaluation.replay import build_shared_race
from evaluation.shared_race import simulate_shared_race


def _mine_and_safe_actions(game: Minesweeper):
    mine_action = None
    safe_actions = []
    for r in range(game.rows):
        for c in range(game.cols):
            action = coords_to_action(r, c, game.cols)
            if game.mines[r][c]:
                mine_action = action
            else:
                safe_actions.append(action)
    return mine_action, safe_actions


def _scripted_agent(actions):
    """An action_fn that yields the given flattened actions in order, one per call."""
    it = iter(actions)
    return lambda observation: next(it)


def test_elimination_does_not_stop_the_game_for_survivors():
    game = Minesweeper(rows=2, cols=2, num_mines=1, seed=1)
    mine_action, safe_actions = _mine_and_safe_actions(game)
    assert len(safe_actions) == 3

    agent_a = ("A", _scripted_agent([mine_action]), None)  # steps on the mine immediately
    agent_b = ("B", _scripted_agent(safe_actions), None)  # clears the rest of the board alone

    result = simulate_shared_race(game, [agent_a, agent_b])

    assert result["won"] is True
    assert result["surviving_agents"] == ["B"]
    assert result["eliminated_agents"] == {"A": 1}
    assert result["total_turns"] == 4  # A's 1 fatal turn + B's 3 safe reveals

    assert result["turns"][0]["agent"] == "A"
    assert result["turns"][0]["eliminated"] is True
    # Every turn after A's elimination belongs to B -- A never gets another turn.
    assert all(t["agent"] == "B" for t in result["turns"][1:])


def test_eliminated_agents_fatal_cell_stays_hidden_from_every_later_turn():
    game = Minesweeper(rows=2, cols=2, num_mines=1, seed=1)
    mine_action, safe_actions = _mine_and_safe_actions(game)
    mine_row, mine_col = mine_action // game.cols, mine_action % game.cols

    agent_a = ("A", _scripted_agent([mine_action]), None)
    agent_b = ("B", _scripted_agent(safe_actions), None)

    result = simulate_shared_race(game, [agent_a, agent_b])

    # The mine cell must never show as revealed in any later turn's board --
    # not to B, and not in the final state -- exactly what keeps DQN/PPO's
    # observations in-distribution and keeps the danger genuinely hidden.
    for turn in result["turns"]:
        assert turn["board_state"][mine_row][mine_col] == -1


def test_all_agents_eliminated_is_a_valid_no_winner_outcome():
    # B doesn't know A died on that cell -- it's still hidden -- so scripting
    # B to click the exact same cell demonstrates the danger stayed hidden.
    game = Minesweeper(rows=2, cols=2, num_mines=1, seed=1)
    mine_action, _ = _mine_and_safe_actions(game)

    agent_a = ("A", _scripted_agent([mine_action]), None)
    agent_b = ("B", _scripted_agent([mine_action]), None)

    result = simulate_shared_race(game, [agent_a, agent_b])

    assert result["won"] is False
    assert result["surviving_agents"] == []
    assert result["eliminated_agents"] == {"A": 1, "B": 2}
    assert result["total_turns"] == 2


def test_win_ends_the_round_immediately_with_no_phantom_turns():
    # A single agent that clears the whole board alone -- confirms the loop
    # doesn't run a partner through an extra turn after the board is won.
    game = Minesweeper(rows=2, cols=2, num_mines=1, seed=1)
    _, safe_actions = _mine_and_safe_actions(game)

    agent_a = ("A", _scripted_agent(safe_actions), None)

    result = simulate_shared_race(game, [agent_a])

    assert result["won"] is True
    assert result["total_turns"] == len(safe_actions)


def test_reasoning_fn_is_called_with_observation_and_action():
    game = Minesweeper(rows=2, cols=2, num_mines=1, seed=1)
    mine_action, safe_actions = _mine_and_safe_actions(game)

    calls = []

    def reasoning_fn(observation, action):
        calls.append(action)
        return {"note": "recorded"}

    agent_a = ("A", _scripted_agent(safe_actions), reasoning_fn)
    result = simulate_shared_race(game, [agent_a])

    assert calls == safe_actions
    assert all(t["reasoning"] == {"note": "recorded"} for t in result["turns"])


def test_build_shared_race_assembles_expected_fields():
    result = {
        "initial_board": [[-1, -1], [-1, -1]],
        "turns": [
            {"turn": 1, "agent": "Random", "action": {"row": 0, "col": 0}, "board_state": [[-1, -1], [-1, -1]], "eliminated": True, "reasoning": None},
            {"turn": 2, "agent": "CSP", "action": {"row": 0, "col": 1}, "board_state": [[-1, 1], [-1, -1]], "eliminated": False, "reasoning": {"deduction_type": "probability_guess"}},
        ],
        "won": False,
        "total_turns": 2,
        "surviving_agents": ["CSP"],
        "eliminated_agents": {"Random": 1},
    }

    race = build_shared_race(
        race_number=3,
        seed=42,
        board_size="2x2",
        mines=1,
        generated_at="2026-01-01T00:00:00+00:00",
        turn_order=["Random", "CSP"],
        result=result,
    )

    assert race["id"] == "race_3"
    assert race["seed"] == 42  # kept for reproducibility, decoupled from the id
    assert race["turn_order"] == ["Random", "CSP"]
    assert race["turns"] == result["turns"]
    assert race["surviving_agents"] == ["CSP"]
    assert race["eliminated_agents"] == {"Random": 1}
