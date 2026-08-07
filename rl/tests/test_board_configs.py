"""Tests for board_configs.py."""

import pytest

from board_configs import BOARD_LEVELS, DENSITY_ORDER, LEVEL_ORDER, resolve


def test_every_level_has_every_density():
    for level_name in LEVEL_ORDER:
        assert set(BOARD_LEVELS[level_name].densities.keys()) == set(DENSITY_ORDER)


def test_beginner_standard_matches_existing_benchmark_board():
    # The rest of the project's data (experiments, replays, races generated
    # so far) all assume 5x5/5 mines -- this pins that assumption down so it
    # can never silently drift out of sync with this catalog.
    rows, cols, mines = resolve("beginner", "standard")
    assert (rows, cols, mines) == (5, 5, 5)


def test_mine_counts_are_valid_for_their_board_size():
    for level_name, level in BOARD_LEVELS.items():
        cell_count = level.rows * level.cols
        for density_name, mines in level.densities.items():
            assert 0 < mines < cell_count, f"{level_name}/{density_name}: {mines} mines on {cell_count} cells"


def test_densities_increase_within_a_level():
    for level in BOARD_LEVELS.values():
        assert level.densities["sparse"] < level.densities["standard"] < level.densities["dense"]


def test_resolve_unknown_level_raises_key_error():
    with pytest.raises(KeyError, match="beginner"):
        resolve("nightmare", "standard")


def test_resolve_unknown_density_raises_key_error():
    with pytest.raises(KeyError, match="sparse"):
        resolve("beginner", "nightmare")
