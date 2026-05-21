"""Behavioural tests for the custom 10x20 Tetris environment."""

from __future__ import annotations

import numpy as np
import pytest

from epistemic_arcade.env.tetris import (
    ACTION_HARD_DROP,
    ACTION_LEFT,
    ACTION_NOP,
    ACTION_ROTATE,
    BOARD_COLS,
    BOARD_ROWS,
    NUM_ACTIONS,
    TetrisEnv,
)


def test_observation_shape_and_dtype() -> None:
    env = TetrisEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (BOARD_ROWS, BOARD_COLS)
    assert obs.dtype == np.int8
    # Some cells should be occupied (the spawned piece)
    assert obs.sum() == 4  # all tetrominoes are 4 cells


def test_action_space_is_discrete_5() -> None:
    env = TetrisEnv()
    assert env.action_space.n == NUM_ACTIONS == 5


def test_seeded_reset_is_deterministic() -> None:
    env_a = TetrisEnv()
    env_b = TetrisEnv()
    obs_a, info_a = env_a.reset(seed=123)
    obs_b, info_b = env_b.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)
    assert info_a["current_piece"] == info_b["current_piece"]
    # And subsequent piece sequences should match
    for _ in range(50):
        obs_a, _, term_a, _, _ = env_a.step(ACTION_NOP)
        obs_b, _, term_b, _, _ = env_b.step(ACTION_NOP)
        assert np.array_equal(obs_a, obs_b)
        assert term_a == term_b
        if term_a:
            break


def test_left_then_right_moves_one_column() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    start_col = env.unwrapped._anchor_col
    env.step(ACTION_LEFT)
    assert env.unwrapped._anchor_col == start_col - 1


def test_rotate_changes_orientation_for_t_piece() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    env.unwrapped._piece_name = "T"
    env.unwrapped._orientation = 0
    env.unwrapped._anchor_col = 4
    env.unwrapped._anchor_row = 0
    _, _, _, _, info = env.step(ACTION_ROTATE)
    # T has 4 orientations; rotate CW once → orientation 1
    assert info["piece_orientation"] == 1
    assert info["action_changed_orientation"] is True
    assert info["rotations_since_spawn"] == 1


def test_hard_drop_locks_piece_and_signals_lock() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(ACTION_HARD_DROP)
    assert info["piece_locked"] is True
    # After lock + spawn, the board has the locked piece somewhere
    assert env.unwrapped._board.sum() == 4


def test_line_clear_reward() -> None:
    """Constructing a near-full bottom row and dropping an I-piece clears a line."""
    env = TetrisEnv()
    env.reset(seed=0)
    # Hand-stack the bottom row with 9 of 10 cells filled
    board = env.unwrapped._board
    board[BOARD_ROWS - 1, : BOARD_COLS - 4] = 1  # cols 0-5 filled
    # Force a horizontal I-piece into the remaining gap on the right
    env.unwrapped._piece_name = "I"
    env.unwrapped._orientation = 0
    env.unwrapped._anchor_row = 0
    env.unwrapped._anchor_col = BOARD_COLS - 4  # cols 6-9
    _, reward, _, _, info = env.step(ACTION_HARD_DROP)
    assert info["lines_cleared_this_step"] == 1
    assert info["piece_locked"] is True
    # Reward = survival (0.01) + single-line (1.0) minus nothing (base env, no shaping)
    assert reward == pytest.approx(1.01)


def test_game_over_on_spawn_collision() -> None:
    """Blocking the spawn region in row 0 (without making a full row) terminates the next spawn."""
    env = TetrisEnv()
    env.reset(seed=0)
    # All seven tetrominoes occupy at least one cell in columns 3-6 at spawn.
    # Filling just those cells leaves no full row to clear yet guarantees a
    # collision on the next piece's spawn.
    env.unwrapped._board[0, 3:7] = 1
    _, _, terminated, _, _ = env.step(ACTION_HARD_DROP)
    assert terminated is True


def test_step_after_termination_raises() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    env.unwrapped._board[0, 3:7] = 1
    env.step(ACTION_HARD_DROP)  # triggers termination
    with pytest.raises(RuntimeError):
        env.step(ACTION_NOP)


def test_invalid_action_raises() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(5)


def test_gravity_drops_piece_one_row_per_step() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    start_row = env.unwrapped._anchor_row
    env.step(ACTION_NOP)
    assert env.unwrapped._anchor_row == start_row + 1
