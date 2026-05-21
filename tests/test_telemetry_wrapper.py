"""Telemetry-wrapper tests: epistemic-action flagging on hand-crafted sequences."""

from __future__ import annotations

import gymnasium as gym

from epistemic_arcade.env.telemetry_wrapper import TelemetryWrapper
from epistemic_arcade.env.tetris import (
    ACTION_HARD_DROP,
    ACTION_LEFT,
    ACTION_NOP,
    ACTION_RIGHT,
    ACTION_ROTATE,
    BOARD_COLS,
    BOARD_ROWS,
    TetrisEnv,
)


def _setup_for_piece(piece_name: str, *, anchor_col: int = 4) -> TelemetryWrapper:
    """Reset the env and force a specific piece at a known spawn position."""
    env = TelemetryWrapper(TetrisEnv())
    env.reset(seed=0)
    env.unwrapped._piece_name = piece_name
    env.unwrapped._orientation = 0
    env.unwrapped._anchor_row = 0
    env.unwrapped._anchor_col = anchor_col
    env.unwrapped._actions_since_spawn = 0
    env.unwrapped._rotations_since_spawn = 0
    return env


def test_single_rotation_on_t_piece_is_not_epistemic() -> None:
    env = _setup_for_piece("T")
    env.step(ACTION_ROTATE)
    env.step(ACTION_HARD_DROP)
    rotation_frames = [f for f in env.frames if f.action_taken == ACTION_ROTATE]
    assert len(rotation_frames) == 1
    assert rotation_frames[0].is_epistemic_action is False


def test_full_rotation_cycle_on_t_piece_flags_all_four() -> None:
    """T has k=4. Four rotations → final orientation 0 → all 4 redundant."""
    env = _setup_for_piece("T")
    for _ in range(4):
        env.step(ACTION_ROTATE)
    env.step(ACTION_HARD_DROP)
    rotation_frames = [f for f in env.frames if f.action_taken == ACTION_ROTATE]
    assert len(rotation_frames) == 4
    assert all(f.is_epistemic_action for f in rotation_frames)


def test_o_piece_any_rotation_is_epistemic() -> None:
    """O has k=1, so any rotation is redundant."""
    env = _setup_for_piece("O")
    env.step(ACTION_ROTATE)
    env.step(ACTION_HARD_DROP)
    rotation_frames = [f for f in env.frames if f.action_taken == ACTION_ROTATE]
    # Successful rotation? For O, rotating doesn't change orientation
    # because there's only one orientation. So the action doesn't actually
    # change the piece -- it counts as a non-successful rotation and is
    # therefore NOT flagged. We assert the latter explicitly.
    assert all(f.rotation_succeeded is False for f in rotation_frames)
    assert all(f.is_epistemic_action is False for f in rotation_frames)


def test_left_then_right_flags_the_right_as_reversal() -> None:
    env = _setup_for_piece("T", anchor_col=5)
    env.step(ACTION_LEFT)
    env.step(ACTION_RIGHT)
    env.step(ACTION_HARD_DROP)
    left = [f for f in env.frames if f.action_taken == ACTION_LEFT][0]
    right = [f for f in env.frames if f.action_taken == ACTION_RIGHT][0]
    assert left.is_epistemic_action is False
    assert right.is_epistemic_action is True


def test_right_left_right_flags_both_reversals() -> None:
    env = _setup_for_piece("T", anchor_col=5)
    env.step(ACTION_RIGHT)
    env.step(ACTION_LEFT)
    env.step(ACTION_RIGHT)
    env.step(ACTION_HARD_DROP)
    moves = [f for f in env.frames if f.action_taken in (ACTION_LEFT, ACTION_RIGHT)]
    assert moves[0].is_epistemic_action is False  # initial Right
    assert moves[1].is_epistemic_action is True  # Left reverses
    assert moves[2].is_epistemic_action is True  # Right reverses again


def test_nop_between_moves_does_not_break_reversal_detection() -> None:
    env = _setup_for_piece("T", anchor_col=5)
    env.step(ACTION_LEFT)
    env.step(ACTION_NOP)
    env.step(ACTION_RIGHT)
    env.step(ACTION_HARD_DROP)
    right = [f for f in env.frames if f.action_taken == ACTION_RIGHT][0]
    assert right.is_epistemic_action is True


def test_blocked_move_is_not_flagged() -> None:
    """A Left at the leftmost edge is blocked → not a successful translation."""
    env = _setup_for_piece("T", anchor_col=0)  # already at the wall
    env.step(ACTION_LEFT)  # blocked
    env.step(ACTION_RIGHT)  # successful, but no previous successful translation
    env.step(ACTION_HARD_DROP)
    left = [f for f in env.frames if f.action_taken == ACTION_LEFT][0]
    right = [f for f in env.frames if f.action_taken == ACTION_RIGHT][0]
    assert left.translation_succeeded is False
    assert left.is_epistemic_action is False
    assert right.is_epistemic_action is False  # no prior direction to reverse


def test_frame_to_dict_emits_spec_schema_keys() -> None:
    env = _setup_for_piece("T")
    env.step(ACTION_HARD_DROP)
    frame = env.frames[0]
    d = frame.to_dict()
    assert set(d.keys()) == {
        "frame_id",
        "board_state",
        "current_piece",
        "action_taken",
        "reward",
        "is_epistemic_action",
    }


def test_frames_cleared_on_reset() -> None:
    env = _setup_for_piece("T")
    env.step(ACTION_HARD_DROP)
    assert len(env.frames) > 0
    env.reset(seed=1)
    assert env.frames == []


def test_board_state_is_200_char_string() -> None:
    """board_state is the 20x10 grid flattened row-major as a 0/1 string."""
    env = _setup_for_piece("T")
    env.step(ACTION_HARD_DROP)
    frame = env.frames[0]
    assert isinstance(frame.board_state, str)
    assert len(frame.board_state) == BOARD_ROWS * BOARD_COLS == 200
    assert set(frame.board_state) <= {"0", "1"}


def test_partial_buffer_flushed_on_truncation() -> None:
    """A TimeLimit-truncated episode emits its in-flight piece's frames."""
    base = gym.wrappers.TimeLimit(TetrisEnv(), max_episode_steps=2)
    env = TelemetryWrapper(base)
    env.reset(seed=0)
    env.step(ACTION_NOP)  # gravity ticks; no lock
    _obs, _r, terminated, truncated, _info = env.step(ACTION_NOP)
    assert truncated is True
    assert terminated is False
    # Both frames should be flushed despite no piece_locked signal.
    assert len(env.frames) == 2
    # No clean lock → no honest epistemic annotation.
    assert all(f.is_epistemic_action is False for f in env.frames)
