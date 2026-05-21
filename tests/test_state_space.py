"""Property-based tests for the env (satisfies spec.md §6 bullet 2).

We use hypothesis to drive random action sequences and seeded resets and
assert that the env never crashes, that observations remain in bounds, and
that rewards stay within mathematically expected limits.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from epistemic_arcade.env.tetris import (
    BOARD_COLS,
    BOARD_ROWS,
    LINE_CLEAR_REWARDS,
    NUM_ACTIONS,
    SURVIVAL_BONUS,
    TetrisEnv,
)

MAX_REWARD_PER_STEP = SURVIVAL_BONUS + max(LINE_CLEAR_REWARDS.values())
MIN_REWARD_PER_STEP = 0.0  # base env never produces a negative reward


_ACTION_SEQ = st.lists(
    st.integers(min_value=0, max_value=NUM_ACTIONS - 1),
    min_size=0,
    max_size=300,
)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1), actions=_ACTION_SEQ)
def test_env_never_crashes_under_random_actions(seed: int, actions: list[int]) -> None:
    env = TetrisEnv()
    obs, info = env.reset(seed=seed)
    assert obs.shape == (BOARD_ROWS, BOARD_COLS)
    for action in actions:
        if env.unwrapped._game_over:
            obs, info = env.reset(seed=seed)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (BOARD_ROWS, BOARD_COLS)
        assert obs.dtype == np.int8
        assert np.all(obs >= 0) and np.all(obs <= 1)
        assert MIN_REWARD_PER_STEP <= reward <= MAX_REWARD_PER_STEP
        assert isinstance(terminated, bool)
        assert truncated is False
        assert info["current_piece"] in {"I", "O", "T", "S", "Z", "J", "L"}
        if terminated:
            obs, info = env.reset(seed=seed)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_initial_board_is_active_piece_only(seed: int) -> None:
    """After reset(), the observation has exactly 4 active cells (the spawned piece)."""
    env = TetrisEnv()
    obs, _ = env.reset(seed=seed)
    assert int(obs.sum()) == 4


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_board_occupancy_never_exceeds_capacity(seed: int) -> None:
    """Locked cells in the underlying board never exceed total cell count."""
    env = TetrisEnv()
    env.reset(seed=seed)
    # Run a deterministic series of hard drops to fill the board fast.
    for _ in range(50):
        if env.unwrapped._game_over:
            break
        env.step(4)  # hard drop
    assert int(env.unwrapped._board.sum()) <= BOARD_ROWS * BOARD_COLS
