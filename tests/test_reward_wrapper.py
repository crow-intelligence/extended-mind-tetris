"""Reward-shaping wrapper tests (satisfies spec.md §6 bullet 1)."""

from __future__ import annotations

import pytest

from epistemic_arcade.env.reward_wrapper import (
    KEYSTROKE_PENALTY,
    EpistemicRewardWrapper,
)
from epistemic_arcade.env.tetris import (
    ACTION_HARD_DROP,
    ACTION_LEFT,
    ACTION_NOP,
    ACTION_RIGHT,
    ACTION_ROTATE,
    SURVIVAL_BONUS,
    TetrisEnv,
)


@pytest.mark.parametrize("action", [ACTION_LEFT, ACTION_RIGHT, ACTION_ROTATE, ACTION_HARD_DROP])
def test_alpha_penalises_every_non_nop_action(action: int) -> None:
    env = EpistemicRewardWrapper(TetrisEnv(), agent_type="alpha")
    env.reset(seed=0)
    _, reward, _, _, info = env.step(action)
    line_bonus = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}[info["lines_cleared_this_step"]]
    expected = SURVIVAL_BONUS + line_bonus - KEYSTROKE_PENALTY
    assert reward == pytest.approx(expected)


def test_alpha_does_not_penalise_nop() -> None:
    env = EpistemicRewardWrapper(TetrisEnv(), agent_type="alpha")
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(ACTION_NOP)
    assert reward == pytest.approx(SURVIVAL_BONUS)


@pytest.mark.parametrize(
    "action",
    [ACTION_NOP, ACTION_LEFT, ACTION_RIGHT, ACTION_ROTATE, ACTION_HARD_DROP],
)
def test_beta_is_reward_passthrough(action: int) -> None:
    alpha = EpistemicRewardWrapper(TetrisEnv(), agent_type="alpha")
    beta = EpistemicRewardWrapper(TetrisEnv(), agent_type="beta")
    alpha.reset(seed=0)
    beta.reset(seed=0)
    _, _, _, _, _ = alpha.step(action)
    _, beta_reward, _, _, info = beta.step(action)
    line_bonus = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}[info["lines_cleared_this_step"]]
    expected = SURVIVAL_BONUS + line_bonus
    assert beta_reward == pytest.approx(expected)


def test_invalid_agent_type_rejected() -> None:
    with pytest.raises(ValueError):
        EpistemicRewardWrapper(TetrisEnv(), agent_type="gamma")  # type: ignore[arg-type]
