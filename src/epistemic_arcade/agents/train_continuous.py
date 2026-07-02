r"""Continue PPO training from a saved checkpoint, with auto-stop.

Loads a starting checkpoint, then loops::

    train one chunk → save → quick-evaluate → check stopping criteria → repeat.

Three exit conditions:

1. **Mastery** -- average lines-cleared per evaluation game crosses
   ``--mastery-lines`` (default 30).
2. **Plateau** -- the trailing window of ``--plateau-window`` chunks (default
   3) shows no improvement greater than ``--plateau-tolerance`` (default 0.05
   = 5 %).
3. **Cap** -- the last entry in ``--milestones`` is reached.

The script writes ``models/<agent>/ckpt_<milestone>.zip`` for every milestone
it actually trains to, so the subsequent ``evaluate.py`` run can pick them up.
Run with::

    uv run python -m epistemic_arcade.agents.train_continuous \\
      --agent alpha \\
      --resume-from models/alpha/ckpt_1000000.zip \\
      --save-dir models/alpha
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from epistemic_arcade.agents.evaluate import evaluate_checkpoint
from epistemic_arcade.agents.train import make_env
from epistemic_arcade.env.reward_wrapper import AgentType

DEFAULT_MILESTONES: Final[tuple[int, ...]] = tuple(range(2_000_000, 20_000_001, 1_000_000))
DEFAULT_EVAL_GAMES: Final[int] = 30
DEFAULT_MASTERY_LINES: Final[float] = 30.0
DEFAULT_PLATEAU_WINDOW: Final[int] = 3
DEFAULT_PLATEAU_TOLERANCE: Final[float] = 0.05
DEFAULT_PLATEAU_FLOOR: Final[float] = 1.0  # lines/game — see plateaued()
DEFAULT_N_ENVS: Final[int] = 8
DEFAULT_SEED: Final[int] = 42

# Offset eval-game seeds well away from the training base seed so the agent
# isn't graded on piece sequences it trained on.
EVAL_SEED_OFFSET: Final[int] = 1_000_000

BOARD_ROWS: Final[int] = 20
BOARD_COLS: Final[int] = 10


@dataclass
class ChunkSummary:
    """One eval round's verdict.

    Attributes:
        milestone: Cumulative timesteps at which this eval was run.
        avg_lines: Mean lines cleared per game in the eval set.
        avg_survival_frames: Mean episode length in the eval set.
        avg_final_stack_height: Mean stack height at game end (out of 20).
    """

    milestone: int
    avg_lines: float
    avg_survival_frames: float
    avg_final_stack_height: float


def _stack_height(board_str: str) -> int:
    """Height of the locked stack on a 20x10 board, given the 200-char string.

    Examples:
        >>> _stack_height("0" * 200)
        0
        >>> _stack_height("0" * 190 + "1" * 10)
        1
        >>> _stack_height("1" * 10 + "0" * 190)
        20
    """
    for r in range(BOARD_ROWS):
        row = board_str[r * BOARD_COLS : (r + 1) * BOARD_COLS]
        if "1" in row:
            return BOARD_ROWS - r
    return 0


def quick_eval(
    model_path: Path, agent_type: AgentType, num_games: int, seed: int
) -> ChunkSummary:
    """Run ``num_games`` deterministic games and summarise play quality.

    Returns a :class:`ChunkSummary` with ``milestone=0`` -- the caller fills
    it in. Reuses :func:`evaluate.evaluate_checkpoint` so the eval logic stays
    in one place.
    """
    metrics, results = evaluate_checkpoint(model_path, agent_type, num_games, seed)
    heights: list[int] = []
    for g in results:
        if g.frames:
            heights.append(_stack_height(g.frames[-1].board_state))
        else:
            heights.append(0)
    avg_height = sum(heights) / len(heights) if heights else 0.0
    return ChunkSummary(
        milestone=0,
        avg_lines=metrics.avg_lines,
        avg_survival_frames=metrics.avg_survival_frames,
        avg_final_stack_height=avg_height,
    )


def plateaued(
    history: Sequence[ChunkSummary],
    window: int,
    tolerance: float,
    abs_floor: float = DEFAULT_PLATEAU_FLOOR,
) -> bool:
    """True iff the trailing ``window`` chunks have stopped improving.

    Two ways a plateau can fire:

    * **Noise-floor short-circuit.** If even the *best* chunk in the window
      can't clear ``abs_floor`` lines/game, we're stuck and the ratio test
      below is meaningless (tiny absolute swings in a near-zero metric look
      huge in relative terms — 0.03 vs 0.13 is a 333 % spread but both are
      noise). Declared plateau.
    * **Ratio corridor.** Otherwise, plateau iff ``max/min - 1 < tolerance``.

    Examples:
        >>> from epistemic_arcade.agents.train_continuous import ChunkSummary
        >>> # Meaningful values in a tight corridor → plateau.
        >>> h = [
        ...     ChunkSummary(1_000_000, 5.0, 0, 0),
        ...     ChunkSummary(2_000_000, 5.1, 0, 0),
        ...     ChunkSummary(3_000_000, 5.05, 0, 0),
        ... ]
        >>> plateaued(h, window=3, tolerance=0.05)
        True
        >>> # Real jump → not a plateau.
        >>> h.append(ChunkSummary(4_000_000, 6.0, 0, 0))
        >>> plateaued(h, window=3, tolerance=0.05)
        False
        >>> # Stuck at near-zero → plateau even though the ratio swings wildly.
        >>> stuck = [
        ...     ChunkSummary(2_000_000, 0.03, 0, 0),
        ...     ChunkSummary(3_000_000, 0.13, 0, 0),
        ...     ChunkSummary(4_000_000, 0.07, 0, 0),
        ... ]
        >>> plateaued(stuck, window=3, tolerance=0.05)
        True
        >>> # ...but a floor below the best chunk un-plateaus it.
        >>> plateaued(stuck, window=3, tolerance=0.05, abs_floor=0.01)
        False
    """
    if len(history) < window:
        return False
    recent = [h.avg_lines for h in history[-window:]]
    hi = max(recent)
    if hi < abs_floor:
        return True
    lo = min(recent)
    if lo <= 0:
        return False
    return (hi / lo - 1.0) < tolerance


def train_continuous(
    *,
    agent_type: AgentType,
    resume_from: Path,
    save_dir: Path,
    milestones: tuple[int, ...],
    eval_games: int,
    mastery_lines: float,
    plateau_window: int,
    plateau_tolerance: float,
    plateau_floor: float,
    n_envs: int,
    seed: int,
) -> None:
    """Drive the resume-train-eval-decide loop until a stop condition fires.

    Args:
        agent_type: ``"alpha"`` or ``"beta"``.
        resume_from: Path to the saved PPO checkpoint to load.
        save_dir: Per-agent directory under which ``ckpt_<milestone>.zip``
            files are written.
        milestones: Cumulative timestep targets. Sorted ascending; targets at
            or below the resumed model's existing ``num_timesteps`` are
            skipped without training.
        eval_games: Number of games to play for the in-loop quick eval.
        mastery_lines: Per-game line-clear average that triggers an early
            "mastery" exit.
        plateau_window: Number of recent chunks to compare for plateau
            detection.
        plateau_tolerance: Max relative spread (``hi / lo - 1``) inside the
            window before we declare a plateau.
        plateau_floor: Absolute lines/game floor; if the best chunk in the
            window is below this, declare a plateau regardless of ratio.
        n_envs: Number of parallel envs in the ``SubprocVecEnv``.
        seed: Base seed for the training envs (eval seeds are offset).
    """
    torch.set_num_threads(1)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(agent_type, rank=i, base_seed=seed) for i in range(n_envs)])
    model = PPO.load(str(resume_from), env=vec_env, device="cpu")
    model.verbose = 0  # silence per-rollout PPO chatter; we log our own summary lines
    start_ts = int(model.num_timesteps)
    print(f"[resume] {agent_type}: loaded {resume_from} at {start_ts:_d} timesteps", flush=True)

    history: list[ChunkSummary] = []
    sorted_milestones = sorted(milestones)
    stopped_early = False
    for milestone in sorted_milestones:
        if milestone <= int(model.num_timesteps):
            continue
        chunk = milestone - int(model.num_timesteps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        ckpt_path = save_dir / f"ckpt_{milestone}.zip"
        model.save(str(ckpt_path))

        summary = quick_eval(
            ckpt_path, agent_type, eval_games, seed=seed + EVAL_SEED_OFFSET
        )
        summary.milestone = milestone
        history.append(summary)

        delta = ""
        if len(history) >= 2 and history[-2].avg_lines > 0:
            pct = (summary.avg_lines / history[-2].avg_lines - 1.0) * 100
            delta = f" | Δlines {pct:+.1f}%"
        print(
            f"[{agent_type}] {milestone:>10_d} ts | "
            f"lines/game {summary.avg_lines:6.2f} | "
            f"survival {summary.avg_survival_frames:6.1f} | "
            f"end_height {summary.avg_final_stack_height:5.1f}{delta}",
            flush=True,
        )

        if summary.avg_lines >= mastery_lines:
            print(
                f"[stop] {agent_type}: mastery reached at {milestone:_d} "
                f"(avg_lines={summary.avg_lines:.2f} >= {mastery_lines:.1f})",
                flush=True,
            )
            stopped_early = True
            break
        if plateaued(history, plateau_window, plateau_tolerance, plateau_floor):
            recent = [f"{h.avg_lines:.2f}" for h in history[-plateau_window:]]
            print(
                f"[stop] {agent_type}: plateau detected at {milestone:_d} "
                f"(last {plateau_window} lines/game: {recent}, floor={plateau_floor})",
                flush=True,
            )
            stopped_early = True
            break

    if not stopped_early:
        print(
            f"[stop] {agent_type}: hit milestone cap at {sorted_milestones[-1]:_d}",
            flush=True,
        )
    print(f"[done] {agent_type}: final timesteps {int(model.num_timesteps):_d}", flush=True)
    vec_env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue PPO training with auto-stop.")
    parser.add_argument("--agent", choices=["alpha", "beta"], required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="Path to the saved PPO checkpoint to continue from.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory under which per-milestone checkpoints are written.",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=list(DEFAULT_MILESTONES),
        help="Cumulative timestep targets to train to and evaluate at.",
    )
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES)
    parser.add_argument("--mastery-lines", type=float, default=DEFAULT_MASTERY_LINES)
    parser.add_argument("--plateau-window", type=int, default=DEFAULT_PLATEAU_WINDOW)
    parser.add_argument("--plateau-tolerance", type=float, default=DEFAULT_PLATEAU_TOLERANCE)
    parser.add_argument(
        "--plateau-floor",
        type=float,
        default=DEFAULT_PLATEAU_FLOOR,
        help=(
            "Absolute lines/game floor; if no chunk in the window clears this many "
            "lines on average, stop immediately."
        ),
    )
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    train_continuous(
        agent_type=args.agent,
        resume_from=args.resume_from,
        save_dir=args.save_dir,
        milestones=tuple(args.milestones),
        eval_games=args.eval_games,
        mastery_lines=args.mastery_lines,
        plateau_window=args.plateau_window,
        plateau_tolerance=args.plateau_tolerance,
        plateau_floor=args.plateau_floor,
        n_envs=args.n_envs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
