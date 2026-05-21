"""Evaluation pipeline: run 100 games per checkpoint and export JSON telemetry.

For each ``(agent, checkpoint)`` pair, loads the PPO model, plays
``--num-games`` games with the deterministic policy on a freshly-seeded
``TelemetryWrapper(EpistemicRewardWrapper(TetrisEnv))``, then:

* Aggregates avg lines, avg keystrokes per piece, and avg survival frames.
* Selects the top-N games by cumulative (shaped) reward as sample games.

The reward observed in the JSON matches the spec §5.2 example: it includes
the Alpha keystroke penalty (so for Alpha, cumulative reward = lines minus
keystroke cost, which naturally surfaces minimal-keystroke play; see the
``perfect-sample-selection`` memory).

Run with::

    uv run python -m epistemic_arcade.agents.evaluate --agent alpha
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import gymnasium as gym
from stable_baselines3 import PPO

from epistemic_arcade.env.reward_wrapper import AgentType, EpistemicRewardWrapper
from epistemic_arcade.env.telemetry_wrapper import FrameRecord, TelemetryWrapper
from epistemic_arcade.env.tetris import ACTION_NOP, TetrisEnv
from epistemic_arcade.io.telemetry import (
    AgentTelemetry,
    AggregateMetrics,
    SampleGame,
    SampleGameMetadata,
    write_telemetry_json,
)

DEFAULT_CHECKPOINTS: Final[tuple[int, ...]] = (10_000, 250_000, 1_000_000)
DEFAULT_NUM_GAMES: Final[int] = 100
DEFAULT_NUM_SAMPLES: Final[int] = 5
DEFAULT_MODELS_ROOT: Final[Path] = Path("models")
DEFAULT_DATA_ROOT: Final[Path] = Path("data")

EVAL_MAX_EPISODE_FRAMES: Final[int] = 3_000
EXPORT_FRAMES_PER_GAME_CAP: Final[int] = 1_000


@dataclass
class GameResult:
    """Per-game accounting from one evaluation rollout."""

    total_reward: float
    total_lines: int
    total_keystrokes: int
    pieces_dropped: int
    survival_frames: int
    frames: list[FrameRecord] = field(repr=False)

    @property
    def keystrokes_per_piece(self) -> float:
        """Mean non-NOP actions per dropped piece in this game."""
        return self.total_keystrokes / self.pieces_dropped if self.pieces_dropped else 0.0


def play_one_game(model: PPO, agent_type: AgentType, seed: int) -> GameResult:
    """Run a single deterministic-policy episode and return its frames + counters.

    Args:
        model: Loaded PPO model.
        agent_type: Which shaping to apply during evaluation.
        seed: Per-game seed for the env (piece sequence determinism).

    Returns:
        A :class:`GameResult` with per-frame telemetry and aggregate counters.
    """
    # TimeLimit innermost so its `truncated` flag propagates outward through
    # both wrappers, letting TelemetryWrapper flush its partial buffer.
    base = gym.wrappers.TimeLimit(TetrisEnv(), max_episode_steps=EVAL_MAX_EPISODE_FRAMES)
    env = TelemetryWrapper(EpistemicRewardWrapper(base, agent_type=agent_type))
    obs, _info = env.reset(seed=seed)

    total_reward = 0.0
    total_lines = 0
    total_keystrokes = 0
    pieces_dropped = 0
    survival_frames = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action_arr, _ = model.predict(obs, deterministic=True)
        action = int(action_arr)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_lines += int(info["lines_cleared_this_step"])
        if action != ACTION_NOP:
            total_keystrokes += 1
        if bool(info["piece_locked"]):
            pieces_dropped += 1
        survival_frames += 1

    return GameResult(
        total_reward=total_reward,
        total_lines=total_lines,
        total_keystrokes=total_keystrokes,
        pieces_dropped=pieces_dropped,
        survival_frames=survival_frames,
        frames=list(env.frames),
    )


def evaluate_checkpoint(
    model_path: Path,
    agent_type: AgentType,
    num_games: int,
    seed: int,
) -> tuple[AggregateMetrics, list[GameResult]]:
    """Play ``num_games`` games and return aggregate metrics plus per-game results.

    Args:
        model_path: Path to a saved PPO checkpoint.
        agent_type: ``"alpha"`` or ``"beta"``.
        num_games: Number of games to roll out.
        seed: Base seed; game ``i`` uses ``seed + i``.

    Returns:
        ``(metrics, results)`` -- the aggregate metrics across all games and
        the per-game results (frames included, for later top-N selection).
    """
    model = PPO.load(str(model_path), device="cpu")
    results = [play_one_game(model, agent_type, seed + i) for i in range(num_games)]

    total_lines = sum(g.total_lines for g in results)
    total_keystrokes = sum(g.total_keystrokes for g in results)
    total_pieces = sum(g.pieces_dropped for g in results)
    total_frames = sum(g.survival_frames for g in results)

    metrics = AggregateMetrics(
        avg_lines=total_lines / num_games,
        avg_keystrokes_per_piece=(total_keystrokes / total_pieces) if total_pieces else 0.0,
        avg_survival_frames=total_frames / num_games,
    )
    return metrics, results


def evaluate_agent(
    *,
    agent_type: AgentType,
    models_root: Path,
    checkpoints: tuple[int, ...],
    num_games: int,
    num_samples: int,
    seed: int,
) -> AgentTelemetry:
    """Evaluate one agent across all checkpoints and assemble its telemetry payload.

    Args:
        agent_type: ``"alpha"`` or ``"beta"``.
        models_root: Root directory containing ``{agent}/ckpt_<timesteps>.zip``.
        checkpoints: Timestep milestones to evaluate.
        num_games: Number of games per checkpoint.
        num_samples: Top-N sample games per checkpoint to include in the output.
        seed: Base RNG seed; each ``(checkpoint, game_idx)`` pair derives its own.

    Returns:
        A fully-populated :class:`AgentTelemetry`.
    """
    aggregate_metrics: dict[str, AggregateMetrics] = {}
    sample_games: list[SampleGame] = []

    for ckpt_step in checkpoints:
        model_path = models_root / agent_type / f"ckpt_{ckpt_step}.zip"
        ckpt_seed = seed + ckpt_step  # ensure different game streams per checkpoint
        metrics, results = evaluate_checkpoint(
            model_path=model_path,
            agent_type=agent_type,
            num_games=num_games,
            seed=ckpt_seed,
        )
        aggregate_metrics[str(ckpt_step)] = metrics

        top = sorted(results, key=lambda g: -g.total_reward)[:num_samples]
        for game in top:
            sample_games.append(
                SampleGame(
                    metadata=SampleGameMetadata(
                        agent_type=agent_type,
                        epoch=ckpt_step,
                        total_lines_cleared=game.total_lines,
                        avg_keystrokes_per_piece=game.keystrokes_per_piece,
                    ),
                    game_frames=[f.to_dict() for f in game.frames[:EXPORT_FRAMES_PER_GAME_CAP]],
                )
            )

    return AgentTelemetry(
        agent_type=agent_type,
        aggregate_metrics=aggregate_metrics,
        sample_games=sample_games,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Epistemic Arcade agents.")
    parser.add_argument(
        "--agent",
        choices=["alpha", "beta"],
        required=True,
        help="Which agent's checkpoints to evaluate.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=DEFAULT_NUM_GAMES,
        help="Number of games per checkpoint.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of top-reward sample games per checkpoint to export.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help="Checkpoint timesteps to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Directory containing the per-agent model subdirectories.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory under which telemetry JSON files are written.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint -- writes ``data/telemetry_<agent>.json``."""
    args = _parse_args()
    telemetry = evaluate_agent(
        agent_type=args.agent,
        models_root=args.models_root,
        checkpoints=tuple(args.checkpoints),
        num_games=args.num_games,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    out_path = args.data_root / f"telemetry_{args.agent}.json"
    write_telemetry_json(telemetry, out_path)
    print(f"[telemetry] wrote {out_path}")


if __name__ == "__main__":
    main()
