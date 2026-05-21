"""PPO training entrypoint for the Alpha (Internalist) and Beta (Extended Mind) agents.

Trains a single agent at a time (selected via CLI), with `SubprocVecEnv`-based
multi-core CPU rollouts and checkpoints saved at the spec's three target
timesteps (10k / 250k / 1M -- override via ``--checkpoints``).

Training is CPU-bound by design (see ``CLAUDE.md``); torch is pinned to CPU
to keep things reproducible across dev machines.

Run with::

    uv run python -m epistemic_arcade.agents.train --agent alpha --total-timesteps 1_000_000
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Final

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from epistemic_arcade.env.reward_wrapper import AgentType, EpistemicRewardWrapper
from epistemic_arcade.env.tetris import TetrisEnv

DEFAULT_CHECKPOINTS: Final[tuple[int, ...]] = (10_000, 250_000, 1_000_000)
DEFAULT_TOTAL_TIMESTEPS: Final[int] = 1_000_000
DEFAULT_N_ENVS: Final[int] = 8
DEFAULT_MODELS_ROOT: Final[Path] = Path("models")


def make_env(agent_type: AgentType, rank: int, base_seed: int) -> Callable[[], gym.Env]:
    """Return a thunk that builds a fresh shaped Tetris env for a worker process.

    Args:
        agent_type: Either ``"alpha"`` or ``"beta"``.
        rank: Worker index (used to derive a per-worker seed).
        base_seed: Base seed for the run.

    Returns:
        A zero-arg callable that constructs and seed-resets a wrapped env.
    """

    def _thunk() -> gym.Env:
        env = EpistemicRewardWrapper(TetrisEnv(), agent_type=agent_type)
        env.reset(seed=base_seed + rank)
        return env

    return _thunk


class TargetTimestepCheckpointCallback(BaseCallback):
    """Save a model snapshot the first time training crosses each target timestep.

    `stable_baselines3.common.callbacks.CheckpointCallback` ticks on a fixed
    period rather than an arbitrary list of milestones; we want exact saves
    at 10k / 250k / 1M (or whatever the user supplied), no others.
    """

    def __init__(self, target_timesteps: tuple[int, ...], save_dir: Path) -> None:
        """Initialize the callback.

        Args:
            target_timesteps: Timestep milestones at which to save the model.
            save_dir: Directory under which ``ckpt_<timesteps>.zip`` files land.
        """
        super().__init__(verbose=1)
        self._targets: list[int] = sorted(set(target_timesteps))
        self._save_dir: Path = save_dir
        self._next_idx: int = 0

    def _on_step(self) -> bool:
        while (
            self._next_idx < len(self._targets)
            and self.num_timesteps >= self._targets[self._next_idx]
        ):
            target = self._targets[self._next_idx]
            path = self._save_dir / f"ckpt_{target}.zip"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))
            if self.verbose > 0:
                print(f"[checkpoint] saved {path} at {self.num_timesteps} timesteps")
            self._next_idx += 1
        return True


def train(
    *,
    agent_type: AgentType,
    total_timesteps: int,
    n_envs: int,
    checkpoints: tuple[int, ...],
    seed: int,
    models_root: Path,
) -> Path:
    """Train a single agent and return the path to the final saved model.

    Args:
        agent_type: Which reward-shaping variant to train.
        total_timesteps: Number of environment steps to roll out.
        n_envs: Number of parallel envs in the SubprocVecEnv.
        checkpoints: Timestep targets at which to save intermediate models.
        seed: Base RNG seed for the run.
        models_root: Root directory for model checkpoints (subdir per agent).

    Returns:
        Path to the final-model zip on disk.
    """
    torch.set_num_threads(1)  # Avoid oversubscription with SubprocVecEnv
    save_dir = models_root / agent_type
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(agent_type, rank=i, base_seed=seed) for i in range(n_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        verbose=1,
        device="cpu",
    )

    callback = TargetTimestepCheckpointCallback(target_timesteps=checkpoints, save_dir=save_dir)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_path = save_dir / "ckpt_final.zip"
    model.save(str(final_path))
    print(f"[final] saved {final_path}")
    vec_env.close()
    return final_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Epistemic Arcade PPO agent.")
    parser.add_argument(
        "--agent", choices=["alpha", "beta"], required=True, help="Reward-shaping variant."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total environment steps to train for.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help="Number of parallel envs in the SubprocVecEnv.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help="Timestep targets at which to save intermediate checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Directory under which per-agent checkpoint subdirectories are written.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    train(
        agent_type=args.agent,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        checkpoints=tuple(args.checkpoints),
        seed=args.seed,
        models_root=args.models_root,
    )


if __name__ == "__main__":
    main()
