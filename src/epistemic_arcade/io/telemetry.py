"""JSON telemetry schema and serialization for the visual essay export.

The schema layers on top of the per-frame structure in ``spec.md`` §5.2 to
hold three checkpoints' worth of aggregate metrics plus the top-five
sample games per checkpoint in one file per agent. See ``CLAUDE.md`` for
the definition of ``is_epistemic_action``.

The dataclasses here are intentionally thin -- they exist so callers can
build telemetry programmatically and so the schema lives in one place.
:func:`write_telemetry_json` performs the only file write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AggregateMetrics:
    """Per-checkpoint aggregate evaluation statistics over many games.

    Attributes:
        avg_lines: Mean lines cleared per game.
        avg_keystrokes_per_piece: Mean non-NOP actions per dropped piece across all games.
        avg_survival_frames: Mean episode length in frames.
    """

    avg_lines: float
    avg_keystrokes_per_piece: float
    avg_survival_frames: float

    def to_dict(self) -> dict[str, float]:
        """Serialize to a JSON-ready dict.

        Examples:
            >>> AggregateMetrics(1.0, 2.0, 3.0).to_dict()
            {'avg_lines': 1.0, 'avg_keystrokes_per_piece': 2.0, 'avg_survival_frames': 3.0}
        """
        return {
            "avg_lines": self.avg_lines,
            "avg_keystrokes_per_piece": self.avg_keystrokes_per_piece,
            "avg_survival_frames": self.avg_survival_frames,
        }


@dataclass(frozen=True)
class SampleGameMetadata:
    """Per-game metadata block matching ``spec.md`` §5.2.

    Attributes:
        agent_type: ``"alpha"`` or ``"beta"``.
        epoch: Checkpoint timestep at which the controlling model was saved.
        total_lines_cleared: Lines cleared in this specific game.
        avg_keystrokes_per_piece: Non-NOP actions per piece in this game.
    """

    agent_type: str
    epoch: int
    total_lines_cleared: int
    avg_keystrokes_per_piece: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict."""
        return {
            "agent_type": self.agent_type,
            "epoch": self.epoch,
            "total_lines_cleared": self.total_lines_cleared,
            "avg_keystrokes_per_piece": self.avg_keystrokes_per_piece,
        }


@dataclass
class SampleGame:
    """A single sample game: metadata plus its full frame sequence.

    Attributes:
        metadata: Per-game header.
        game_frames: List of spec-§5.2 frame dicts (use
            :meth:`FrameRecord.to_dict` to produce them).
    """

    metadata: SampleGameMetadata
    game_frames: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-ready ``{metadata, game_frames}`` block."""
        return {"metadata": self.metadata.to_dict(), "game_frames": self.game_frames}


@dataclass
class AgentTelemetry:
    """Top-level export structure for one agent.

    Attributes:
        agent_type: ``"alpha"`` or ``"beta"``.
        aggregate_metrics: Mapping from string-keyed checkpoint timesteps to
            their :class:`AggregateMetrics`.
        sample_games: Flat list of top-N sample games across all checkpoints,
            each carrying its own ``epoch`` in :class:`SampleGameMetadata`.
    """

    agent_type: str
    aggregate_metrics: dict[str, AggregateMetrics]
    sample_games: list[SampleGame]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire telemetry payload to a JSON-ready dict."""
        return {
            "agent_type": self.agent_type,
            "aggregate_metrics": {k: v.to_dict() for k, v in self.aggregate_metrics.items()},
            "sample_games": [g.to_dict() for g in self.sample_games],
        }


def write_telemetry_json(telemetry: AgentTelemetry, path: Path) -> None:
    """Write a telemetry payload to disk as UTF-8 JSON.

    Args:
        telemetry: The agent telemetry to write.
        path: Destination path; parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(telemetry.to_dict(), f)
