# Epistemic Arcade — Backend

Data-generation backend for a visual essay on Andy Clark & David Chalmers' "Extended Mind" thesis. We train two PPO Tetris agents with contrasting reward shaping and export annotated JSON telemetry to drive a web scrollytelling frontend. The deliverable is the JSON, not a Tetris champion — favour clarity of the philosophical contrast over peak play strength.

Source of truth for requirements: `spec.md`.

## Agents

- **Agent Alpha — Internalist.** Reward = standard line-clear score **minus** `0.01` per non-NOP action. Pays for every keystroke, so it learns to compute placements internally and act minimally.
- **Agent Beta — Extended Mind.** Reward = standard line-clear score, no per-action penalty. Free to "think out loud" by manipulating the piece — rotating, sliding back and forth — to offload spatial reasoning onto the environment.

## `is_epistemic_action` (do not redefine)

Per-frame boolean in the telemetry JSON. `True` iff the action is either:

1. **Redundant rotation** — total rotations within the current fall sequence exceed the minimum needed to reach the piece's final orientation at lock-down. The last `n - (n mod k)` rotation actions are flagged, where `n` = rotations issued and `k` = the piece's distinct orientations (O=1, I/S/Z=2, J/L/T=4).
2. **Translation reversal** — the action is Left after a Right (or vice versa) within the same fall sequence; flag the reversing action.

Why this specific shape: defining *every* lateral move as epistemic collapses the Alpha/Beta distinction (Alpha must still move pieces to place them). We flag only pragmatically redundant actions — the "exploratory fidgeting" that's the visible signature of the Extended Mind agent.

## Tech stack

- **Python 3.12**, locked via `pyenv` (`.python-version`).
- **`uv`** for env + deps. `uv sync` to install, `uv add <pkg>` to add.
- **`ruff`** for lint + format. Google-style docstrings, line length 100.
- **`ty`** for type checking (Astral, pre-release). Fall back to `mypy` if blocked.
- **`pytest`** + **`hypothesis`** + **`doctest`** (wired through pytest via `addopts = --doctest-modules`).
- **`stable-baselines3`** (PPO) + **`gymnasium`** + `SubprocVecEnv` for multi-core CPU training. Training is **CPU-bound by design**; don't introduce GPU assumptions.

## Common commands

```bash
uv sync                                # install deps
uv run pytest                          # tests (incl. doctest + hypothesis)
uv run ruff check && uv run ruff format
uv run ty check                        # mypy fallback if ty is unstable
uv run python -m epistemic_arcade.agents.train --agent alpha --total-timesteps 1_000_000 --n-envs 8
uv run python -m epistemic_arcade.agents.evaluate --agent alpha
```

## Layout

```
src/epistemic_arcade/
  env/      tetris.py · pieces.py · reward_wrapper.py · telemetry_wrapper.py
  agents/   train.py · evaluate.py
  io/       telemetry.py        # JSON schema + serialization
  utils/    rotations.py        # rotation math (doctested)
tests/      test_env · test_reward_wrapper · test_telemetry_wrapper · test_state_space
models/     PPO checkpoints (gitignored)
data/       telemetry_alpha.json, telemetry_beta.json (gitignored)
```

## Environment contract

- Observation: `Box(0, 1, shape=(20, 10), dtype=int8)` — locked cells + active piece overlaid.
- Action: `Discrete(5)` — `0:NOP, 1:Left, 2:Right, 3:Rotate CW, 4:Hard Drop`.
- Reward (base, before wrappers): standard Tetris `{1:1, 2:3, 3:5, 4:8}` lines→points, plus `+0.01`/frame survival bonus. The Alpha/Beta penalty difference lives entirely in `reward_wrapper.py`.

## Telemetry JSON

Per-agent output file (`data/telemetry_alpha.json`, `data/telemetry_beta.json`) contains aggregate metrics keyed by checkpoint, plus top-5 sample games (by cumulative shaped reward) per checkpoint, each game following the frame schema from spec §5.2. See `io/telemetry.py` for the authoritative dataclass.

**Eval-time anti-bloat patches** (apply only during evaluation, not training):

- Each evaluation episode is wrapped in `gymnasium.wrappers.TimeLimit(max_episode_steps=3000)`; on hit, `truncated=True` ends the game cleanly. Prevents a perfect policy from hanging the eval loop.
- `board_state` is exported as a **200-char `"0"/"1"` string** (row-major), not a nested int array. Cuts payload size roughly 3x and stays human-readable.
- `game_frames` is sliced to **at most the first 1 000 frames** per sample game on export. Aggregate metrics still reflect the full untruncated game.

## Checkpoints

PPO timesteps, not "epochs": **10_000**, **250_000**, **1_000_000**.
