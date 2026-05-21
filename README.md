# The Epistemic Arcade

Data-generation backend for a visual essay on Andy Clark & David Chalmers'
**Extended Mind** thesis. Two PPO agents are trained to play Tetris under
contrasting reward shaping; the project exports annotated JSON telemetry that
drives a web-based scrollytelling frontend.

The deliverable is the **JSON telemetry**, not a Tetris champion — the design
favours a clear philosophical contrast between the agents over peak play
strength.

## The two agents

| Agent | Philosophy | Reward shaping |
| :--- | :--- | :--- |
| **Alpha — Internalist** | Computes placements internally; acts minimally. | Standard line-clear score **minus** `0.01` per non-NOP action. |
| **Beta — Extended Mind** | Thinks "out loud" by manipulating the piece. | Standard line-clear score; **no** per-action penalty. |

Both agents share the same neural architecture and see the same board. The only
difference is the reward function — Alpha pays for every keystroke, so it learns
to reason internally; Beta is free to rotate and slide pieces to offload spatial
reasoning onto the environment.

## `is_epistemic_action`

A per-frame boolean in the telemetry. It is `True` iff the action is either:

1. **Redundant rotation** — total rotations within the current fall sequence
   exceed the minimum needed to reach the piece's final orientation. The last
   `n - (n mod k)` rotation actions are flagged, where `n` = rotations issued
   and `k` = the piece's distinct orientations (O=1, I/S/Z=2, J/L/T=4).
2. **Translation reversal** — a Left after a Right (or vice versa) within the
   same fall sequence; the reversing action is flagged.

Only *pragmatically redundant* actions are flagged — the "exploratory fidgeting"
that is the visible signature of the Extended Mind agent. Flagging every lateral
move would collapse the Alpha/Beta distinction, since Alpha must still move
pieces to place them.

## Tech stack

- **Python 3.12**, locked via `pyenv` (`.python-version`).
- **`uv`** for environments and dependencies.
- **`ruff`** for lint + format (Google-style docstrings, line length 100).
- **`ty`** for type checking (mypy as a fallback if `ty` is unstable).
- **`pytest`** + **`hypothesis`** + **`doctest`** (doctests run via pytest's
  `--doctest-modules`).
- **`stable-baselines3`** (PPO) + **`gymnasium`**, with `SubprocVecEnv` for
  multi-core CPU training. Training is **CPU-bound by design** — there are no
  GPU assumptions.

## Setup

```bash
uv sync          # create the environment and install all dependencies
```

## Common commands

```bash
# Tests (unit + doctest + hypothesis)
uv run pytest

# Lint and format
uv run ruff check && uv run ruff format

# Type check
uv run ty check

# Train an agent (writes checkpoints to models/<agent>/)
uv run python -m epistemic_arcade.agents.train --agent alpha --total-timesteps 1_000_000 --n-envs 8

# Evaluate an agent and export telemetry to data/telemetry_<agent>.json
uv run python -m epistemic_arcade.agents.evaluate --agent alpha
```

PPO checkpoints are saved at **10 000**, **250 000**, and **1 000 000**
timesteps.

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
web/        scrollytelling frontend (see web/README.md)
```

## Environment contract

- **Observation:** `Box(0, 1, shape=(20, 10), dtype=int8)` — locked cells with
  the active piece overlaid.
- **Action:** `Discrete(5)` — `0:NOP, 1:Left, 2:Right, 3:Rotate CW, 4:Hard Drop`.
- **Reward (base, before wrappers):** standard Tetris `{1:1, 2:3, 3:5, 4:8}`
  lines→points, plus a `+0.01`/frame survival bonus. The Alpha/Beta penalty
  difference lives entirely in `env/reward_wrapper.py`.

## Telemetry JSON

Each agent produces one file (`data/telemetry_alpha.json`,
`data/telemetry_beta.json`) containing aggregate metrics keyed by checkpoint,
plus the top-5 sample games (by cumulative shaped reward) per checkpoint. See
`src/epistemic_arcade/io/telemetry.py` for the authoritative dataclass schema.

**Eval-time anti-bloat patches** (applied during evaluation only):

- Each evaluation episode is wrapped in
  `gymnasium.wrappers.TimeLimit(max_episode_steps=3000)` so a perfect policy
  cannot hang the eval loop.
- `board_state` is exported as a **200-char `"0"/"1"` string** (row-major),
  not a nested array — roughly 3× smaller and still human-readable.
- `game_frames` is sliced to **at most the first 1 000 frames** per sample
  game. Aggregate metrics still reflect the full untruncated game.

## Web frontend

The scrollytelling visualization that consumes this telemetry lives in `web/`.
See `web/README.md` for how to serve it locally.

## Source of truth

`spec.md` holds the canonical requirements; `CLAUDE.md` summarizes conventions
for contributors and tooling.
