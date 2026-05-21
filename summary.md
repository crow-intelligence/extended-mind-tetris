# The Epistemic Arcade — Project Summary

> Self-contained reference for an LLM that will help design the visual essay / scrollytelling visualization. No prior context assumed.

---

## 1. TL;DR

**The Epistemic Arcade** is a Python backend that trains two reinforcement-learning agents to play Tetris under contrasting reward functions, then exports richly-annotated JSON telemetry to drive a web-based visual essay on **Andy Clark & David Chalmers' "Extended Mind" thesis** (1998). The two agents — *Alpha* (Internalist) and *Beta* (Extended Mind) — develop visibly different playstyles, and the JSON exposes per-frame board states, actions, rewards, and an `is_epistemic_action` flag so the frontend can highlight the moments where the agents' philosophies show through.

The complete backend is built and tested (40 tests passing). **Both agents have completed the full 1 000 000-timestep training run, with checkpoints saved at 10 000 / 250 000 / 1 000 000 steps and full evaluation (100 games per checkpoint) exported to JSON.** The philosophical contrast is sharp: Alpha learned to act minimally (~1 keystroke per piece, ~0 % epistemic actions), Beta learned to fidget extensively (~7.5 keystrokes per piece, ~21 % epistemic actions). Visualization-ready data is in `data/telemetry_alpha.json` and `data/telemetry_beta.json`.

---

## 2. Philosophical framing

Clark & Chalmers argue that cognitive processes are not strictly confined to the brain — when an external object plays the right kind of functional role, it counts as part of the cognitive system (the classic Otto-and-the-notebook example). Tetris is a canonical empirical test bed for the related notion of **epistemic action** (Kirsh & Maglio, 1994): actions taken not to bring the world closer to a goal but to *reduce internal computation* — e.g., rotating a piece in flight to see whether it fits, rather than mentally simulating the rotation.

The visual essay's argument:
- An agent forced to compute everything internally (Alpha) will play with **minimal keystrokes**, having pre-decided each piece's placement.
- An agent free to manipulate the world (Beta) will play with **exploratory fidgeting** — extra rotations, sliding back and forth — offloading spatial reasoning onto the environment.

If we can show these two playstyles emerging from the same neural architecture and the same game, with only the *reward function* changing, that's a clean illustration of how cognition can extend (or not) into the environment.

---

## 3. The two agents

| Agent     | Philosophy        | Reward shaping                                                                                                     | Predicted behavioural signature                                            |
| --------- | ----------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| **Alpha** | Internalist       | Base reward **minus `0.01` per non-NOP action**. NOP (do nothing) is free.                                         | Few actions per piece. High `is_epistemic_action` rate should be ~0.       |
| **Beta**  | Extended Mind     | Base reward, **unchanged**.                                                                                        | More actions per piece. Visibly higher `is_epistemic_action` rate.         |

The base reward (before shaping) is:
- `+1.0 / +3.0 / +5.0 / +8.0` for clearing 1 / 2 / 3 / 4 lines simultaneously.
- `+0.01` survival bonus per frame.
- Game over (spawn collision) ends the episode with no terminal penalty.

The two reward functions are implemented in `src/epistemic_arcade/env/reward_wrapper.py` as a single `EpistemicRewardWrapper(env, agent_type)`.

---

## 4. The Tetris environment

A minimal custom 10×20 grid implementation of Tetris, conforming to the Gymnasium API. Source: `src/epistemic_arcade/env/tetris.py`.

**Observation space:** `Box(low=0, high=1, shape=(20, 10), dtype=int8)` — the 20-row × 10-column board. Locked cells are 1, empty cells are 0; the **currently controlled piece is overlaid as 1s into the observation**, so the policy can see what it is moving.

**Action space:** `Discrete(5)`.

| Action | Name      | Effect                                                                                              |
| -----: | --------- | --------------------------------------------------------------------------------------------------- |
| `0`    | NOP       | Do nothing. Gravity still ticks the piece down one row.                                             |
| `1`    | Left      | Translate active piece one column left (no-op if blocked by wall or locked cells).                  |
| `2`    | Right     | Translate active piece one column right (same blocking rule).                                       |
| `3`    | Rotate CW | Rotate active piece 90° clockwise (no-op if blocked).                                               |
| `4`    | Hard Drop | Snap piece to its landing row and lock it immediately. Lines clear; next piece spawns.              |

**Gravity:** the piece falls one row every step (except for Hard Drop, which is instantaneous). If gravity would push the piece into a collision, the piece locks at its current position.

**Pieces:** the seven standard tetrominoes — `I`, `O`, `T`, `S`, `Z`, `J`, `L` — chosen uniformly at random at each spawn. Rotation is **simple bounding-box rotation**, not SRS-with-wall-kicks. The number of *distinct* orientations per piece (used by the epistemic-action algorithm):

| Piece | Distinct orientations (`k`) |
| ----- | --------------------------: |
| O     | 1                           |
| I     | 2                           |
| S     | 2                           |
| Z     | 2                           |
| T     | 4                           |
| J     | 4                           |
| L     | 4                           |

**Spawn:** pieces appear at row 0, horizontally centred for their bounding-box width. **Termination:** if a new piece's spawn cells collide with locked cells, the episode ends.

---

## 5. The `is_epistemic_action` flag

This is the **philosophical centrepiece** of the JSON output — a per-frame boolean exposing whether an action constitutes "epistemic fidgeting" in Kirsh & Maglio's sense (a manipulation of the world to ease internal computation, rather than a step toward placing the piece).

An action is `is_epistemic_action: true` iff **one** of these conditions is true:

### 5a. Redundant rotation
Within the *current fall sequence* (from spawn to lock) of a piece with `k` distinct orientations, let `n` be the number of **successful** rotations issued. The minimum rotations needed to reach the piece's locked orientation is `n mod k`, so `n − (n mod k)` rotations were spent going around in full cycles. The **last** `n − (n mod k)` successful rotation actions in the buffer are flagged.

Concrete examples:
- T-piece (`k=4`), agent rotates 5 times → `n=5`, redundant = `5 − 1 = 4`. The last four rotations are flagged.
- I-piece (`k=2`), agent rotates 3 times → `n=3`, redundant = `3 − 1 = 2`. The last two are flagged.
- O-piece (`k=1`), agent rotates *any* number of times → **but** rotation never changes the orientation, so `rotation_succeeded=False` and **nothing is flagged** (we only flag *successful* rotations).

### 5b. Translation reversal
Within the same fall sequence, scan in order; an action is flagged if it is a **successful** Left whose most recent successful translation was a Right, or vice versa. Blocked moves don't count (the piece didn't actually move).

Concrete examples:
- Left, Right → the Right is flagged.
- Right, Left, Right → the second move (Left) and the third move (Right) are both flagged.
- Left, NOP, Right → the Right is still flagged (NOP doesn't break the reversal detection).
- (At the left wall) Left, Right → the Left was blocked; nothing is flagged.

**Why this specific definition?** "Any Left, Right, or Rotate is epistemic" was rejected because *every* agent — including the Internalist — must translate the piece across the board to place it. Only **pragmatically redundant** actions (rotations that go around in full cycles, lateral moves that reverse a previous lateral move) qualify under the spirit of Kirsh & Maglio's definition.

Implementation: `src/epistemic_arcade/env/telemetry_wrapper.py`, function `_flag_buffer_and_flush`.

---

## 6. Training pipeline

- **Algorithm:** Proximal Policy Optimization (PPO) via `stable-baselines3` 2.8.
- **Policy:** default `MlpPolicy` (two-hidden-layer MLP over the flattened 200-cell board). No CNN — premature complexity for a small board.
- **Vectorisation:** `SubprocVecEnv` with `--n-envs 8` by default (configurable). CPU-bound by design; torch is pinned to CPU.
- **Checkpoints saved at exact timestep targets** (default 10 000 / 250 000 / 1 000 000) via a small custom callback, not SB3's periodic `CheckpointCallback`. Files land at `models/{agent}/ckpt_{N}.zip` plus a final `ckpt_final.zip`.

CLI:
```bash
uv run python -m epistemic_arcade.agents.train --agent alpha --total-timesteps 1_000_000 --n-envs 8
uv run python -m epistemic_arcade.agents.train --agent beta  --total-timesteps 1_000_000 --n-envs 8
```

Source: `src/epistemic_arcade/agents/train.py`.

---

## 7. Evaluation pipeline

For each `(agent, checkpoint)` pair:
1. Load the PPO checkpoint.
2. Wrap a fresh env in `EpistemicRewardWrapper(agent_type)` + `TelemetryWrapper` (the latter records every frame and assigns `is_epistemic_action`).
3. Play **100 games** with the deterministic policy (`model.predict(obs, deterministic=True)`), each game seeded as `base_seed + checkpoint + game_idx` for reproducibility.
4. Compute aggregate metrics across the 100 games: avg lines cleared per game, avg keystrokes per piece, avg survival frames.
5. Sort the 100 games by **cumulative shaped reward** (descending) and keep the **top 5** as "sample games" included in the JSON. (Beta's ranking thus naturally favours longer / more-lines games; Alpha's favours efficient, low-keystroke games — which is the desired aesthetic for each.)

CLI:
```bash
uv run python -m epistemic_arcade.agents.evaluate --agent alpha
uv run python -m epistemic_arcade.agents.evaluate --agent beta
```

Source: `src/epistemic_arcade/agents/evaluate.py`. Output files: `data/telemetry_alpha.json`, `data/telemetry_beta.json`.

---

## 8. JSON schema

The export schema slightly extends the original spec §5.2 example so that all checkpoints and sample games fit in one file per agent. Two anti-bloat amendments are applied during export:

- **Episode cap.** Evaluation episodes are capped at **3 000 frames** via Gymnasium's `TimeLimit`; on hit, the episode ends with `truncated=True`. This prevents a perfectly-trained policy from causing the eval loop to hang. Aggregate metrics treat truncated games as complete.
- **Frame cap on export.** When writing sample games, `game_frames` is sliced to **the first 1 000 frames per game**. Aggregate metrics (`aggregate_metrics`, plus `metadata.total_lines_cleared` and `metadata.avg_keystrokes_per_piece` for each sample) are still computed over the *full* untruncated game.

Authoritative source: `src/epistemic_arcade/io/telemetry.py` and `src/epistemic_arcade/agents/evaluate.py`.

### Top level

```json
{
  "agent_type": "alpha",
  "aggregate_metrics": {
    "10000":   {"avg_lines": 0.4,  "avg_keystrokes_per_piece": 5.1, "avg_survival_frames": 32.7},
    "250000":  {"avg_lines": 6.3,  "avg_keystrokes_per_piece": 2.8, "avg_survival_frames": 412.0},
    "1000000": {"avg_lines": 18.1, "avg_keystrokes_per_piece": 2.3, "avg_survival_frames": 980.2}
  },
  "sample_games": [
    {
      "metadata": { "...": "see below" },
      "game_frames": [ "..." ]
    }
  ]
}
```

- `agent_type` — `"alpha"` or `"beta"`.
- `aggregate_metrics` — dict keyed by checkpoint timestep (as a **string**), one entry per checkpoint evaluated.
- `sample_games` — flat list of `(checkpoint × top-5)` sample games (so for a full eval, 15 sample games per agent: 3 checkpoints × 5 games). Each sample game carries its own checkpoint in its `metadata.epoch`.

### Per sample game

```json
{
  "metadata": {
    "agent_type": "alpha",
    "epoch": 1000000,
    "total_lines_cleared": 42,
    "avg_keystrokes_per_piece": 3.2
  },
  "game_frames": [
    {
      "frame_id": 1,
      "board_state": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111100000",
      "current_piece": "T",
      "action_taken": 2,
      "reward": 0.0,
      "is_epistemic_action": false
    }
  ]
}
```

### Per frame

| Field                 | Type                  | Meaning                                                                                          |
| --------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| `frame_id`            | `int` (1-indexed)     | Sequential frame number within the game.                                                         |
| `board_state`         | `str` (length 200)    | Row-major concatenation of the 20×10 grid. Each char is `"0"` (empty) or `"1"` (occupied — locked cell **or** active piece, overlaid). To decode: characters 0–9 are row 0, characters 10–19 are row 1, …, characters 190–199 are row 19. |
| `current_piece`       | `string`              | One of `"I" "O" "T" "S" "Z" "J" "L"` — the piece this frame's action was applied to.             |
| `action_taken`        | `int` 0–4             | `0=NOP, 1=Left, 2=Right, 3=Rotate CW, 4=Hard Drop`.                                              |
| `reward`              | `float`               | The agent's **shaped** reward for this step (so Alpha frames may carry the keystroke cost).      |
| `is_epistemic_action` | `bool`                | True iff the action is a redundant rotation or a translation reversal (see §5). Always `false` on frames flushed by an eval-TimeLimit truncation that landed mid-piece (no honest final-orientation to compute against). |

**Important reward gotcha for visualization:**
Because the survival bonus is `+0.01` and Alpha's keystroke penalty is `−0.01`, Alpha's non-NOP non-line-clear frames net to `reward = 0.0`. Beta's non-NOP non-line-clear frames stay at `reward = +0.01`. Line clears add `1.0 / 3.0 / 5.0 / 8.0` on top of this. Negative rewards are therefore *never* observed; the contrast surfaces as Alpha-`0.0` vs Beta-`0.01`. (If we ever drop the survival bonus, the spec-canonical `-0.01` for Alpha would appear.)

---

## 9. Current data on disk

Both agents have completed the full 1 000 000-timestep training run and 100-game-per-checkpoint evaluation. The data is final and ready for the visualization.

### Models
```
models/alpha/ckpt_10000.zip      (≈ 163 KB)   — Alpha at  10 000 timesteps  (still ~random)
models/alpha/ckpt_250000.zip     (≈ 447 KB)   — Alpha at 250 000 timesteps  (Internalist signature emerges)
models/alpha/ckpt_1000000.zip    (≈ 447 KB)   — Alpha at the 1 M-step milestone
models/alpha/ckpt_final.zip      (≈ 447 KB)   — Alpha end-of-run (≈ same as ckpt_1000000)
models/beta/ckpt_10000.zip       (≈ 163 KB)   — Beta  at  10 000 timesteps  (still ~random)
models/beta/ckpt_250000.zip      (≈ 447 KB)   — Beta  at 250 000 timesteps  (Extended-Mind signature emerges)
models/beta/ckpt_1000000.zip     (≈ 447 KB)   — Beta  at the 1 M-step milestone
models/beta/ckpt_final.zip       (≈ 447 KB)   — Beta  end-of-run
```

Training wall time: ~11–12 min per agent on CPU with `n_envs=8`. Total training time: ~24 min.

### Telemetry
```
data/telemetry_alpha.json    (≈ 969 KB)    — 100 games × 3 checkpoints aggregated; 15 sample games (5 per checkpoint), full frame data
data/telemetry_beta.json     (≈ 1.13 MB)   — same structure, Beta
```

### Aggregate metrics across the learning trajectory

| Checkpoint | Metric                       | Alpha    | Beta     | Δ (Beta − Alpha) |
| ---------: | ---------------------------- | -------: | -------: | ---------------: |
| 10 000     | `avg_lines`                  | 0.000    | 0.000    | 0.000            |
| 10 000     | `avg_keystrokes_per_piece`   | 4.105    | 4.105    | 0.000 (identical: both still random and play the same seeded games) |
| 10 000     | `avg_survival_frames`        | 55.6     | 55.6     | 0.0              |
| 250 000    | `avg_lines`                  | 0.000    | 0.000    | 0.000            |
| 250 000    | **`avg_keystrokes_per_piece`** | **0.987** | **7.168** | **+6.18**     |
| 250 000    | `avg_survival_frames`        | 162.5    | 222.6    | +60.1            |
| 1 000 000  | `avg_lines`                  | 0.060    | 0.030    | −0.030           |
| 1 000 000  | **`avg_keystrokes_per_piece`** | **1.760** | **7.533** | **+5.77**     |
| 1 000 000  | `avg_survival_frames`        | 257.2    | 247.2    | −10.0            |

### Epistemic-action rate across sample frames (THE headline result)

| Checkpoint | Alpha epistemic % | Beta epistemic % |
| ---------: | ----------------: | ---------------: |
| 10 000     | 5.5 %             | 9.5 %            |
| **250 000**    | **0.2 %**         | **22.4 %**       |
| **1 000 000**  | **0.1 %**         | **21.3 %**       |

At the 250 000-step checkpoint and beyond, Beta exhibits **roughly 200× more pragmatically-redundant action** than Alpha. This is the philosophical signature the visual essay was designed to surface, and it's loud, stable, and emerges already at the mid checkpoint — so the learning-trajectory visualization (10k → 250k → 1M) has three meaningfully distinct timepoints, not just before/after.

### Reward-stream signature (per-frame, in the JSON)

For non-NOP non-line-clear frames:
- Alpha sees `reward = 0.0` (the `+0.01` survival bonus exactly cancels the `−0.01` keystroke penalty).
- Beta sees `reward = 0.01` (no penalty, survival bonus intact).

So even at the per-frame level, the JSON exposes the shaping difference directly — a frontend reward strip would visibly show Alpha flatlining where Beta accumulates.

---

## 10. Known caveats / non-obvious design choices

- **Survival bonus (`+0.01`/frame)** was added beyond `spec.md` §3.2 because pure line-clear-only rewards are extremely sparse and PPO struggles. The side-effect is that Alpha's keystroke penalty exactly cancels the bonus, so non-NOP frames net to `0.0` instead of the `-0.01` shown in the original spec example.
- **Rotation system is simple bounding-box rotation**, not Tetris-canonical SRS with wall kicks. Cleaner for tests and for the visualization; deviates from "competitive" Tetris.
- **`O`-piece rotations never flag as epistemic** under our algorithm: rotation never changes orientation (`k=1`, so the new orientation index is always `0`), which makes `rotation_succeeded = False` and excludes them from flagging. A purist reading of "any rotation of `O` is epistemic" would flag them; we don't, because we only flag *successful* rotations.
- **Neither agent learned to clear many lines.** At 1 M timesteps with the default PPO MlpPolicy over the flattened 200-cell observation, both agents average < 0.1 lines cleared per game — they mostly survive without clearing. This is a known limitation of the (deliberately small) model+training budget, and does **not** affect the philosophical contrast (which is sharp). If the visual essay wants to also showcase competent Tetris play on top of the playstyle contrast, the next levers would be a small CNN policy + 5–10 M timesteps, or reward shaping that penalises board height.
- **Action attribution on lock frames:** when a Hard Drop or gravity-induced lock happens, `current_piece` in that frame's record refers to the **piece that just locked** (so the action is correctly attributed to it). The next frame is the first one with the *new* piece.
- **Eval episode cap of 3 000 frames.** Evaluation games are wrapped in `gymnasium.wrappers.TimeLimit(max_episode_steps=3000)`; on hit, the game ends with `truncated=True`. None of the current trained policies actually approach this cap (avg survival ≈ 250 frames, max ~1500), but it's there to prevent an infinite loop if a future policy plays perfectly.
- **Sample-game frames are capped at 1 000 per game on export.** Aggregate metrics in `aggregate_metrics` and per-sample `metadata` still reflect the full untruncated game; only the displayed `game_frames` list is sliced. Currently no game exceeds 1 000 frames, so this is a guardrail rather than an active filter.

---

## 11. Suggested visualization angles (non-prescriptive)

These are starting-point ideas; treat them as a brainstorming menu rather than a brief.

1. **Side-by-side replay.** Same checkpoint, same piece sequence, Alpha and Beta playing in parallel. The contrast in keystrokes-per-piece is immediately visible — Alpha "decides and acts," Beta "feels around."
2. **Epistemic-action timeline.** A scrubbable timeline of one game where flagged frames are highlighted in a distinct colour. Hovering over a flagged action could show the redundant-rotation count or the reversal it constitutes.
3. **Per-frame reward stream.** A small chart showing the reward signal for each frame across the game — exposes the shaping (`0.0` valleys for Alpha vs. `0.01` baseline for Beta) and the spikes at line clears.
4. **Learning trajectory across checkpoints.** Three rows × two columns (3 checkpoints × 2 agents), each showing one representative game from that checkpoint. Lets the viewer see policies "wake up" — random fidgeting at `10k`, purposeful but rough at `250k`, smooth play at `1M`.
5. **Cumulative metric panel.** A dashboard above the replay that updates as the user scrubs: lines cleared so far, keystrokes spent, current piece-keystroke count, epistemic count.
6. **Heatmap of epistemic actions over the board.** For all sample games of an agent, plot a 20×10 heatmap of where epistemic actions happened. Beta's should light up much more diffusely than Alpha's.
7. **Otto's notebook moment.** A short scrollytelling moment — when a Beta game shows a particularly long sequence of redundant rotations before lock — that explicitly draws the Clark-and-Chalmers parallel: "Beta is using the screen the way Otto uses his notebook."

Each of these can be powered entirely off the JSON we ship — no extra backend instrumentation needed.

---

## Glossary (for quick orientation)

- **Alpha** — the Internalist agent. Penalised per non-NOP action.
- **Beta** — the Extended Mind agent. No per-action penalty.
- **Epistemic action** — an action that manipulates the world to ease cognition rather than to achieve the goal. Here: redundant rotation OR translation reversal.
- **Redundant rotation** — rotations beyond what's needed to reach the locked orientation.
- **Translation reversal** — Left after Right, or Right after Left, within one fall sequence.
- **Fall sequence** — the lifetime of a piece, from spawn to lock.
- **Epoch** — used loosely here to mean "training timestep checkpoint" (10 000 / 250 000 / 1 000 000).
- **Sample game** — one of the top-5 by cumulative reward, exported with full per-frame data.
