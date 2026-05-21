# Project Specification: The Epistemic Arcade (Python Backend)

## 1. Project Overview
This project serves as the data-generation backend for a visual essay exploring Andy Clark and David Chalmers' "Extended Mind" thesis. We are using Reinforcement Learning (RL) to train two distinct agents playing Tetris. The goal is to export highly detailed telemetry (board states, actions, rewards, and performance metrics) into JSON format for a web-based scrollytelling visualization. 

## 2. Technical Stack & Tooling Setup
The project mandates modern, strict, and performant Python development practices.

* **Python Version Management:** Use `pyenv` to install and lock a vanilla Python version (e.g., Python 3.11 or 3.12).
* **Package Management & Virtual Environments:** Use `uv` strictly. Initialize the project and manage dependencies (`stable-baselines3`, `gymnasium`, `gym-tetris`, etc.) via `uv pip`.
* **Formatting & Linting:** Use `ruff`. Code must be formatted and linted with strict adherence to standard Python rules.
* **Type Hinting:** Comprehensive type hints are required for all functions and classes. Use `ty` (or equivalent type checking workflow) to enforce type safety.
* **Docstrings:** All modules, classes, and functions must use **Google-style docstrings**.
* **Testing:** * Use `pytest` for unit and integration testing.
    * Use `doctest` embedded in the Google-style docstrings for utility functions.
    * Use `hypothesis` for property-based testing (e.g., ensuring matrix transformations or reward calculations never fall out of expected bounds).

---

## 3. Core Architecture: Environments & Agents

### 3.1 The Tetris Environment
Wrap a standard Gymnasium Tetris environment (such as `gym-tetris` or a custom lightweight 10x20 grid implementation). The environment must expose:
* **Observation Space:** A 2D array (10x20) representing the board state (0 = empty, 1 = occupied).
* **Action Space:** Discrete actions mapping to standard Tetris moves (0: NOP, 1: Left, 2: Right, 3: Rotate, 4: Drop).

### 3.2 Reward Shaping (The Philosophical Core)
The project requires training two models with different reward functions using Proximal Policy Optimization (PPO) via `stable-baselines3`. Implement a custom Gym Wrapper that takes a parameter `agent_type` to shape rewards accordingly.

| Agent | Philosophy | Reward Function Modifications |
| :--- | :--- | :--- |
| **Agent Alpha** (Internalist) | Must compute spatial logic internally. | Standard line clear rewards **MINUS** a strict penalty (e.g., `-0.01`) for every keystroke/action taken. |
| **Agent Beta** (Extended Mind) | Offloads spatial logic to the environment. | Standard line clear rewards. **NO penalty** for keystrokes or rotations. |

---

## 4. Training Pipeline
* **Algorithm:** `stable-baselines3.PPO`.
* **Vectorization:** Use `SubprocVecEnv` to utilize multi-core CPU training.
* **Checkpoints:** Save model weights at regular epoch intervals so we can track their progression from "novice" to "expert".

---

## 5. Evaluation & Telemetry Export (JSON Generation)
This is the most critical phase. After training, the models must be evaluated extensively in inference mode. We need to save the game states to power the web frontend.

### 5.1 Extensive Evaluation Metrics
For both models, run 100 evaluation games and log aggregate metrics:
* Average lines cleared per game.
* Average keystrokes per piece dropped.
* Average survival time (frames).

### 5.2 JSON Telemetry Schema
For the visual essay, generate 5 "perfect" sample games for both Agent Alpha and Agent Beta at specific checkpoints (e.g., Epoch 1000, Epoch 50000, Final).

The output must be saved as `telemetry_alpha.json` and `telemetry_beta.json` matching this exact schema:

```json
{
  "metadata": {
    "agent_type": "alpha",
    "epoch": 50000,
    "total_lines_cleared": 42,
    "avg_keystrokes_per_piece": 3.2
  },
  "game_frames": [
    {
      "frame_id": 1,
      "board_state": [[0,0,0], [0,0,0]], 
      "current_piece": "T",
      "action_taken": 2,
      "reward": -0.01,
      "is_epistemic_action": false
    }
  ]
}
```

---

## 6. Testing Strategy Specs
* **Environment Wrapper Tests (`pytest`):** Verify that Agent Alpha receives penalties for actions while Agent Beta does not.
* **State Space Tests (`hypothesis`):** Generate random valid and invalid 10x20 board matrices to ensure the state evaluator doesn't crash or return impossible rewards.
* **Utility Tests (`doctest`):** Any helper functions for array flattening, JSON serialization, or matrix rotation must include verifiable doctests.
