# env — RL Seminar Environment

Gymnasium-based environment for the RL seminar: **DogSocks** — a 4×4 grid world where an agent (dog) must pick up socks and deliver them to a target cell.

## Contents

- **`socks_env.py`** — `SocksGridEnv`: core environment and `ACTION` enum.
- **`episode_length_wrapper.py`** — `EpisodeLengthWrapper`: limits episode length (truncation after max steps).

## Setup

From the **project root** (`rl_seminara`), use the root `setup.py` to install the env package:

```bash
# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install env package in editable mode (so `env` is importable)
pip install -e .

# Or install dependencies only (run notebooks from project root)
pip install -r requirements.txt
```

When the project is installed with `pip install -e .`, you can import from anywhere:

```python
from env import SocksGridEnv, EpisodeLengthWrapper, ACTION
```

If you did not install the package, run scripts and Jupyter from the project root and ensure the root is on `PYTHONPATH` (e.g. start Jupyter from the project root).

## Environment Summary

- **Grid:** 4×4.
- **Actions:** `LEFT`, `RIGHT`, `TOP`, `DOWN`, `PICK_UP`, `PUT` (see `ACTION`).
- **Observation:** `[dog_x, dog_y, socks_x, socks_y, target_x, target_y, has_socks]` (all in 0..3, `has_socks` 0 or 1).
- **Reward:** +1 for putting socks on the target (episode ends); −1 for putting socks on a non-target cell.
- **Render modes:** `"human"`, `"rgb_array"`.

Textures (cell, dog, socks) are loaded from the project’s `textures/` directory inside `env/textures/`.

## Quick Example

```python
from env import SocksGridEnv, EpisodeLengthWrapper

env = SocksGridEnv(render_mode="rgb_array")
env = EpisodeLengthWrapper(env, max_episode_steps=50)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

See `seminar_1.ipynb` in the project root for full usage and experiments.
