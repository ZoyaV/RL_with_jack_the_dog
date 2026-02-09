<p align="center">
  <img src="jack_the_dog/textures/dog.png" alt="Jack the dog" width="120" />
</p>

# RL with Jack the dog

Educational reinforcement learning project with a dog-themed environment and accompanying seminar materials. The agent is **Jack the dog**, who finds himself in various situations solved through reinforcement learning: sometimes he learns new commands, such as putting socks in a box in a grid environment; sometimes he learns the right strategy—for example, avoiding the cold street and hiding from his master and many other funny stories.

## Contents

- **`jack_the_dog/`** — Gymnasium-based environments: `SocksGridEnv`, `NoWalkEnv`, and wrappers (`EpisodeLengthWrapper`, `AutoSocksWrapper`, `StateIndexWrapper`, etc.).
- **`seminars/`** — Jupyter notebooks covering Gym intro, dynamic programming, SARSA, Q-learning vs SARSA, and DQN, with task and solution versions.

## Setup

```bash
# Create conda environment
conda create -n jack_the_dog python=3.9 -y
conda jack_the_dog rl_seminara

# Install package (editable) and dependencies
pip install -e .
# Or dependencies only:
pip install -r requirements.txt
```

## Quick Example

```python
from jack_the_dog import SocksGridEnv, EpisodeLengthWrapper

env = SocksGridEnv(render_mode="rgb_array")
env = EpisodeLengthWrapper(env, max_episode_steps=50)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

See `jack_the_dog/README.md` for environment details and `seminars/` for notebooks.
