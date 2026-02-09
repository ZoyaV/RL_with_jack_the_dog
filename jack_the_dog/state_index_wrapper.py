import numpy as np
from gymnasium import spaces
from gymnasium import Wrapper
import inspect


def _obs_to_state_idx(obs, env):
    """
    Compute a state index from a raw observation.

    Supports:
    - Original "socks" grid envs with `_state_to_idx(dog_x, dog_y, socks_x, socks_y, has_socks)`.
    - `NoWalkEnv` with `_state_to_idx(dog_x, dog_y, snack_consumed)`.
    - Fallback to the original formula for SocksGridEnv (grid_size=4).
    """
    base = env.unwrapped
    if hasattr(base, "_state_to_idx"):
        # Determine how many parameters the bound `_state_to_idx` method expects
        n_params = len(inspect.signature(base._state_to_idx).parameters)

        # Socks-style environments: (dog_x, dog_y, socks_x, socks_y, has_socks)
        if n_params == 5:
            return base._state_to_idx(
                int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]), int(obs[6])
            )

        # NoWalkEnv-style: (dog_x, dog_y, snack_consumed)
        if n_params == 3:
            dog_x, dog_y = int(obs[0]), int(obs[1])
            # In NoWalkEnv, snack_x/snack_y == grid_size means "consumed"
            grid_size = getattr(base, "grid_size", 4)
            snack_x = int(obs[6])
            snack_y = int(obs[7]) if len(obs) > 7 else grid_size
            snack_consumed = snack_x >= grid_size or snack_y >= grid_size
            return base._state_to_idx(dog_x, dog_y, snack_consumed)

        # If `_state_to_idx` exists but has an unexpected signature, fall back.
    # Fallback: same formula as SocksGridEnv (grid_size=4)
    grid_size = getattr(base, "grid_size", 4)
    dog_x, dog_y = int(obs[0]), int(obs[1])
    socks_x, socks_y = int(obs[2]), int(obs[3])
    has_socks = int(obs[6])
    dog_pos = dog_x * grid_size + dog_y
    if has_socks:
        return 256 + dog_pos
    socks_pos = socks_x * grid_size + socks_y
    return dog_pos * 16 + socks_pos


class StateIndexWrapper(Wrapper):
    """
    Wrapper that replaces the raw state observation with its ordinal state index.

    Observation becomes a single integer: 0..271 for non-terminal states, 272 for the
    success terminal state. Useful for tabular methods that index by state id.
    """

    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped

        # Default flags
        self._uses_discrete_obs = False
        self._success_state = None  # if not None, used as extra success-terminal index

        # Case 1: Environment already has a discrete observation space.
        # We treat the observation as the state index and reserve one extra
        # index for the success terminal state.
        if isinstance(base.observation_space, spaces.Discrete):
            self._uses_discrete_obs = True
            self._n_states = base.observation_space.n
            self._success_state = self._n_states
            self.observation_space = spaces.Discrete(self._n_states + 1)

        # Case 2: Envs exposing `_state_to_idx`. We may or may not need an
        # extra success state depending on the specific env.
        elif hasattr(base, "_state_to_idx"):
            n_params = len(inspect.signature(base._state_to_idx).parameters)

            if n_params == 3:
                # NoWalkEnv-style: state space already includes terminal states,
                # so we DO NOT add an extra success index.
                if hasattr(base, "get_all_states"):
                    self._n_states = len(base.get_all_states())
                else:
                    self._n_states = 32  # fallback for NoWalkEnv (4x4 * 2)
                self.observation_space = spaces.Discrete(self._n_states)
                self._success_state = None
            else:
                # Socks-style Jack env: we keep the original 0..271 plus one
                # extra success terminal state at index 272.
                self._n_states = 273
                self._success_state = 272
                self.observation_space = spaces.Discrete(self._n_states)

        # Case 3: Fallback to the original Jack-the-dog assumptions.
        else:
            self._n_states = 273
            self._success_state = 272
            self.observation_space = spaces.Discrete(self._n_states)

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if self._uses_discrete_obs:
            state_idx = int(observation)
        else:
            state_idx = _obs_to_state_idx(observation, self.env)
        return np.int32(state_idx), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated and reward == 1 and self._success_state is not None:
            state_idx = self._success_state  # success terminal state
        else:
            if self._uses_discrete_obs:
                state_idx = int(observation)
            else:
                state_idx = _obs_to_state_idx(observation, self.env)
        return np.int32(state_idx), reward, terminated, truncated, info
