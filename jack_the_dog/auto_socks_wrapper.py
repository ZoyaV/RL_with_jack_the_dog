import gymnasium as gym
from gymnasium import Wrapper

from .socks_env import ACTION


class AutoSocksWrapper(Wrapper):
    """
    Wrapper that automatically picks up the sock when Jack is on the socks cell
    and automatically puts the sock when Jack is on the goal cell.

    After each step, if the episode is not done:
    - If Jack is on (socks_x, socks_y) and does not have socks → perform PICK_UP.
    - Else if Jack is on (target_x, target_y) and has socks → perform PUT.

    The agent can still use PICK_UP and PUT actions; they are redundant when
    auto-triggered but harmless.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        while not terminated and not truncated:
            dog_x, dog_y = int(observation[0]), int(observation[1])
            socks_x, socks_y = int(observation[2]), int(observation[3])
            target_x, target_y = int(observation[4]), int(observation[5])
            has_socks = int(observation[6])

            if (dog_x, dog_y) == (socks_x, socks_y) and not has_socks:
                observation, r, terminated, truncated, info = self.env.step(ACTION.PICK_UP)
                reward += r
            elif (dog_x, dog_y) == (target_x, target_y) and has_socks:
                observation, r, terminated, truncated, info = self.env.step(ACTION.PUT)
                reward += r
            else:
                break

        return observation, reward, terminated, truncated, info
