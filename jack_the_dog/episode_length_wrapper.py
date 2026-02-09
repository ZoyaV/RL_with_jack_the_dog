import gymnasium as gym
from gymnasium import Wrapper


class EpisodeLengthWrapper(Wrapper):
    """
    Wrapper that terminates episodes after a maximum number of steps.
    """
    
    def __init__(self, env, max_episode_steps=50):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        self.step_count += 1
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Truncate if max steps reached
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        return observation, reward, terminated, truncated, info
