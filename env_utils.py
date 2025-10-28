import gymnasium as gym
import cv2
import numpy as np

# This is a standard wrapper to process Atari observations
class AtariWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AtariWrapper, self).__init__(env)
        # Set observation space to (84, 84, 1) - HxWxChannels
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.preprocess(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = self.preprocess(observation)
        return observation, info

    def preprocess(self, observation):
        # Convert to grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB_GRAY)
        # Resize to 84x84
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        # Add channel dimension (84, 84) -> (84, 84, 1)
        return observation[:, :, None]

# Function to create the stacked environment
def create_env(env_id):
    # This repeats the action for 4 frames and returns the last frame.
    env = gym.make(env_id, render_mode='rgb_array', frameskip=4)
    
    # 1. Apply our custom preprocessing
    env = AtariWrapper(env)
    
    # 2. Use gymnasium's built-in FrameStack wrapper
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    return env
