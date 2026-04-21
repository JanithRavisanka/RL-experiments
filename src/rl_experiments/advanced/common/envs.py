from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


def make_pixel_env(env_id: str, seed: int = 0, width: int = 84, height: int = 84, frame_stack: int = 4):
    base = gym.make(env_id, render_mode="rgb_array")
    return PixelEnvAdapter(base, width=width, height=height, frame_stack=frame_stack, seed=seed)


def make_state_env(env_id: str, seed: int = 0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def obs_to_chw(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 3:
        return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 4:
        return np.transpose(arr, (0, 3, 1, 2))
    return arr


class PixelEnvAdapter:
    """
    Render-based pixel adapter for environments that expose state observations.
    Returns (H, W, C_stack) uint8 observations.
    """

    def __init__(self, env, width: int = 84, height: int = 84, frame_stack: int = 4, seed: int = 0):
        self.env = env
        self.width = width
        self.height = height
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.action_space = env.action_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, frame_stack),
            dtype=np.uint8,
        )
        self._seed = seed

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        # simple nearest-neighbor resize via slicing (dependency-free)
        h, w = frame.shape[:2]
        ys = np.linspace(0, h - 1, self.height).astype(np.int32)
        xs = np.linspace(0, w - 1, self.width).astype(np.int32)
        small = frame[ys][:, xs]
        gray = (0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]).astype(np.uint8)
        return gray

    def _stack_obs(self):
        return np.stack(list(self.frames), axis=-1)

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=self._seed if seed is None else seed)
        frame = self.env.render()
        proc = self._preprocess(frame)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(proc)
        return self._stack_obs(), info

    def step(self, action):
        _obs, reward, term, trunc, info = self.env.step(action)
        frame = self.env.render()
        self.frames.append(self._preprocess(frame))
        return self._stack_obs(), reward, term, trunc, info

    def close(self):
        self.env.close()

