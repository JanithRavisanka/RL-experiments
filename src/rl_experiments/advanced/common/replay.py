from __future__ import annotations

import random
from collections import deque
import numpy as np


class TransitionReplay:
    def __init__(self, capacity: int = 200_000):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class SequenceReplay:
    def __init__(self, capacity: int = 5000):
        self.episodes = deque(maxlen=capacity)

    def add_episode(self, transitions):
        self.episodes.append(list(transitions))

    def sample_sequences(self, batch_size: int, seq_len: int):
        out = []
        if not self.episodes:
            return out
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            if len(ep) <= seq_len:
                out.append(ep)
            else:
                i = random.randint(0, len(ep) - seq_len - 1)
                out.append(ep[i : i + seq_len])
        return out

