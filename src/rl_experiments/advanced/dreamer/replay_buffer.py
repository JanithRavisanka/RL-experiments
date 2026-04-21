"""
advanced/dreamer/replay_buffer.py
─────────────────────────────────────────────────────────────────────────────
Episode replay buffer for Dreamer.

Dreamer collects full episodes and samples random sub-sequences (chunks)
of length L for sequence-level training (the RSSM needs temporal order).

This differs from DQN's transition buffer — here we store full episodes
and sample contiguous chunks.

Paper: Hafner et al. 2019 §B.3 — "collect episodes, sample of length L=50"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class Episode:
    observations: List[np.ndarray] = field(default_factory=list)
    actions:      List[np.ndarray] = field(default_factory=list)
    rewards:      List[float]      = field(default_factory=list)
    dones:        List[bool]       = field(default_factory=list)

    def __len__(self):
        return len(self.rewards)


class EpisodeReplayBuffer:
    """
    Fixed-capacity replay buffer that stores entire episodes.
    Allows sampling random contiguous sub-sequences of length `chunk_len`.

    Storage: up to `max_episodes` episodes, FIFO when full.
    """

    def __init__(
        self,
        max_episodes: int = 500,
        chunk_len:    int = 50,
        obs_dim:      int = 4,
        action_dim:   int = 1,
    ):
        self.max_episodes = max_episodes
        self.chunk_len    = chunk_len
        self.obs_dim      = obs_dim
        self.action_dim   = action_dim
        self._buffer: List[Episode] = []
        self._ptr = 0

    # ─────────────────────────────────────────────────────────────────────────

    def add_episode(self, episode: Episode):
        """Add a completed episode to the buffer."""
        if len(self._buffer) < self.max_episodes:
            self._buffer.append(episode)
        else:
            self._buffer[self._ptr % self.max_episodes] = episode
        self._ptr += 1

    def __len__(self):
        return len(self._buffer)

    def total_steps(self) -> int:
        return sum(len(ep) for ep in self._buffer)

    # ─────────────────────────────────────────────────────────────────────────

    def sample(
        self, batch_size: int, device: torch.device
    ) -> dict:
        """
        Sample `batch_size` random chunks of length `chunk_len`.

        Returns:
          A dict of tensors, each shape (chunk_len, batch_size, dim):
            obs     : (L, B, obs_dim)
            actions : (L, B, action_dim)
            rewards : (L, B, 1)
            dones   : (L, B, 1)
        """
        L, B = self.chunk_len, batch_size

        obs_buf  = np.zeros((L, B, self.obs_dim),    dtype=np.float32)
        act_buf  = np.zeros((L, B, self.action_dim), dtype=np.float32)
        rew_buf  = np.zeros((L, B, 1),               dtype=np.float32)
        done_buf = np.zeros((L, B, 1),               dtype=np.float32)

        # Eligible episodes must be at least chunk_len long
        eligible = [ep for ep in self._buffer if len(ep) >= L]
        if not eligible:
            raise RuntimeError(
                f"No episodes of length ≥ {L} in buffer. "
                "Collect more data before training."
            )

        for b in range(B):
            ep     = eligible[np.random.randint(len(eligible))]
            start  = np.random.randint(0, len(ep) - L + 1)
            for t in range(L):
                obs_buf[t, b]  = ep.observations[start + t]
                act_buf[t, b]  = ep.actions[start + t]
                rew_buf[t, b]  = ep.rewards[start + t]
                done_buf[t, b] = float(ep.dones[start + t])

        return {
            "obs":     torch.tensor(obs_buf,  device=device),
            "actions": torch.tensor(act_buf,  device=device),
            "rewards": torch.tensor(rew_buf,  device=device),
            "dones":   torch.tensor(done_buf, device=device),
        }
