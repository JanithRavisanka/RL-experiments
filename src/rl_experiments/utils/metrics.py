"""
utils/metrics.py
────────────────
Metrics collection and CSV logging utilities for all RL experiments.
"""

import csv
import os
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from stable_baselines3.common.callbacks import BaseCallback


# ─────────────────────────────────────────────
# Data class to store one episode result
# ─────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    timestep: int
    reward: float
    ep_len: int
    wall_time: float  # seconds since training start


# ─────────────────────────────────────────────
# SB3 Callback – captures episodic rewards
# ─────────────────────────────────────────────

class RLMetricsCallback(BaseCallback):
    """
    Stable-Baselines3 callback that logs episode rewards to a CSV file.
    Compatible with PPO, SAC, DQN.
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_rewards: List[EpisodeRecord] = []
        self._start_time: Optional[float] = None
        self._csv_file = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._csv_file = open(self.log_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestep", "reward", "ep_len", "wall_time"])

    def _on_step(self) -> bool:
        # SB3 stores episode info in self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_len    = info["episode"]["l"]
                ts        = self.num_timesteps
                wt        = time.time() - self._start_time

                record = EpisodeRecord(ts, ep_reward, ep_len, wt)
                self.episode_rewards.append(record)
                self._csv_writer.writerow([ts, ep_reward, ep_len, f"{wt:.2f}"])
                self._csv_file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()

    @property
    def rewards(self) -> np.ndarray:
        return np.array([r.reward for r in self.episode_rewards])

    @property
    def timesteps(self) -> np.ndarray:
        return np.array([r.timestep for r in self.episode_rewards])


# ─────────────────────────────────────────────
# Standalone metrics for custom algorithms
# (Dreamer, MuZero)
# ─────────────────────────────────────────────

class ExperimentLogger:
    """
    Simple CSV logger for custom RL algorithms that don't use SB3.
    Usage:
        logger = ExperimentLogger("logs/dreamer_cartpole.csv")
        logger.log(episode=1, reward=42.0, loss=0.3)
        logger.close()
    """

    def __init__(self, path: str, fields: List[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = fields
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()
        self._start = time.time()

    def log(self, **kwargs):
        kwargs.setdefault("wall_time", f"{time.time() - self._start:.2f}")
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# Smoothing helper
# ─────────────────────────────────────────────

def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values, dtype=float)
    alpha = 2.0 / (window + 1)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed
