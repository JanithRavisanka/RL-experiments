"""Public training API: `TrainConfig`, `TrainResult`, `train()`, algorithm registry."""

from rl_experiments.api import registry as _registry  # noqa: F401 — register algorithms
from rl_experiments.api.training import (
    TRAIN_REGISTRY,
    TrainConfig,
    TrainResult,
    train,
)

__all__ = ["TrainConfig", "TrainResult", "train", "TRAIN_REGISTRY"]
