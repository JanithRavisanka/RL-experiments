"""Unified training config and result types for all algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from rl_experiments.utils.run_paths import make_run_id


@dataclass
class TrainConfig:
    """Single training run — model-free and model-based use the same shape."""

    algorithm: str
    env_id: str
    seed: int = 0
    run_id: str | None = None
    quick: bool = False
    budget_steps: int | None = None
    budget_episodes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_run_id(self) -> str:
        return self.run_id or make_run_id()


@dataclass
class TrainResult:
    algorithm: str
    env_id: str
    seed: int
    run_id: str
    log_csv: Path
    artifacts: dict[str, Path] = field(default_factory=dict)


class TrainFn(Protocol):
    def __call__(self, config: TrainConfig) -> TrainResult: ...


TRAIN_REGISTRY: dict[str, TrainFn] = {}


def register_algorithm(name: str, fn: TrainFn) -> None:
    TRAIN_REGISTRY[name] = fn


def train(config: TrainConfig) -> TrainResult:
    """Dispatch training by `config.algorithm` (registered name)."""
    import rl_experiments.api.registry  # noqa: F401 — populate TRAIN_REGISTRY

    rid = config.resolved_run_id()
    config = replace(config, run_id=rid)
    key = config.algorithm.lower()
    if key not in TRAIN_REGISTRY:
        raise KeyError(f"Unknown algorithm {key!r}. Registered: {sorted(TRAIN_REGISTRY)}")
    return TRAIN_REGISTRY[key](config)


def _scale(quick: bool, base: float, quick_scale: float = 0.1) -> int:
    return int(base * (quick_scale if quick else 1.0))


__all__ = [
    "TrainConfig",
    "TrainResult",
    "TrainFn",
    "TRAIN_REGISTRY",
    "register_algorithm",
    "train",
    "_scale",
]
