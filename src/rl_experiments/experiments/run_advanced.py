"""
Phase 3 — Advanced Algorithm Experiments (unified API).

Uses ``rl_experiments.api.training.train`` with ``run_id`` under ``logs/<run_id>/``.
"""

from __future__ import annotations

from rich.console import Console

from rl_experiments.api.training import TrainConfig, train
from rl_experiments.utils.run_paths import make_run_id

console = Console()


def run_dreamer(env_id: str = "CartPole-v1", n_episodes: int = 300, seed: int = 0, run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold magenta]Phase 3 — Dreamer · {env_id}")
    result = train(
        TrainConfig(
            "dreamer",
            env_id,
            seed,
            run_id=run_id,
            budget_episodes=n_episodes,
        )
    )
    console.print(f"  [green]Dreamer log → {result.log_csv}[/green]")
    return result


def run_muzero(env_id: str = "CartPole-v1", n_episodes: int = 300, seed: int = 0, run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold magenta]Phase 3 — MuZero · {env_id}")
    result = train(
        TrainConfig(
            "muzero",
            env_id,
            seed,
            run_id=run_id,
            budget_episodes=n_episodes,
        )
    )
    console.print(f"  [green]MuZero log → {result.log_csv}[/green]")
    return result
