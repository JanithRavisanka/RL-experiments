"""
Phase 3 — Advanced Algorithm Experiments (unified API).

Uses ``rl_experiments.api.training.train`` with ``run_id`` under ``logs/<run_id>/``.
"""

from __future__ import annotations

import os

import numpy as np
from rich.console import Console
from rich.rule import Rule

from rl_experiments.analysis.plot_results import generate_all_plots
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


if __name__ == "__main__":
    import rl_experiments.api.registry  # noqa: F401

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    rid = make_run_id()
    train(TrainConfig("dreamer", "CartPole-v1", 0, run_id=rid, budget_episodes=300))
    train(TrainConfig("muzero", "CartPole-v1", 0, run_id=rid, budget_episodes=200))

    console.rule("[bold cyan]Phase 4 — Generating Analysis Plots")
    generate_all_plots()
