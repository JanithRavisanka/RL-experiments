#!/usr/bin/env python3
"""
Master runner for all RL experiments.

Usage:
  rl-experiments              # or: python -m rl_experiments.cli.run_all
  rl-experiments --phase 1
  rl-experiments --quick --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_experiments.api.training import TrainConfig, train
from rl_experiments.utils.device_utils import get_device, get_device_str
from rl_experiments.utils.run_paths import make_run_id

console = Console()


PHASE1_ALGO_CHOICES = ("ppo", "sac", "dqn", "double_dqn", "per_dqn", "rainbow")


def parse_args():
    p = argparse.ArgumentParser(description="RL Experiments — Master Runner")
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run a single phase only")
    p.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=None,
        choices=[1, 2, 3, 4],
        help="Run only these phases (overrides --phase when set). Example: --phases 1 3",
    )
    p.add_argument("--quick", action="store_true", help="Quick run with reduced timesteps (smoke test)")
    p.add_argument("--device", action="store_true", help="Print detected device and exit")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds to use")
    p.add_argument("--env", type=str, default=None, help="Override environment (e.g. CartPole-v1)")
    p.add_argument(
        "--phase1-include",
        nargs="*",
        default=None,
        choices=list(PHASE1_ALGO_CHOICES),
        help="Subset of phase-1 baselines (default: all). Example: --phase1-include ppo sac dqn",
    )
    p.add_argument(
        "--algorithms",
        nargs="*",
        default=[],
        choices=[
            "dreamer",
            "muzero",
            "pets",
            "mbpo",
            "planet",
            "tdmpc",
            "tdmpc2",
            "world_models",
            "i2a",
            "mve",
            "steve",
        ],
        help="Explicit algorithms to run (mainly for phase 3)",
    )
    p.add_argument("--all-model-based", action="store_true", help="Run full model-based suite in phase 3")
    p.add_argument(
        "--strict-phase3",
        action="store_true",
        help="If set, do not default to dreamer+muzero when --algorithms is empty for phase 3",
    )
    return p.parse_args()


def resolve_phases(args: argparse.Namespace) -> list[int]:
    """Which phases to run (sorted unique)."""
    if getattr(args, "phases", None) is not None:
        return sorted(set(args.phases))
    if args.phase is None:
        return [1, 2, 3, 4]
    return [args.phase]


def run_phase1(
    seeds,
    run_id: str,
    quick: bool = False,
    phase1_include: set[str] | None = None,
):
    """Phase 1 — Foundation: PPO, SAC, DQN baselines."""
    ts_scale = 0.1 if quick else 1.0
    all_keys = set(PHASE1_ALGO_CHOICES)
    use = all_keys if phase1_include is None else (set(phase1_include) & all_keys)
    if not use:
        console.print("[yellow]Phase 1 — no baselines selected, skipping.[/yellow]")
        return

    console.rule("[bold blue]Phase 1 — Model-Free Baselines")

    for seed in seeds:
        if "ppo" in use:
            train(
                TrainConfig(
                    "ppo",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(200_000 * ts_scale),
                )
            )
            train(
                TrainConfig(
                    "ppo",
                    "LunarLander-v3",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(300_000 * ts_scale),
                )
            )
        if "sac" in use:
            train(
                TrainConfig(
                    "sac",
                    "Pendulum-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(100_000 * ts_scale),
                )
            )
            train(
                TrainConfig(
                    "sac",
                    "LunarLanderContinuous-v3",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(300_000 * ts_scale),
                )
            )
        if "dqn" in use:
            train(
                TrainConfig(
                    "dqn",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(200_000 * ts_scale),
                )
            )
            train(
                TrainConfig(
                    "dqn",
                    "LunarLander-v3",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(300_000 * ts_scale),
                )
            )
        if "double_dqn" in use:
            train(
                TrainConfig(
                    "double_dqn",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(200_000 * ts_scale),
                )
            )
        if "per_dqn" in use:
            train(
                TrainConfig(
                    "per_dqn",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(200_000 * ts_scale),
                )
            )
        if "rainbow" in use:
            train(
                TrainConfig(
                    "rainbow",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(200_000 * ts_scale),
                )
            )


def run_phase2(seeds):
    """Phase 2 — Behavioral Analysis: multi-seed comparison + plots."""
    from rl_experiments.experiments.compare_phase1 import main as compare_main

    console.rule("[bold yellow]Phase 2 — Behavioral Analysis")
    compare_main()


def run_phase3(
    seeds,
    run_id: str,
    quick: bool = False,
    algorithms: list[str] | None = None,
    all_model_based: bool = False,
    default_if_empty: bool = True,
):
    """Phase 3 — Advanced algorithms via unified `train()`."""
    n_ep_scale = 0.2 if quick else 1.0
    step_scale = 0.2 if quick else 1.0
    algorithms = algorithms or []
    selected = set(algorithms)
    if all_model_based:
        selected.update(
            {
                "dreamer",
                "muzero",
                "pets",
                "mbpo",
                "planet",
                "tdmpc",
                "tdmpc2",
                "world_models",
                "i2a",
                "mve",
                "steve",
            }
        )
    if not selected:
        if not default_if_empty:
            console.print("[yellow]Phase 3 — no algorithms selected, skipping.[/yellow]")
            return
        selected = {"dreamer", "muzero"}

    console.rule("[bold magenta]Phase 3 — Advanced Algorithms")

    for seed in seeds[:1]:
        if "dreamer" in selected:
            train(
                TrainConfig(
                    "dreamer",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_episodes=int(300 * n_ep_scale),
                )
            )
        if "muzero" in selected:
            train(
                TrainConfig(
                    "muzero",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_episodes=int(200 * n_ep_scale),
                )
            )
        if "pets" in selected:
            train(
                TrainConfig(
                    "pets",
                    "Pendulum-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(40_000 * step_scale),
                )
            )
        if "mbpo" in selected:
            train(
                TrainConfig(
                    "mbpo",
                    "Pendulum-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(30_000 * step_scale),
                )
            )
        if "planet" in selected:
            train(
                TrainConfig(
                    "planet",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_episodes=int(120 * n_ep_scale),
                )
            )
        if "tdmpc" in selected:
            train(
                TrainConfig(
                    "tdmpc",
                    "Pendulum-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(35_000 * step_scale),
                )
            )
        if "tdmpc2" in selected:
            train(
                TrainConfig(
                    "tdmpc2",
                    "Pendulum-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(35_000 * step_scale),
                )
            )
        if "world_models" in selected:
            train(
                TrainConfig(
                    "world_models",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_episodes=int(120 * n_ep_scale),
                )
            )
        if "i2a" in selected:
            train(
                TrainConfig(
                    "i2a",
                    "CartPole-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(50_000 * step_scale),
                )
            )
        if "mve" in selected:
            train(
                TrainConfig(
                    "mve",
                    "Pendulum-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(35_000 * step_scale),
                )
            )
        if "steve" in selected:
            train(
                TrainConfig(
                    "steve",
                    "Pendulum-v1",
                    seed,
                    run_id=run_id,
                    quick=quick,
                    budget_steps=int(35_000 * step_scale),
                )
            )


def write_run_metadata(run_id: str, args, start_unix: float, start_iso: str):
    results_dir = Path("results") / run_id
    logs_dir = Path("logs") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "started_at_iso": start_iso,
        "started_at_unix": start_unix,
        "command": " ".join(sys.argv),
        "args": vars(args),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
    }

    metadata_text = json.dumps(metadata, indent=2)
    (results_dir / "metadata.json").write_text(metadata_text + "\n", encoding="utf-8")
    (logs_dir / "metadata.json").write_text(metadata_text + "\n", encoding="utf-8")


def run_phase4():
    from rl_experiments.analysis.plot_figures import generate_all_plots

    console.rule("[bold cyan]Phase 4 — Paradigm Comparison Plots")
    generate_all_plots()


def print_experiment_plan(args):
    table = Table(
        title="[bold]RL Experiment Plan",
        header_style="bold cyan",
        border_style="blue",
    )
    table.add_column("Phase", style="yellow", justify="center")
    table.add_column("Algorithms", style="white", justify="left")
    table.add_column("Envs", style="cyan", justify="left")
    table.add_column("Mode", style="magenta", justify="center")

    mode = "[red]QUICK[/red]" if args.quick else "[green]FULL[/green]"

    table.add_row(
        "1 — Foundation",
        "PPO · SAC · DQN · DoubleDQN · PER-DQN · Rainbow",
        "CartPole / LunarLander / Pendulum",
        mode,
    )
    table.add_row("2 — Analysis", "PPO · SAC · DQN", "Multi-seed comparison", mode)
    table.add_row("3 — Advanced", "Dreamer · MuZero", "CartPole-v1", mode)
    table.add_row("4 — Comparison", "[all]", "Paradigm plots", "[green]AUTO[/green]")

    console.print(table)
    console.print(f"  [dim]Seeds: {args.seeds}")
    console.print(f"  [dim]Device: {get_device_str(verbose=False)}[/dim]")
    console.print()


def run_experiments(args: argparse.Namespace, *, skip_banner: bool = False) -> None:
    """Execute training from CLI or UI (`argparse.Namespace` with same fields as `parse_args`)."""
    import rl_experiments.api.registry  # noqa: F401

    run_start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(run_start))
    run_id = make_run_id()

    for d in ["logs", "results", "analysis/figures"]:
        os.makedirs(d, exist_ok=True)

    if args.device:
        get_device(verbose=True)
        return

    phases = resolve_phases(args)
    p1: set[str] | None = None
    if getattr(args, "phase1_include", None) is not None:
        p1 = set(args.phase1_include) if args.phase1_include else None

    if not skip_banner:
        console.print(
            Panel.fit(
                "[bold white]RL Experiments — Modern Reinforcement Learning Exploration\n"
                "[dim]PPO · SAC · DQN · Dreamer · MuZero[/dim]",
                border_style="blue",
            )
        )

    print_experiment_plan(args)
    console.print(f"  [dim]Run ID: {run_id}[/dim]  [dim]Phases: {phases}[/dim]")

    write_run_metadata(run_id, args, run_start, start_iso)
    t0 = run_start

    strict_p3 = getattr(args, "strict_phase3", False)

    if 1 in phases:
        run_phase1(args.seeds, run_id=run_id, quick=args.quick, phase1_include=p1)

    if 2 in phases:
        run_phase2(args.seeds)

    if 3 in phases:
        run_phase3(
            args.seeds,
            run_id=run_id,
            quick=args.quick,
            algorithms=list(args.algorithms) if args.algorithms else [],
            all_model_based=args.all_model_based,
            default_if_empty=not strict_p3,
        )

    if 4 in phases:
        run_phase4()

    elapsed = time.time() - t0
    console.print(f"\n[bold green]✓ All done in {elapsed/60:.1f} minutes.[/bold green]")
    console.print("  Plots → [cyan]analysis/figures/[/cyan]")
    console.print("  Logs  → [cyan]logs/[/cyan]")
    console.print("  Models→ [cyan]results/[/cyan]")


def main():
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
