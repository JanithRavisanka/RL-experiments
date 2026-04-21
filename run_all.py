#!/usr/bin/env python3
"""
run_all.py
─────────────────────────────────────────────────────────────────────────────
Master runner for all RL experiments.

Usage:
  python run_all.py              # Run everything (slow)
  python run_all.py --phase 1    # Phase 1 only (PPO, SAC, DQN baselines)
  python run_all.py --phase 2    # Phase 2 only (multi-seed comparison)
  python run_all.py --phase 3    # Phase 3 only (Dreamer + MuZero)
  python run_all.py --phase 4    # Phase 4 only (generate plots)
  python run_all.py --quick      # Quick smoke-test (small step budgets)
  python run_all.py --device     # Show detected Mac GPU device and exit
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.panel import Panel

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from utils.device_utils import get_device, get_device_str
from utils.run_paths import make_run_id

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RL Experiments — Master Runner")
    p.add_argument("--phase",  type=int, choices=[1, 2, 3, 4],
                   help="Run a specific phase only")
    p.add_argument("--quick",  action="store_true",
                   help="Quick run with reduced timesteps (smoke test)")
    p.add_argument("--device", action="store_true",
                   help="Print detected device and exit")
    p.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds to use")
    p.add_argument("--env",    type=str, default=None,
                   help="Override environment (e.g. CartPole-v1)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Phase runners
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(seeds, quick: bool = False):
    """Phase 1 — Foundation: PPO, SAC, DQN baselines."""
    from baselines.ppo_experiment import run_ppo
    from baselines.sac_experiment import run_sac
    from baselines.dqn_experiment import run_dqn

    ts_scale = 0.1 if quick else 1.0
    console.rule("[bold blue]Phase 1 — Model-Free Baselines")

    for seed in seeds:
        # PPO
        run_ppo("CartPole-v1",              int(200_000 * ts_scale), seed, run_id=make_run_id())
        run_ppo("LunarLander-v3",           int(300_000 * ts_scale), seed, run_id=make_run_id())

        # SAC
        run_sac("Pendulum-v1",              int(100_000 * ts_scale), seed, run_id=make_run_id())
        run_sac("LunarLanderContinuous-v3", int(300_000 * ts_scale), seed, run_id=make_run_id())

        # DQN
        run_dqn("CartPole-v1",              int(200_000 * ts_scale), seed, run_id=make_run_id())
        run_dqn("LunarLander-v3",           int(300_000 * ts_scale), seed, run_id=make_run_id())


def run_phase2(seeds):
    """Phase 2 — Behavioral Analysis: multi-seed comparison + plots."""
    from experiments.compare_phase1 import main as compare_main
    console.rule("[bold yellow]Phase 2 — Behavioral Analysis")
    compare_main()


def run_phase3(seeds, quick: bool = False):
    """Phase 3 — Advanced: Dreamer + MuZero."""
    from advanced.dreamer.dreamer_agent import DreamerAgent
    from advanced.muzero.muzero_agent   import MuZeroAgent

    n_ep_scale = 0.2 if quick else 1.0
    console.rule("[bold magenta]Phase 3 — Advanced Algorithms")

    for seed in seeds[:1]:  # Advanced algos: run 1 seed (expensive)
        # Dreamer
        agent = DreamerAgent("CartPole-v1", seed=seed)
        agent.train(
            n_episodes=int(300 * n_ep_scale),
            run_id=make_run_id(),
        )

        # MuZero
        agent = MuZeroAgent("CartPole-v1", seed=seed)
        agent.train(
            n_episodes=int(200 * n_ep_scale),
            run_id=make_run_id(),
        )


def run_phase4():
    """Phase 4 — Generate all comparison plots."""
    from analysis.plot_results import generate_all_plots
    console.rule("[bold cyan]Phase 4 — Paradigm Comparison Plots")
    generate_all_plots()


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_plan(args):
    table = Table(title="[bold]RL Experiment Plan", header_style="bold cyan",
                  border_style="blue")
    table.add_column("Phase",     style="yellow",  justify="center")
    table.add_column("Algorithms",style="white",   justify="left")
    table.add_column("Envs",      style="cyan",    justify="left")
    table.add_column("Mode",      style="magenta", justify="center")

    mode = "[red]QUICK[/red]" if args.quick else "[green]FULL[/green]"

    table.add_row("1 — Foundation",    "PPO · SAC · DQN",
                  "CartPole / LunarLander / Pendulum", mode)
    table.add_row("2 — Analysis",      "PPO · SAC · DQN",
                  "Multi-seed comparison",             mode)
    table.add_row("3 — Advanced",      "Dreamer · MuZero",
                  "CartPole-v1",                       mode)
    table.add_row("4 — Comparison",    "[all]",
                  "Paradigm plots",                    "[green]AUTO[/green]")

    console.print(table)
    console.print(f"  [dim]Seeds: {args.seeds}")
    console.print(f"  [dim]Device: {get_device_str(verbose=False)}[/dim]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Create output directories
    for d in ["logs", "results", "analysis/figures"]:
        os.makedirs(d, exist_ok=True)

    # Device-only mode
    if args.device:
        get_device(verbose=True)
        return

    console.print(Panel.fit(
        "[bold white]🧠 RL Experiments — Modern Reinforcement Learning Exploration\n"
        "[dim]PPO · SAC · DQN · Dreamer · MuZero[/dim]",
        border_style="blue",
    ))

    print_experiment_plan(args)

    t0 = time.time()

    if args.phase is None or args.phase == 1:
        run_phase1(args.seeds, args.quick)

    if args.phase is None or args.phase == 2:
        run_phase2(args.seeds)

    if args.phase is None or args.phase == 3:
        run_phase3(args.seeds, args.quick)

    if args.phase is None or args.phase == 4:
        run_phase4()

    elapsed = time.time() - t0
    console.print(f"\n[bold green]✓ All done in {elapsed/60:.1f} minutes.[/bold green]")
    console.print("  Plots → [cyan]analysis/figures/[/cyan]")
    console.print("  Logs  → [cyan]logs/[/cyan]")
    console.print("  Models→ [cyan]results/[/cyan]")


if __name__ == "__main__":
    main()
