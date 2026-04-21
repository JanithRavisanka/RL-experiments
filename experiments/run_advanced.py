"""
experiments/run_advanced.py
─────────────────────────────────────────────────────────────────────────────
Phase 3 — Advanced Algorithm Experiments

Runs:
  1. Dreamer on CartPole-v1   — model-based RL with latent world model
  2. MuZero  on CartPole-v1   — planning-based RL with MCTS

Then generates Phase 4 comparison plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from rich.console import Console
from rich.rule import Rule

from advanced.dreamer.dreamer_agent import DreamerAgent
from advanced.muzero.muzero_agent   import MuZeroAgent
from analysis.plot_results import generate_all_plots

console = Console()


def run_dreamer(env_id: str = "CartPole-v1", n_episodes: int = 300, seed: int = 0):
    console.rule(f"[bold magenta]Phase 3 — Dreamer · {env_id}")
    agent   = DreamerAgent(env_id, seed=seed)
    rewards = agent.train(
        n_episodes=n_episodes,
        log_path=f"logs/dreamer_{env_id.replace('-','_').lower()}_seed{seed}.csv",
    )
    final = np.mean(rewards[-20:])
    console.print(f"  [green]Dreamer final mean reward (last 20 eps): {final:.1f}[/green]")
    return rewards


def run_muzero(env_id: str = "CartPole-v1", n_episodes: int = 300, seed: int = 0):
    console.rule(f"[bold magenta]Phase 3 — MuZero · {env_id}")
    agent   = MuZeroAgent(env_id, seed=seed)
    rewards = agent.train(
        n_episodes=n_episodes,
        log_path=f"logs/muzero_{env_id.replace('-','_').lower()}_seed{seed}.csv",
    )
    final = np.mean(rewards[-20:])
    console.print(f"  [green]MuZero final mean reward (last 20 eps): {final:.1f}[/green]")
    return rewards


if __name__ == "__main__":
    os.makedirs("logs",    exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Phase 3 — Advanced algorithms
    dreamer_rewards = run_dreamer("CartPole-v1", n_episodes=300)
    muzero_rewards  = run_muzero("CartPole-v1",  n_episodes=200)

    # Phase 4 — Generate all comparison plots
    console.rule("[bold cyan]Phase 4 — Generating Analysis Plots")
    generate_all_plots()
