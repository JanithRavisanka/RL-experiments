"""
experiments/compare_phase1.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 — Behavioral Analysis

Runs PPO, SAC, and DQN on matching environments, then generates side-by-side
comparison plots to answer the Phase 2 questions:

  Q1. Why is PPO more stable than DQN?
  Q2. Why does SAC handle noise better?
  Q3. Why does DQN struggle with variance?

This script:
  1. Runs all 3 algorithms on CartPole-v1 (PPO + DQN only — discrete)
  2. Runs PPO + SAC on LunarLander-v3 / LunarLanderContinuous-v3
  3. Multi-seed runs (seeds 0,1,2) for variance estimation
  4. Saves comparison plots to analysis/

Mac GPU: inherits device from utils/device_utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from rl_experiments.utils.metrics import smooth
from rl_experiments.baselines.ppo_experiment import run_ppo_multiseed
from rl_experiments.baselines.sac_experiment import run_sac_multiseed
from rl_experiments.baselines.dqn_experiment import run_dqn_multiseed

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEEDS        = [0, 1, 2]
TIMESTEPS    = {
    "CartPole-v1":                  200_000,
    "LunarLander-v3":               300_000,
    "LunarLanderContinuous-v3":     300_000,
    "Pendulum-v1":                  100_000,
}
COLORS = {
    "PPO": "#4FC3F7",   # light blue
    "SAC": "#81C784",   # light green
    "DQN": "#FF8A65",   # light orange
}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_mean_std(ax, callbacks, label, color, n_points=200):
    """
    Plot mean ± std band across seeds.

    Each callback has `.timesteps` and `.rewards` arrays of different length
    → interpolate all onto a common x-axis.
    """
    all_ts    = np.linspace(0, max(cb.timesteps[-1] for cb in callbacks), n_points)
    all_rews  = []
    for cb in callbacks:
        if len(cb.rewards) < 2:
            continue
        interp = np.interp(all_ts, cb.timesteps, smooth(cb.rewards, window=15))
        all_rews.append(interp)

    if not all_rews:
        return

    arr  = np.array(all_rews)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    ax.plot(all_ts, mean, label=label, color=color, linewidth=2)
    ax.fill_between(all_ts, mean - std, mean + std, alpha=0.25, color=color)


def plot_comparison(results: dict, env_id: str, save_dir: str = "analysis"):
    """
    results: {"PPO": [cb0,cb1,...], "SAC": [...], "DQN": [...]}
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for algo, callbacks in results.items():
        if callbacks:
            _plot_mean_std(ax, callbacks, algo, COLORS[algo])

    ax.set_title(f"PPO vs SAC vs DQN — {env_id}\n(mean ± std, {len(SEEDS)} seeds)",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel("Environment Steps", color="white")
    ax.set_ylabel("Episode Reward",    color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=11)
    ax.grid(alpha=0.2, color="white")

    fname = f"{save_dir}/compare_{env_id.replace('-','_').lower()}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"  [green]✓ Saved → {fname}[/green]")


def print_summary_table(results: dict):
    table = Table(title="Final Performance Summary (mean last 20 episodes)",
                  style="bold", header_style="bold cyan")
    table.add_column("Algorithm",    style="yellow",  justify="left")
    table.add_column("Env",          style="white",   justify="left")
    table.add_column("Mean Reward",  style="green",   justify="right")
    table.add_column("Std Reward",   style="magenta", justify="right")
    table.add_column("# Episodes",   style="cyan",    justify="right")

    for (algo, env_id), callbacks in sorted(results.items()):
        for cb in callbacks:
            if len(cb.rewards) > 0:
                fin = cb.rewards[-20:]
                table.add_row(
                    algo, env_id,
                    f"{fin.mean():.1f}",
                    f"{fin.std():.1f}",
                    str(len(cb.rewards)),
                )

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("logs",     exist_ok=True)
    os.makedirs("results",  exist_ok=True)
    os.makedirs("analysis", exist_ok=True)

    all_results = {}

    # ── Experiment A: CartPole-v1 (discrete → PPO + DQN) ────────────────────
    console.print(Rule("[bold magenta]Experiment A — CartPole-v1 (PPO vs DQN)"))
    ppo_cp  = run_ppo_multiseed("CartPole-v1",    SEEDS, TIMESTEPS["CartPole-v1"])
    dqn_cp  = run_dqn_multiseed("CartPole-v1",    SEEDS, TIMESTEPS["CartPole-v1"])

    plot_comparison(
        {"PPO": ppo_cp, "DQN": dqn_cp, "SAC": []},
        "CartPole-v1",
    )
    all_results[("PPO", "CartPole-v1")] = ppo_cp
    all_results[("DQN", "CartPole-v1")] = dqn_cp

    # ── Experiment B: LunarLander variants (PPO discrete, SAC+PPO continuous) ─
    console.print(Rule("[bold magenta]Experiment B — LunarLander (PPO vs DQN)"))
    ppo_ll  = run_ppo_multiseed("LunarLander-v3", SEEDS, TIMESTEPS["LunarLander-v3"])
    dqn_ll  = run_dqn_multiseed("LunarLander-v3", SEEDS, TIMESTEPS["LunarLander-v3"])

    plot_comparison(
        {"PPO": ppo_ll, "DQN": dqn_ll, "SAC": []},
        "LunarLander-v3",
    )
    all_results[("PPO", "LunarLander-v3")] = ppo_ll
    all_results[("DQN", "LunarLander-v3")] = dqn_ll

    # ── Experiment C: Continuous LunarLander (PPO vs SAC) ───────────────────
    console.print(Rule("[bold magenta]Experiment C — LunarLanderContinuous (PPO vs SAC)"))
    ppo_llc = run_ppo_multiseed("LunarLanderContinuous-v3", SEEDS, TIMESTEPS["LunarLanderContinuous-v3"])
    sac_llc = run_sac_multiseed("LunarLanderContinuous-v3", SEEDS, TIMESTEPS["LunarLanderContinuous-v3"])

    plot_comparison(
        {"PPO": ppo_llc, "SAC": sac_llc, "DQN": []},
        "LunarLanderContinuous-v3",
    )
    all_results[("PPO", "LunarLanderContinuous-v3")] = ppo_llc
    all_results[("SAC", "LunarLanderContinuous-v3")] = sac_llc

    # ── Summary table ────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Results Summary"))
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
