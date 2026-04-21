"""
analysis/plot_results.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive plotting module for all RL experiments.

Produces publication-quality dark-themed plots for:
  Phase 2 — PPO / SAC / DQN comparison curves
  Phase 3 — Dreamer world model losses + reward curves
  Phase 4 — Model-free vs Model-based vs Planning paradigms

All plots are saved to analysis/figures/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional

from utils.metrics import smooth

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "PPO":     "#4FC3F7",   # sky blue
    "SAC":     "#81C784",   # soft green
    "DQN":     "#FF8A65",   # coral
    "Dreamer": "#CE93D8",   # lavender
    "MuZero":  "#FFD54F",   # amber
}

DARK_BG    = "#1a1a2e"
PANEL_BG   = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply consistent dark-theme styling to an Axes."""
    ax.set_facecolor(PANEL_BG)
    ax.title.set_text(title)
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontsize(12)
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)


def _save(fig, filename: str, figures_dir: str = "analysis/figures"):
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    path = f"{figures_dir}/{filename}"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Load CSV log files
# ─────────────────────────────────────────────────────────────────────────────

def load_sb3_logs(pattern: str, log_dir: str = "logs") -> List[pd.DataFrame]:
    """
    Load one or more CSV files matching `pattern` glob.
    Each CSV has columns: timestep, reward, ep_len, wall_time
    """
    files = sorted(Path(log_dir).glob(pattern))
    dfs   = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
    return dfs


def load_custom_logs(pattern: str, log_dir: str = "logs") -> List[pd.DataFrame]:
    """Load custom algorithm CSVs (Dreamer, MuZero)."""
    files = sorted(Path(log_dir).glob(pattern))
    dfs   = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# Plot helper: mean ± std band
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_std_band(
    ax,
    dfs:       List[pd.DataFrame],
    x_col:     str,
    y_col:     str,
    label:     str,
    color:     str,
    smooth_w:  int = 15,
    n_points:  int = 300,
    alpha:     float = 0.25,
):
    """Plot a mean curve ± std shaded band across multiple seed DataFrames."""
    if not dfs:
        return

    x_max = max(df[x_col].iloc[-1] for df in dfs)
    xs    = np.linspace(0, x_max, n_points)
    curves = []
    for df in dfs:
        ys = smooth(df[y_col].values, window=smooth_w)
        curves.append(np.interp(xs, df[x_col].values, ys))

    arr  = np.array(curves)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    ax.plot(xs, mean, label=label, color=color, linewidth=2.0)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Phase 2: CartPole comparison (PPO vs DQN)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cartpole_comparison(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    ppo_dfs = load_sb3_logs("ppo_cartpole_v1_*.csv", log_dir)
    dqn_dfs = load_sb3_logs("dqn_cartpole_v1_*.csv", log_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    plot_mean_std_band(ax, ppo_dfs, "timestep", "reward", "PPO", ALGO_COLORS["PPO"])
    plot_mean_std_band(ax, dqn_dfs, "timestep", "reward", "DQN", ALGO_COLORS["DQN"])

    _style_ax(ax,
        title  = "CartPole-v1 — PPO vs DQN  (mean ± std, 3 seeds)",
        xlabel = "Environment Steps",
        ylabel = "Episode Reward",
    )
    ax.axhline(y=500, color="white", linestyle="--", alpha=0.3, linewidth=1, label="Max (500)")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=10)

    # Annotations
    ax.annotate(
        "PPO: stable, monotone\nimprovement (clipped obj.)",
        xy=(0.55, 0.35), xycoords="axes fraction",
        color=ALGO_COLORS["PPO"], fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=ALGO_COLORS["PPO"], alpha=0.8),
    )
    ax.annotate(
        "DQN: high variance early,\nε-greedy exploration",
        xy=(0.15, 0.15), xycoords="axes fraction",
        color=ALGO_COLORS["DQN"], fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=ALGO_COLORS["DQN"], alpha=0.8),
    )

    return _save(fig, "phase2_cartpole_ppo_dqn.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Phase 2: LunarLander continuous (PPO vs SAC)
# ─────────────────────────────────────────────────────────────────────────────

def plot_lunarlander_comparison(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    ppo_dfs = load_sb3_logs("ppo_lunarlandercontinuous_v3_*.csv", log_dir)
    sac_dfs = load_sb3_logs("sac_lunarlandercontinuous_v3_*.csv", log_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    plot_mean_std_band(ax, ppo_dfs, "timestep", "reward", "PPO", ALGO_COLORS["PPO"])
    plot_mean_std_band(ax, sac_dfs, "timestep", "reward", "SAC", ALGO_COLORS["SAC"])

    _style_ax(ax,
        title  = "LunarLanderContinuous-v3 — PPO vs SAC  (mean ± std, 3 seeds)",
        xlabel = "Environment Steps",
        ylabel = "Episode Reward",
    )
    ax.axhline(y=200, color="white", linestyle="--", alpha=0.3, linewidth=1, label="Solved (200)")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=10)

    return _save(fig, "phase2_lunarlander_ppo_sac.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase 3: Dreamer world model losses
# ─────────────────────────────────────────────────────────────────────────────

def plot_dreamer_losses(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    dfs = load_custom_logs("dreamer_*.csv", log_dir)
    if not dfs:
        print(f"  No Dreamer logs found in {log_dir}. Run Dreamer first.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Dreamer — World Model Training Losses", color=TEXT_COLOR, fontsize=13)

    for df in dfs:
        eps = df["episode"].values
        for ax, col, label, color in [
            (axes[0], "wm_loss", "Reconstruction Loss",   "#4FC3F7"),
            (axes[1], "kl",      "KL Divergence",         "#CE93D8"),
            (axes[2], "reward",  "Episode Reward",        "#81C784"),
        ]:
            if col in df.columns:
                ax.plot(eps, smooth(df[col].values, 8), color=color, linewidth=1.5, alpha=0.9)

    _style_ax(axes[0], title="Reconstruction",   xlabel="Episode", ylabel="MSE")
    _style_ax(axes[1], title="KL (free bits=1)", xlabel="Episode", ylabel="KL")
    _style_ax(axes[2], title="Episode Reward",   xlabel="Episode", ylabel="Reward")

    return _save(fig, "phase3_dreamer_losses.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Phase 3: MuZero training loss + reward
# ─────────────────────────────────────────────────────────────────────────────

def plot_muzero_training(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    dfs = load_custom_logs("muzero_*.csv", log_dir)
    if not dfs:
        print(f"  No MuZero logs found in {log_dir}. Run MuZero first.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("MuZero — Self-Play Training", color=TEXT_COLOR, fontsize=13)

    for df in dfs:
        eps = df["episode"].values
        if "reward" in df.columns:
            ax1.plot(eps, smooth(df["reward"].values, 10), color=ALGO_COLORS["MuZero"], linewidth=1.5)
        if "loss" in df.columns:
            ax2.plot(eps, smooth(df["loss"].values, 10), color="#FF8A65", linewidth=1.5)

    _style_ax(ax1, title="Episode Reward",  xlabel="Episode", ylabel="Reward")
    _style_ax(ax2, title="Training Loss",   xlabel="Episode", ylabel="Loss")

    return _save(fig, "phase3_muzero_training.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Phase 4: Grand Paradigm Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_paradigm_comparison(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    """
    All 5 algorithms on the same CartPole-v1 reward curve.
    Answers Phase 4: Model-Free vs Model-Based vs Planning.
    """
    logs = {
        "PPO":     load_sb3_logs("ppo_cartpole_v1_*.csv",    log_dir),
        "DQN":     load_sb3_logs("dqn_cartpole_v1_*.csv",    log_dir),
        "Dreamer": load_custom_logs("dreamer_cartpole_v1*.csv", log_dir),
        "MuZero":  load_custom_logs("muzero_cartpole_v1*.csv",  log_dir),
    }

    fig = plt.figure(figsize=(14, 6), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
    ax_curve = fig.add_subplot(gs[0])
    ax_bar   = fig.add_subplot(gs[1])

    ax_curve.set_facecolor(PANEL_BG)
    ax_bar.set_facecolor(PANEL_BG)

    final_means = {}

    for algo, dfs in logs.items():
        if not dfs:
            continue
        color = ALGO_COLORS[algo]

        # Choose x/y columns based on format
        if algo in ("PPO", "DQN"):
            x_col, y_col = "timestep", "reward"
        else:
            x_col = "episode" if "episode" in dfs[0].columns else "total_steps"
            y_col = "reward"

        plot_mean_std_band(ax_curve, dfs, x_col, y_col, algo, color)

        # Collect final mean reward for bar chart
        last_vals = []
        for df in dfs:
            last_vals.extend(df[y_col].values[-20:].tolist())
        if last_vals:
            final_means[algo] = (np.mean(last_vals), np.std(last_vals))

    # Curve plot
    _style_ax(ax_curve,
        title  = "CartPole-v1 — All Paradigms",
        xlabel = "Steps / Episodes",
        ylabel = "Episode Reward",
    )
    ax_curve.axhline(y=500, color="white", linestyle="--", alpha=0.25, linewidth=1)
    ax_curve.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=10)

    # Bar chart — final reward
    if final_means:
        algos  = list(final_means.keys())
        means  = [final_means[a][0] for a in algos]
        stds   = [final_means[a][1] for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]
        bars = ax_bar.bar(algos, means, yerr=stds, color=colors,
                          capsize=5, error_kw={"ecolor": "white", "elinewidth": 1.5})
        _style_ax(ax_bar, title="Final Reward\n(last 20 episodes)",
                  xlabel="Algorithm", ylabel="Mean Reward")
        ax_bar.tick_params(axis="x", colors=TEXT_COLOR, labelsize=9)

    # Paradigm labels (text annotation on right)
    paradigm_labels = [
        (0.02, 0.97, "🔵 Model-Free",    "#4FC3F7"),
        (0.02, 0.91, "🟣 Model-Based",   "#CE93D8"),
        (0.02, 0.85, "🟡 Planning-Based","#FFD54F"),
    ]
    for x, y, txt, col in paradigm_labels:
        ax_curve.text(x, y, txt, transform=ax_curve.transAxes,
                      color=col, fontsize=9, va="top")

    fig.suptitle("Phase 4 — RL Paradigm Comparison: CartPole-v1",
                 color=TEXT_COLOR, fontsize=14, y=1.01)

    return _save(fig, "phase4_paradigm_comparison.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Sample Efficiency Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_efficiency(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    """
    Plot 'steps to reach reward threshold' for each algorithm.
    Key Phase 4 metric: how many env interactions does each paradigm need?
    """
    THRESHOLD = 400     # CartPole: threshold for "good" performance

    data = {
        "PPO":     ("ppo_cartpole_v1_*.csv",    "timestep", "reward"),
        "DQN":     ("dqn_cartpole_v1_*.csv",    "timestep", "reward"),
        "Dreamer": ("dreamer_cartpole_v1*.csv",  "total_steps", "reward"),
        "MuZero":  ("muzero_cartpole_v1*.csv",  "total_steps", "reward"),
    }

    steps_to_reach = {}
    for algo, (pattern, x_col, y_col) in data.items():
        loader = load_sb3_logs if algo in ("PPO", "DQN") else load_custom_logs
        dfs    = loader(pattern, log_dir)
        seed_steps = []
        for df in dfs:
            if x_col not in df.columns or y_col not in df.columns:
                continue
            sm   = smooth(df[y_col].values, 10)
            idxs = np.where(sm >= THRESHOLD)[0]
            if len(idxs) > 0:
                seed_steps.append(df[x_col].values[idxs[0]])
            else:
                seed_steps.append(df[x_col].values[-1])  # didn't reach threshold
        if seed_steps:
            steps_to_reach[algo] = (np.mean(seed_steps), np.std(seed_steps))

    if not steps_to_reach:
        print("  No data available for sample efficiency plot. Run experiments first.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    algos  = list(steps_to_reach.keys())
    means  = [steps_to_reach[a][0] for a in algos]
    stds   = [steps_to_reach[a][1] for a in algos]
    colors = [ALGO_COLORS[a] for a in algos]

    bars = ax.barh(algos, means, xerr=stds, color=colors, capsize=5,
                   error_kw={"ecolor": "white", "elinewidth": 1.5}, height=0.5)

    ax.set_xlabel(f"Steps to Reach Reward ≥ {THRESHOLD}", color=TEXT_COLOR)
    ax.set_title(f"Sample Efficiency — CartPole-v1\n(Steps to reach reward {THRESHOLD})",
                 color=TEXT_COLOR, fontsize=12)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR, axis="x")

    # Value labels
    for bar, mean in zip(bars, means):
        ax.text(mean * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{mean:,.0f}", va="center", color="white", fontsize=9)

    return _save(fig, "phase4_sample_efficiency.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Variance Analysis (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_analysis(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    """
    Phase 2: Plot the variance of each algorithm across seeds.
    Shows why PPO is more stable than DQN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Phase 2 — Variance Across Seeds (3 seeds)",
                 color=TEXT_COLOR, fontsize=13)

    pairs = [
        (axes[0], "CartPole-v1",
         [("PPO", "ppo_cartpole_v1_*.csv",  "timestep", "reward"),
          ("DQN", "dqn_cartpole_v1_*.csv",  "timestep", "reward")]),
        (axes[1], "LunarLanderContinuous-v3",
         [("PPO", "ppo_lunarlandercontinuous_v3_*.csv", "timestep", "reward"),
          ("SAC", "sac_lunarlandercontinuous_v3_*.csv", "timestep", "reward")]),
    ]

    for ax, title, algo_list in pairs:
        ax.set_facecolor(PANEL_BG)
        for algo, pattern, xcol, ycol in algo_list:
            loader = load_sb3_logs
            dfs    = loader(pattern, log_dir)
            if dfs:
                plot_mean_std_band(ax, dfs, xcol, ycol, algo, ALGO_COLORS[algo])
        _style_ax(ax, title=title, xlabel="Steps", ylabel="Reward")
        ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COLOR, fontsize=9)

    return _save(fig, "phase2_variance_analysis.png", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Run all plots at once
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(log_dir: str = "logs", out_dir: str = "analysis/figures"):
    """Generate all analysis figures. Call after running experiments."""
    print("\n=== Generating Analysis Plots ===\n")
    plot_cartpole_comparison(log_dir, out_dir)
    plot_lunarlander_comparison(log_dir, out_dir)
    plot_dreamer_losses(log_dir, out_dir)
    plot_muzero_training(log_dir, out_dir)
    plot_paradigm_comparison(log_dir, out_dir)
    plot_sample_efficiency(log_dir, out_dir)
    plot_variance_analysis(log_dir, out_dir)
    print(f"\n✓ All plots saved to {out_dir}/\n")


if __name__ == "__main__":
    generate_all_plots()
