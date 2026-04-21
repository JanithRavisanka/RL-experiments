"""Dark theme and helpers for analysis figures."""

from pathlib import Path

import matplotlib.pyplot as plt

ALGO_COLORS = {
    "PPO": "#4FC3F7",
    "SAC": "#81C784",
    "DQN": "#FF8A65",
    "DoubleDQN": "#64B5F6",
    "PER-DQN": "#BA68C8",
    "Rainbow": "#FFCA28",
    "PETS": "#26A69A",
    "MBPO": "#7E57C2",
    "PlaNet": "#8D6E63",
    "TD-MPC": "#42A5F5",
    "TD-MPC2": "#1E88E5",
    "WorldModels": "#AB47BC",
    "I2A": "#66BB6A",
    "MVE": "#FFA726",
    "STEVE": "#EF5350",
    "Dreamer": "#CE93D8",
    "MuZero": "#FFD54F",
}

DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
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
