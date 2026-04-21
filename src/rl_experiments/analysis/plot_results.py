"""
Analysis plots — re-exports `generate_all_plots` and figure builders from `plot_figures`.
"""

from rl_experiments.analysis.plot_figures import (
    generate_all_plots,
    plot_cartpole_comparison,
    plot_dreamer_losses,
    plot_lunarlander_comparison,
    plot_muzero_training,
    plot_paradigm_comparison,
    plot_sample_efficiency,
    plot_variance_analysis,
)

__all__ = [
    "generate_all_plots",
    "plot_cartpole_comparison",
    "plot_dreamer_losses",
    "plot_lunarlander_comparison",
    "plot_muzero_training",
    "plot_paradigm_comparison",
    "plot_sample_efficiency",
    "plot_variance_analysis",
]
