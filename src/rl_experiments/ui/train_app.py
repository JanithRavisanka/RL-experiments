"""
Streamlit UI — pick phases and algorithms, then run the same pipeline as ``rl-experiments``.

Launch from the repo root::

    pip install -e ".[ui]"
    streamlit run src/rl_experiments/ui/train_app.py

Or use the ``rl-train-ui`` console script.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

# Ensure package import when launched by Streamlit from repo root
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    try:
        import rl_experiments  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(_SRC))

import streamlit as st

from rl_experiments.cli.run_all import PHASE1_ALGO_CHOICES, run_experiments

PHASE3_ALGO_CHOICES = [
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
]

PHASE1_HELP = {
    "ppo": "PPO — CartPole-v1 + LunarLander-v3",
    "sac": "SAC — Pendulum-v1 + LunarLanderContinuous-v3",
    "dqn": "DQN — CartPole-v1 + LunarLander-v3",
    "double_dqn": "Double DQN — CartPole-v1",
    "per_dqn": "PER-DQN — CartPole-v1",
    "rainbow": "Rainbow — CartPole-v1",
}


def _phase1_labels() -> dict[str, str]:
    return {k: f"{k} — {PHASE1_HELP[k]}" for k in PHASE1_ALGO_CHOICES}


def main():
    st.set_page_config(page_title="RL Experiments — Train", layout="wide")
    st.title("RL experiment runner")
    st.caption("Choose phases and algorithms, then start training (same behavior as the CLI).")

    phases = st.multiselect(
        "Phases to run",
        options=[1, 2, 3, 4],
        default=[3],
        help="1 = baselines · 2 = comparison script · 3 = advanced · 4 = plots only",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Phase 1 — model-free baselines")
        p1_options = list(PHASE1_ALGO_CHOICES)
        p1_sel = st.multiselect(
            "Baselines (subset)",
            options=p1_options,
            format_func=lambda x: _phase1_labels().get(x, x),
            default=p1_options,
            help="Leave all selected to run the full Phase 1 suite.",
        )

    with c2:
        st.subheader("Phase 3 — model-based / planning")
        p3_sel = st.multiselect(
            "Algorithms",
            options=PHASE3_ALGO_CHOICES,
            default=["dreamer", "muzero"],
        )
        all_mb = st.checkbox(
            "Include full model-based suite (adds all Phase 3 algorithms)",
            value=False,
        )

    quick = st.checkbox("Quick mode (smaller budgets — smoke test)", value=False)
    seeds_str = st.text_input("Seeds (comma-separated)", value="0, 1, 2")

    run = st.button("Start training", type="primary")

    if not run:
        return

    if not phases:
        st.error("Select at least one phase.")
        return

    try:
        seeds = [int(x.strip()) for x in seeds_str.replace(",", " ").split() if x.strip()]
    except ValueError:
        st.error("Invalid seeds — use integers separated by commas.")
        return

    if not seeds:
        st.error("Enter at least one seed.")
        return

    if 1 in phases and not p1_sel:
        st.error("Phase 1 is selected but no baselines are chosen — pick at least one or deselect phase 1.")
        return

    if 3 in phases and not all_mb and not p3_sel:
        st.error("Phase 3 is selected — pick one or more algorithms, or enable the full model-based suite.")
        return

    phase1_include = None
    if 1 in phases:
        phase1_include = None if len(p1_sel) == len(PHASE1_ALGO_CHOICES) else list(p1_sel)

    args = Namespace(
        phase=None,
        phases=phases,
        quick=quick,
        device=False,
        seeds=seeds,
        env=None,
        phase1_include=phase1_include,
        algorithms=list(p3_sel) if 3 in phases else [],
        all_model_based=all_mb,
        strict_phase3=True,
    )

    with st.spinner("Training… (progress also appears in this terminal if you launched Streamlit from a shell)"):
        try:
            run_experiments(args, skip_banner=True)
        except Exception as e:
            st.exception(e)
            return

    st.success("Finished. Check `logs/<run_id>/` and `results/<run_id>/`.")


# Streamlit executes this script as __main__; keep entry at module level.
main()
