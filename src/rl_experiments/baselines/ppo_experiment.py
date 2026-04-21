"""
baselines/ppo_experiment.py
─────────────────────────────────────────────────────────────────────────────
PPO — Proximal Policy Optimization (Schulman et al., 2017)
https://arxiv.org/abs/1707.06347

Architecture faithful to original paper:
  • Clipped surrogate objective:  L^CLIP = E[min(r_t·Â_t, clip(r_t,1-ε,1+ε)·Â_t)]
  • Value loss coefficient  c₁ = 0.5
  • Entropy bonus coefficient c₂ = 0.01 (encourages exploration)
  • GAE  λ = 0.95, γ = 0.99
  • K epochs per rollout = 10
  • Mini-batch size = 64
  • Gradient clipping max-norm = 0.5
  • Policy & value share the same MLP backbone (2 × 64 tanh) with separate heads.
  • No LSTM — tabular / low-dim observations only.

Mac GPU: Stable-Baselines3 accepts device="mps" for Apple Silicon.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from rich.console import Console
from rich.rule import Rule

from utils.device_utils import get_device_str
from utils.metrics import RLMetricsCallback
from utils.run_paths import build_log_path, build_model_path, make_run_id

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — match Schulman et al. 2017 Table 1 (MuJoCo defaults adapted
# for discrete envs; CartPole uses n_steps=2048, n_envs=1)
# ─────────────────────────────────────────────────────────────────────────────

PPO_CONFIG = {
    # ── Core PPO parameters ──────────────────────────────────────────────────
    "learning_rate":    3e-4,       # Adam step size
    "n_steps":          2048,       # Steps per rollout per env (T in the paper)
    "batch_size":       64,         # Mini-batch size for gradient updates
    "n_epochs":         10,         # K epochs of gradient updates per rollout
    "gamma":            0.99,       # Discount factor γ
    "gae_lambda":       0.95,       # GAE λ for advantage estimation
    "clip_range":       0.2,        # ε — clipping parameter
    "clip_range_vf":    None,       # No VF clipping (matches paper default)
    "ent_coef":         0.0,        # Entropy bonus c₂ (0 for classic envs)
    "vf_coef":          0.5,        # Value loss coefficient c₁
    "max_grad_norm":    0.5,        # Gradient norm clipping
    "normalize_advantage": True,    # Normalize advantages per mini-batch
    # ── Network ─────────────────────────────────────────────────────────────
    "policy_kwargs": {
        "net_arch": [64, 64],       # 2 hidden layers × 64 units (paper default)
        "activation_fn": __import__("torch").nn.Tanh,  # tanh (paper default)
    },
    "verbose":          1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ppo(env_id: str, total_timesteps: int = 200_000, seed: int = 42, run_id: str | None = None):
    """
    Train PPO on `env_id` and save metrics.

    Parameters
    ----------
    env_id          : Gymnasium environment ID (e.g. "CartPole-v1")
    total_timesteps : Training budget in environment steps
    seed            : Random seed for reproducibility
    """
    console.rule(f"[bold blue]PPO · {env_id}")
    device = get_device_str()

    # Vectorised env for PPO (SB3 PPO requires VecEnv)
    n_envs = 4  # paper uses multiple workers; we use 4 for speed
    if env_id.startswith("Pendulum"):
        n_envs = 1  # Pendulum with vec_normalize needs 1 env for simplicity

    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    run_id = run_id or make_run_id()
    # Metrics callback
    log_path = build_log_path("ppo", env_id, seed, run_id)
    cb = RLMetricsCallback(log_path=log_path)
    os.makedirs("logs/tensorboard/ppo", exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        seed=seed,
        tensorboard_log="logs/tensorboard/ppo",
        **PPO_CONFIG,
    )

    console.print(f"  [cyan]Device:[/cyan]     {device}")
    console.print(f"  [cyan]Env:[/cyan]        {env_id}  ×{n_envs} envs")
    console.print(f"  [cyan]Timesteps:[/cyan]  {total_timesteps:,}")
    console.print(f"  [cyan]Seed:[/cyan]       {seed}")
    console.print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=cb,
        progress_bar=True,
    )

    # Save trained policy
    save_path = build_model_path("ppo", env_id, seed, run_id)
    model.save(save_path)
    console.print(f"\n  [green]✓ Model saved → {save_path}.zip[/green]")
    console.print(f"  [green]✓ Metrics  → {log_path}[/green]")

    vec_env.close()
    return model, cb


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed runner for variance estimation (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def run_ppo_multiseed(env_id: str, seeds=(0, 1, 2), total_timesteps: int = 200_000):
    """Run PPO with multiple seeds to assess variance (Phase 2 analysis)."""
    results = []
    for seed in seeds:
        model, cb = run_ppo(env_id, total_timesteps=total_timesteps, seed=seed)
        results.append(cb)
        console.print(f"  Seed {seed} → final mean reward: "
                      f"[bold]{cb.rewards[-20:].mean():.1f}[/bold]  "
                      f"± {cb.rewards[-20:].std():.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Phase 1 environments
    console.print(Rule("[bold magenta]Phase 1 — PPO Experiments"))

    run_ppo("CartPole-v1",    total_timesteps=200_000)
    run_ppo("LunarLander-v3", total_timesteps=500_000)
