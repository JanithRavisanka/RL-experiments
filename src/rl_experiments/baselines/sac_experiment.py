"""
baselines/sac_experiment.py
─────────────────────────────────────────────────────────────────────────────
SAC — Soft Actor-Critic (Haarnoja et al., 2018)
https://arxiv.org/abs/1801.01290
https://arxiv.org/abs/1812.05905  (version 2 with auto-alpha)

Architecture faithful to the paper:
  • Squashed Gaussian policy: a_t = tanh(μ + σ·ε),  ε ~ N(0,I)
    → reparameterisation trick for low-variance gradients
  • Twin Q-networks (Q_θ1, Q_θ2) to reduce over-estimation bias
    TD target: y = r + γ · (min_i Q_θ̄i(s',ã') − α log π(ã'|s'))
  • Target networks with Polyak averaging: θ̄ ← τ·θ + (1−τ)·θ̄, τ=0.005
  • Automatic entropy temperature α tuned by dual gradient descent:
    ∂ℒ/∂α = E[−α(log π(a|s) + H̄)],  H̄ = −dim(A)
  • Replay buffer: 1,000,000 transitions
  • Networks: 2 hidden layers × 256 units, ReLU (Haarnoja et al. 2018 Table 1)
  • Batch size: 256
  • Learning rates: 3e-4 for all networks
  • Updates per step: 1 (gradient update after each env step)

SAC is designed for CONTINUOUS action spaces.
  → CartPole is DISCRETE → NOT suitable. Use Pendulum-v1 / LunarLanderContinuous-v2.

Mac GPU: device="mps" for Apple Silicon.
"""

import os

import gymnasium as gym
from stable_baselines3 import SAC
from rich.console import Console
from rich.rule import Rule

from rl_experiments.utils.device_utils import get_device_str
from rl_experiments.utils.metrics import RLMetricsCallback
from rl_experiments.utils.run_paths import build_log_path, build_model_path, make_run_id

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — Haarnoja et al. 2018 Table 1
# ─────────────────────────────────────────────────────────────────────────────

SAC_CONFIG = {
    # ── Core SAC parameters ──────────────────────────────────────────────────
    "learning_rate":            3e-4,       # Shared for policy, Q1, Q2, α
    "buffer_size":              1_000_000,  # Replay buffer capacity
    "learning_starts":          1_000,      # Fill buffer before updates start
    "batch_size":               256,        # Mini-batch for gradient updates
    "tau":                      0.005,      # Polyak averaging coefficient τ
    "gamma":                    0.99,       # Discount factor γ
    "train_freq":               1,          # Update every env step
    "gradient_steps":           1,          # Gradient updates per env step
    "ent_coef":                 "auto",     # Auto-tune entropy coefficient α
    "target_update_interval":   1,          # Target net updated every step
    "target_entropy":           "auto",     # H̄ = −dim(A) (paper default)
    "use_sde":                  False,      # State-Dependent Exploration (off)
    "optimize_memory_usage":    False,
    # ── Network ─────────────────────────────────────────────────────────────
    "policy_kwargs": {
        "net_arch":        [256, 256],      # 2 hidden × 256, ReLU (paper Table 1)
        "activation_fn":   __import__("torch").nn.ReLU,
    },
    "verbose":                  1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sac(env_id: str, total_timesteps: int = 300_000, seed: int = 42, run_id: str | None = None):
    """
    Train SAC on a continuous-action `env_id`.

    Parameters
    ----------
    env_id          : Gymnasium environment ID
                      (must have continuous action space: Box)
    total_timesteps : Training budget in env steps
    seed            : Random seed
    """
    console.rule(f"[bold blue]SAC · {env_id}")
    device = get_device_str()

    env = gym.make(env_id)
    action_space = env.action_space
    env.close()

    # Safety guard — SAC needs continuous actions
    import gymnasium.spaces as spaces
    if not isinstance(action_space, spaces.Box):
        console.print(f"[red]✗ SAC requires a continuous (Box) action space. "
                      f"'{env_id}' has {type(action_space).__name__}. Skipping.[/red]")
        return None, None

    run_id = run_id or make_run_id()
    log_path = build_log_path("sac", env_id, seed, run_id)
    cb = RLMetricsCallback(log_path=log_path)
    os.makedirs("logs/tensorboard/sac", exist_ok=True)

    model = SAC(
        policy="MlpPolicy",
        env=env_id,
        device=device,
        seed=seed,
        tensorboard_log="logs/tensorboard/sac",
        **SAC_CONFIG,
    )

    console.print(f"  [cyan]Device:[/cyan]     {device}")
    console.print(f"  [cyan]Env:[/cyan]        {env_id}")
    console.print(f"  [cyan]Action space:[/cyan] {action_space}")
    console.print(f"  [cyan]Timesteps:[/cyan]  {total_timesteps:,}")
    console.print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=cb,
        progress_bar=True,
    )

    save_path = build_model_path("sac", env_id, seed, run_id)
    model.save(save_path)
    console.print(f"\n  [green]✓ Model saved → {save_path}.zip[/green]")
    console.print(f"  [green]✓ Metrics  → {log_path}[/green]")

    return model, cb


def run_sac_multiseed(env_id: str, seeds=(0, 1, 2), total_timesteps: int = 300_000):
    results = []
    for seed in seeds:
        model, cb = run_sac(env_id, total_timesteps=total_timesteps, seed=seed)
        if cb is not None:
            results.append(cb)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    console.print(Rule("[bold magenta]Phase 1 — SAC Experiments"))

    # Pendulum-v1: classic continuous control benchmark
    run_sac("Pendulum-v1",              total_timesteps=100_000)

    # LunarLanderContinuous-v3: harder continuous task
    run_sac("LunarLanderContinuous-v3", total_timesteps=300_000)
