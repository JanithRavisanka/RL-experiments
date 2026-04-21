"""
baselines/dqn_experiment.py
─────────────────────────────────────────────────────────────────────────────
DQN — Deep Q-Network (Mnih et al., 2015)
https://www.nature.com/articles/nature14236

Architecture faithful to the original paper:
  • Q-network: maps (state) → Q-values for each discrete action
    – For low-dim inputs (CartPole): MLP 2 × 64, ReLU
    – The paper used CNNs for Atari; for tabular envs MLP is standard.
  • Experience Replay buffer: stores (s, a, r, s', done) tuples
    – Breaks temporal correlations in training data
    – Buffer size: 50,000 (paper: 1,000,000 for Atari; 50k suits CartPole)
  • Target network θ̄ (frozen copy of Q):
    – TD target: y = r + γ · max_a' Q_θ̄(s', a')
    – Updated every C = 1000 steps (hard copy, not Polyak)
  • ε-greedy exploration:
    – ε decays linearly from 1.0 → 0.05 over first 10% of training
  • Loss: Huber loss (smooth L1) — more robust than pure MSE
  • Optimizer: Adam, lr = 1e-4 (paper used RMSProp; Adam is equivalent)
  • Batch size: 32 (Mnih et al. 2015)

DQN requires DISCRETE action spaces.
  → CartPole-v1, LunarLander-v2

Mac GPU: device="mps" for Apple Silicon.
"""

import gymnasium as gym
from stable_baselines3 import DQN
from rich.console import Console
from rich.rule import Rule

from rl_experiments.utils.device_utils import get_device_str
from rl_experiments.utils.metrics import RLMetricsCallback
from rl_experiments.utils.run_paths import build_log_path, build_model_path, make_run_id

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — Mnih et al. 2015 adapted for low-dim envs
# ─────────────────────────────────────────────────────────────────────────────

DQN_CONFIG = {
    # ── Core DQN parameters ──────────────────────────────────────────────────
    "learning_rate":            1e-4,       # Adam lr (paper: RMSProp lr=0.00025)
    "buffer_size":              50_000,     # Replay memory D capacity
    "learning_starts":          1_000,      # Steps before first gradient update
    "batch_size":               32,         # Replay mini-batch size (paper: 32)
    "tau":                      1.0,        # Hard copy to target (τ=1 means hard)
    "gamma":                    0.99,       # Discount factor γ
    "train_freq":               4,          # Update every 4 env steps (paper: 4)
    "gradient_steps":           1,          # Gradient updates per train step
    "target_update_interval":   1_000,      # Copy θ → θ̄ every C steps
    # ── Exploration ε-greedy ────────────────────────────────────────────────
    "exploration_fraction":     0.1,        # Fraction of training for ε decay
    "exploration_initial_eps":  1.0,        # ε start = 1.0
    "exploration_final_eps":    0.05,       # ε end = 0.05 (paper: 0.1 for Atari)
    "optimize_memory_usage":    False,
    # ── Network ─────────────────────────────────────────────────────────────
    "policy_kwargs": {
        "net_arch": [64, 64],               # For low-dim obs; paper used CNN
        "activation_fn": __import__("torch").nn.ReLU,
    },
    "verbose":                  1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dqn(env_id: str, total_timesteps: int = 200_000, seed: int = 42, run_id: str | None = None):
    """
    Train DQN on `env_id` (must have discrete action space).

    Parameters
    ----------
    env_id          : Gymnasium environment ID
    total_timesteps : Training budget
    seed            : Random seed
    """
    console.rule(f"[bold blue]DQN · {env_id}")
    device = get_device_str()

    # Safety guard — DQN needs discrete actions
    env_check = gym.make(env_id)
    import gymnasium.spaces as spaces
    if not isinstance(env_check.action_space, spaces.Discrete):
        console.print(f"[red]✗ DQN requires a Discrete action space. "
                      f"'{env_id}' has {type(env_check.action_space).__name__}. "
                      f"Skipping.[/red]")
        env_check.close()
        return None, None
    env_check.close()

    run_id = run_id or make_run_id()
    log_path = build_log_path("dqn", env_id, seed, run_id)
    cb = RLMetricsCallback(log_path=log_path)
    os.makedirs("logs/tensorboard/dqn", exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        env=env_id,
        device=device,
        seed=seed,
        tensorboard_log="logs/tensorboard/dqn",
        **DQN_CONFIG,
    )

    console.print(f"  [cyan]Device:[/cyan]     {device}")
    console.print(f"  [cyan]Env:[/cyan]        {env_id}")
    console.print(f"  [cyan]Timesteps:[/cyan]  {total_timesteps:,}")
    console.print(f"  [cyan]ε start:[/cyan]    {DQN_CONFIG['exploration_initial_eps']}")
    console.print(f"  [cyan]ε final:[/cyan]    {DQN_CONFIG['exploration_final_eps']}")
    console.print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=cb,
        progress_bar=True,
    )

    save_path = build_model_path("dqn", env_id, seed, run_id)
    model.save(save_path)
    console.print(f"\n  [green]✓ Model saved → {save_path}.zip[/green]")
    console.print(f"  [green]✓ Metrics  → {log_path}[/green]")

    return model, cb


def run_dqn_multiseed(env_id: str, seeds=(0, 1, 2), total_timesteps: int = 200_000):
    results = []
    for seed in seeds:
        _, cb = run_dqn(env_id, total_timesteps=total_timesteps, seed=seed)
        if cb is not None:
            results.append(cb)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    console.print(Rule("[bold magenta]Phase 1 — DQN Experiments"))

    run_dqn("CartPole-v1",    total_timesteps=200_000)
    run_dqn("LunarLander-v3", total_timesteps=500_000)
