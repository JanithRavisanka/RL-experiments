"""
Rainbow DQN experiment runner.
Hessel et al., 2018 — Rainbow: Combining Improvements in Deep RL.
"""

from rich.console import Console
from rich.rule import Rule

from rl_experiments.baselines.dqn_variants import train_dqn_variant

console = Console()


def run_rainbow(env_id: str, total_timesteps: int = 300_000, seed: int = 0, run_id: str | None = None):
    return train_dqn_variant(
        variant_name="rainbow",
        env_id=env_id,
        total_timesteps=total_timesteps,
        seed=seed,
        run_id=run_id,
    )


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    console.print(Rule("[bold magenta]Rainbow DQN Experiments"))
    run_rainbow("CartPole-v1", total_timesteps=300_000, seed=0)
    run_rainbow("LunarLander-v3", total_timesteps=500_000, seed=0)
