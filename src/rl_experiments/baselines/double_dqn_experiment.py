"""
Double DQN experiment runner.
van Hasselt et al., 2015 — Deep Reinforcement Learning with Double Q-learning.
"""

from rich.console import Console
from rich.rule import Rule

from rl_experiments.baselines.dqn_variants import train_dqn_variant

console = Console()


def run_double_dqn(env_id: str, total_timesteps: int = 300_000, seed: int = 0, run_id: str | None = None):
    return train_dqn_variant(
        variant_name="double_dqn",
        env_id=env_id,
        total_timesteps=total_timesteps,
        seed=seed,
        run_id=run_id,
    )


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    console.print(Rule("[bold magenta]Double DQN Experiments"))
    run_double_dqn("CartPole-v1", total_timesteps=300_000, seed=0)
    run_double_dqn("LunarLander-v3", total_timesteps=500_000, seed=0)
