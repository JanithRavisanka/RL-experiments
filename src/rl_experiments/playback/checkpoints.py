"""Checkpoint path parsing and SB3 loading (shared by play_model and view_agent)."""

from __future__ import annotations

import re
from pathlib import Path

from stable_baselines3 import DQN, PPO, SAC

ALGO_MAP = {
    "ppo": "PPO",
    "sac": "SAC",
    "dqn": "DQN",
}

ENV_MAP = {
    "cartpole_v1": "CartPole-v1",
    "lunarlander_v3": "LunarLander-v3",
    "lunarlandercontinuous_v3": "LunarLanderContinuous-v3",
    "pendulum_v1": "Pendulum-v1",
}

SLUG_TO_ENV = dict(ENV_MAP)


def parse_model_file(path: Path):
    """Infer algorithm and environment from filename, e.g. ppo_cartpole_v1_seed0.zip."""
    stem = path.stem.lower()

    algo = None
    for key in ALGO_MAP:
        if stem.startswith(key + "_"):
            algo = ALGO_MAP[key]
            break

    env_id = None
    for key in ENV_MAP:
        if key in stem:
            env_id = ENV_MAP[key]
            break

    seed_match = re.search(r"seed(\d+)", stem)
    seed = int(seed_match.group(1)) if seed_match else 0

    return algo, env_id, seed


def load_sb3_model(model_path: str, algo: str):
    cls = {"PPO": PPO, "SAC": SAC, "DQN": DQN}[algo]
    return cls.load(model_path, device="cpu")


def discover_models(results_dir: str = "results") -> list:
    """Find all .zip model files under ``results/``."""
    p = Path(results_dir)
    models = []
    for f in sorted(p.glob("**/*.zip")):
        algo, env_id, seed = parse_model_file(f)
        if algo and env_id:
            models.append({"path": f, "algo": algo, "env_id": env_id, "seed": seed})
    return models


def infer_env_id_from_path(path: Path) -> str | None:
    """Infer Gymnasium env id from .../<algo>/<env_slug>/... path segments."""
    parts = path.parts
    for i, p in enumerate(parts):
        pl = p.lower()
        if pl in SLUG_TO_ENV:
            return SLUG_TO_ENV[pl]
        if i > 0 and parts[i - 1].lower() in (
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
            "mbpo_dynamics",
        ):
            slug = pl
            if slug in SLUG_TO_ENV:
                return SLUG_TO_ENV[slug]
    return None
