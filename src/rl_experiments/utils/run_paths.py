"""
utils/run_paths.py
──────────────────
Helpers for generating unique run IDs and output paths so repeated
experiments do not overwrite prior artifacts.
"""

from datetime import datetime
from pathlib import Path


def make_run_id() -> str:
    """Return a timestamp run identifier (local time)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_stem(algorithm: str, env_id: str, seed: int, run_id: str) -> str:
    """Build canonical filename stem for one training invocation."""
    env_slug = env_id.replace("-", "_").lower()
    algo_slug = algorithm.lower()
    return f"{algo_slug}_{env_slug}_seed{seed}_{run_id}"


def build_log_path(algorithm: str, env_id: str, seed: int, run_id: str) -> str:
    """Build logs/<run_id>/*.csv path for one run."""
    stem = build_run_stem(algorithm, env_id, seed, run_id)
    path = Path("logs") / run_id / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_model_path(algorithm: str, env_id: str, seed: int, run_id: str) -> str:
    """
    Build model checkpoint base path.
    For SB3 models, callers can pass this directly to model.save(...).
    """
    stem = build_run_stem(algorithm, env_id, seed, run_id)
    path = Path("results") / run_id / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_custom_model_path(
    algorithm: str,
    env_id: str,
    seed: int,
    run_id: str,
    extension: str,
) -> str:
    """
    Build custom model path under results/<run_id>/<algorithm>/<env>/.
    Used by non-SB3 trainers that save their own checkpoint formats.
    """
    env_slug = env_id.replace("-", "_").lower()
    algo_slug = algorithm.lower()
    filename = f"{algo_slug}_seed{seed}_{run_id}.{extension.lstrip('.')}"
    path = Path("results") / run_id / algo_slug / env_slug / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
