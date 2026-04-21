"""Register all algorithms for `rl_experiments.api.training.train`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rl_experiments.api.training import TrainConfig, TrainResult, register_algorithm, _scale


def _result(
    algorithm: str,
    cfg: TrainConfig,
    run_id: str,
    log_csv: str,
    artifacts: dict[str, str | Path] | None = None,
) -> TrainResult:
    arts = {k: Path(v) for k, v in (artifacts or {}).items()}
    return TrainResult(
        algorithm=algorithm,
        env_id=cfg.env_id,
        seed=cfg.seed,
        run_id=run_id,
        log_csv=Path(log_csv),
        artifacts=arts,
    )


def _train_ppo(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.baselines.ppo_experiment import run_ppo
    from rl_experiments.utils.run_paths import build_log_path, build_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 200_000
    if cfg.quick and cfg.budget_steps is None:
        steps = _scale(cfg.quick, 200_000)
    run_ppo(cfg.env_id, steps, cfg.seed, run_id=rid)
    log_csv = build_log_path("ppo", cfg.env_id, cfg.seed, rid)
    model_base = build_model_path("ppo", cfg.env_id, cfg.seed, rid)
    return _result("ppo", cfg, rid, log_csv, {"policy": str(model_base) + ".zip"})


def _train_sac(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.baselines.sac_experiment import run_sac
    from rl_experiments.utils.run_paths import build_log_path, build_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 100_000
    if cfg.quick and cfg.budget_steps is None:
        steps = _scale(cfg.quick, 100_000)
    run_sac(cfg.env_id, steps, cfg.seed, run_id=rid)
    log_csv = build_log_path("sac", cfg.env_id, cfg.seed, rid)
    model_base = build_model_path("sac", cfg.env_id, cfg.seed, rid)
    return _result("sac", cfg, rid, log_csv, {"policy": str(model_base) + ".zip"})


def _train_dqn(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.baselines.dqn_experiment import run_dqn
    from rl_experiments.utils.run_paths import build_log_path, build_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 200_000
    if cfg.quick and cfg.budget_steps is None:
        steps = _scale(cfg.quick, 200_000)
    run_dqn(cfg.env_id, steps, cfg.seed, run_id=rid)
    log_csv = build_log_path("dqn", cfg.env_id, cfg.seed, rid)
    model_base = build_model_path("dqn", cfg.env_id, cfg.seed, rid)
    return _result("dqn", cfg, rid, log_csv, {"policy": str(model_base) + ".zip"})


def _train_double_dqn(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.baselines.double_dqn_experiment import run_double_dqn
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 200_000
    if cfg.quick and cfg.budget_steps is None:
        steps = _scale(cfg.quick, 200_000)
    run_double_dqn(cfg.env_id, steps, cfg.seed, run_id=rid)
    log_csv = build_log_path("double_dqn", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("double_dqn", cfg.env_id, cfg.seed, rid, "pt")
    return _result("double_dqn", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_per_dqn(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.baselines.per_dqn_experiment import run_per_dqn
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 200_000
    if cfg.quick and cfg.budget_steps is None:
        steps = _scale(cfg.quick, 200_000)
    run_per_dqn(cfg.env_id, steps, cfg.seed, run_id=rid)
    log_csv = build_log_path("per_dqn", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("per_dqn", cfg.env_id, cfg.seed, rid, "pt")
    return _result("per_dqn", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_rainbow(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.baselines.rainbow_experiment import run_rainbow
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 200_000
    if cfg.quick and cfg.budget_steps is None:
        steps = _scale(cfg.quick, 200_000)
    run_rainbow(cfg.env_id, steps, cfg.seed, run_id=rid)
    log_csv = build_log_path("rainbow", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("rainbow", cfg.env_id, cfg.seed, rid, "pt")
    return _result("rainbow", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_dreamer(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.dreamer.dreamer_agent import DreamerAgent
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    eps = cfg.budget_episodes if cfg.budget_episodes is not None else 300
    if cfg.quick and cfg.budget_episodes is None:
        eps = int(300 * 0.2)
    agent = DreamerAgent(cfg.env_id, seed=cfg.seed)
    agent.train(n_episodes=eps, run_id=rid)
    log_csv = build_log_path("dreamer", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("dreamer", cfg.env_id, cfg.seed, rid, "pt")
    return _result("dreamer", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_muzero(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.muzero.muzero_agent import MuZeroAgent
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    eps = cfg.budget_episodes if cfg.budget_episodes is not None else 200
    if cfg.quick and cfg.budget_episodes is None:
        eps = int(200 * 0.2)
    agent = MuZeroAgent(cfg.env_id, seed=cfg.seed)
    agent.train(n_episodes=eps, run_id=rid)
    log_csv = build_log_path("muzero", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("muzero", cfg.env_id, cfg.seed, rid, "pt")
    return _result("muzero", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_pets(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.pets.pets_agent import train_pets
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 40_000
    if cfg.quick and cfg.budget_steps is None:
        steps = int(40_000 * 0.2)
    train_pets(cfg.env_id, n_steps=steps, seed=cfg.seed, run_id=rid)
    log_csv = build_log_path("pets", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("pets", cfg.env_id, cfg.seed, rid, "pt")
    return _result("pets", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_mbpo(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.mbpo.mbpo_agent import train_mbpo
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 50_000
    if cfg.quick and cfg.budget_steps is None:
        steps = int(30_000 * 0.2)
    mp, _ = train_mbpo(cfg.env_id, n_steps=steps, seed=cfg.seed, run_id=rid)
    log_csv = build_log_path("mbpo", cfg.env_id, cfg.seed, rid)
    dyn = build_custom_model_path("mbpo_dynamics", cfg.env_id, cfg.seed, rid, "pt")
    return _result("mbpo", cfg, rid, log_csv, {"policy": mp, "dynamics": dyn})


def _train_planet(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.planet.planet_agent import train_planet
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    eps = cfg.budget_episodes if cfg.budget_episodes is not None else 120
    if cfg.quick and cfg.budget_episodes is None:
        eps = int(120 * 0.2)
    train_planet(cfg.env_id, n_episodes=eps, seed=cfg.seed, run_id=rid)
    log_csv = build_log_path("planet", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("planet", cfg.env_id, cfg.seed, rid, "pt")
    return _result("planet", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_tdmpc(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.tdmpc.tdmpc_agent import train_tdmpc
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    variant = cfg.extra.get("variant", "tdmpc")
    steps = cfg.budget_steps if cfg.budget_steps is not None else 35_000
    if cfg.quick and cfg.budget_steps is None:
        steps = int(35_000 * 0.2)
    train_tdmpc(cfg.env_id, n_steps=steps, seed=cfg.seed, variant=variant, run_id=rid)
    log_csv = build_log_path(variant, cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path(variant, cfg.env_id, cfg.seed, rid, "pt")
    return _result(variant, cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_tdmpc2(cfg: TrainConfig) -> TrainResult:
    return _train_tdmpc(replace(cfg, extra={**cfg.extra, "variant": "tdmpc2"}))


def _train_world_models(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.world_models.world_models_agent import train_world_models
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    eps = cfg.budget_episodes if cfg.budget_episodes is not None else 120
    if cfg.quick and cfg.budget_episodes is None:
        eps = int(120 * 0.2)
    train_world_models(cfg.env_id, n_episodes=eps, seed=cfg.seed, run_id=rid)
    log_csv = build_log_path("world_models", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("world_models", cfg.env_id, cfg.seed, rid, "pt")
    return _result("world_models", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_i2a(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.i2a.i2a_agent import train_i2a
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 60_000
    if cfg.quick and cfg.budget_steps is None:
        steps = int(50_000 * 0.2)
    train_i2a(cfg.env_id, n_steps=steps, seed=cfg.seed, run_id=rid)
    log_csv = build_log_path("i2a", cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path("i2a", cfg.env_id, cfg.seed, rid, "pt")
    return _result("i2a", cfg, rid, log_csv, {"checkpoint": ckpt})


def _train_mve_steve(cfg: TrainConfig) -> TrainResult:
    from rl_experiments.advanced.mve_steve.mve_steve_agent import train_mve_steve
    from rl_experiments.utils.run_paths import build_log_path, build_custom_model_path

    method = cfg.extra.get("method") or cfg.algorithm
    rid = cfg.resolved_run_id()
    steps = cfg.budget_steps if cfg.budget_steps is not None else 35_000
    if cfg.quick and cfg.budget_steps is None:
        steps = int(35_000 * 0.2)
    train_mve_steve(cfg.env_id, n_steps=steps, seed=cfg.seed, method=method, run_id=rid)
    log_csv = build_log_path(method, cfg.env_id, cfg.seed, rid)
    ckpt = build_custom_model_path(method, cfg.env_id, cfg.seed, rid, "pt")
    return _result(method, cfg, rid, log_csv, {"checkpoint": ckpt})


def register_all() -> None:
    register_algorithm("ppo", _train_ppo)
    register_algorithm("sac", _train_sac)
    register_algorithm("dqn", _train_dqn)
    register_algorithm("double_dqn", _train_double_dqn)
    register_algorithm("per_dqn", _train_per_dqn)
    register_algorithm("rainbow", _train_rainbow)
    register_algorithm("dreamer", _train_dreamer)
    register_algorithm("muzero", _train_muzero)
    register_algorithm("pets", _train_pets)
    register_algorithm("mbpo", _train_mbpo)
    register_algorithm("planet", _train_planet)
    register_algorithm("tdmpc", _train_tdmpc)
    register_algorithm("tdmpc2", _train_tdmpc2)
    register_algorithm("world_models", _train_world_models)
    register_algorithm("i2a", _train_i2a)
    register_algorithm("mve", _train_mve_steve)
    register_algorithm("steve", _train_mve_steve)


register_all()

__all__ = ["register_all"]
