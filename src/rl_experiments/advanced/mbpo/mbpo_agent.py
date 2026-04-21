from __future__ import annotations

import os

from rich.console import Console
from stable_baselines3 import SAC
import numpy as np
import torch
import torch.nn.functional as F

from rl_experiments.advanced.common.envs import make_state_env
from rl_experiments.advanced.common.models import EnsembleDynamics
from rl_experiments.advanced.common.replay import TransitionReplay
from rl_experiments.utils.device_utils import get_device_str, get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id

console = Console()


def train_mbpo(env_id: str = "Pendulum-v1", n_steps: int = 50_000, seed: int = 0, run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold blue]MBPO · {env_id}")
    env = make_state_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = get_device(verbose=False)
    replay = TransitionReplay(200_000)
    dynamics = EnsembleDynamics(obs_dim, action_dim, ensemble=5).to(device)
    dyn_optim = torch.optim.Adam(dynamics.parameters(), lr=3e-4)

    os.makedirs("logs/tensorboard/mbpo", exist_ok=True)
    sac = SAC("MlpPolicy", env, device=get_device_str(verbose=False), tensorboard_log="logs/tensorboard/mbpo", seed=seed, verbose=0)

    log_path = build_log_path("mbpo", env_id, seed, run_id)
    model_path = build_custom_model_path("mbpo", env_id, seed, run_id, "zip")
    dyn_path = build_custom_model_path("mbpo_dynamics", env_id, seed, run_id, "pt")
    fields = ["step", "episode", "reward", "dynamics_loss", "wall_time"]

    obs, _ = env.reset(seed=seed)
    ep, ep_r = 0, 0.0
    losses = []

    with ExperimentLogger(log_path, fields) as logger:
        for t in range(1, n_steps + 1):
            action, _ = sac.predict(obs, deterministic=False)
            nobs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            replay.add(obs, action, reward, nobs, done)
            obs = nobs
            ep_r += reward

            # train SAC incrementally
            sac.learn(total_timesteps=1, reset_num_timesteps=False, progress_bar=False)

            if len(replay) >= 1024 and t % 2 == 0:
                b_obs, b_act, b_rew, b_nobs, _ = replay.sample(256)
                b_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act = torch.tensor(b_act, dtype=torch.float32, device=device)
                b_nobs = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_rew = torch.tensor(b_rew, dtype=torch.float32, device=device)
                pred = dynamics(b_obs, b_act)
                loss = F.mse_loss(pred[:, :, :obs_dim], (b_nobs - b_obs).unsqueeze(0))
                loss += F.mse_loss(pred[:, :, -1], b_rew.unsqueeze(0))
                dyn_optim.zero_grad()
                loss.backward()
                dyn_optim.step()
                losses.append(float(loss.item()))

            if done:
                logger.log(
                    step=t,
                    episode=ep,
                    reward=ep_r,
                    dynamics_loss=float(np.mean(losses[-20:])) if losses else 0.0,
                )
                ep += 1
                ep_r = 0.0
                obs, _ = env.reset()

    env.close()
    sac.save(model_path.replace(".zip", ""))
    torch.save({"model": dynamics.state_dict(), "env_id": env_id, "seed": seed}, dyn_path)
    console.print(f"[green]✓ Saved policy   → {model_path}[/green]")
    console.print(f"[green]✓ Saved dynamics → {dyn_path}[/green]")
    console.print(f"[green]✓ Log           → {log_path}[/green]")
    return model_path, log_path

