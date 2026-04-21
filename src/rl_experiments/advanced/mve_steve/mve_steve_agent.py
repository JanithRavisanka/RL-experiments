from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.rule import Rule

from rl_experiments.advanced.common.envs import make_state_env
from rl_experiments.advanced.common.models import EnsembleDynamics, MLPPolicy, MLPValue
from rl_experiments.advanced.common.replay import TransitionReplay
from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id

console = Console()


def train_mve_steve(env_id: str = "Pendulum-v1", n_steps: int = 40_000, seed: int = 0, method: str = "mve", run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold blue]{method.upper()} · {env_id}")
    env = make_state_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = get_device(verbose=False)

    ensemble = EnsembleDynamics(obs_dim, action_dim, ensemble=7).to(device)
    policy = MLPPolicy(obs_dim, action_dim).to(device)
    value = MLPValue(obs_dim).to(device)
    opt = torch.optim.Adam(list(ensemble.parameters()) + list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    replay = TransitionReplay(200_000)

    horizon = 3 if method == "mve" else 5
    gamma = 0.99
    log_path = build_log_path(method, env_id, seed, run_id)
    model_path = build_custom_model_path(method, env_id, seed, run_id, "pt")
    fields = ["step", "episode", "reward", "model_loss", "value_loss", "target_var", "wall_time"]

    obs, _ = env.reset(seed=seed)
    ep, ep_r = 0, 0.0
    mlosses, vlosses, tvars = [], [], []

    with ExperimentLogger(log_path, fields) as logger:
        for t in range(1, n_steps + 1):
            with torch.no_grad():
                a = policy(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            a = np.clip(a + 0.1 * np.random.randn(*a.shape), env.action_space.low, env.action_space.high)
            nobs, reward, term, trunc, _ = env.step(a)
            done = term or trunc
            replay.add(obs, a, reward, nobs, done)
            obs = nobs
            ep_r += reward

            if len(replay) >= 1024 and t % 2 == 0:
                b_obs, b_act, b_rew, b_nobs, b_done = replay.sample(256)
                b_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act = torch.tensor(b_act, dtype=torch.float32, device=device)
                b_rew = torch.tensor(b_rew, dtype=torch.float32, device=device)
                b_nobs = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_done = torch.tensor(b_done, dtype=torch.float32, device=device)

                pred = ensemble(b_obs, b_act)
                ds, rr = pred[:, :, :obs_dim], pred[:, :, -1]
                m_loss = F.mse_loss(ds, (b_nobs - b_obs).unsqueeze(0)) + F.mse_loss(rr, b_rew.unsqueeze(0))

                with torch.no_grad():
                    # multi-step model rollout target
                    s = b_obs
                    g = torch.zeros_like(b_rew)
                    w = 1.0
                    means = []
                    for _ in range(horizon):
                        a_roll = policy(s)
                        out = ensemble(s, a_roll)
                        ds_m = out[:, :, :obs_dim]
                        r_m = out[:, :, -1]
                        s = s + ds_m.mean(dim=0)
                        means.append(r_m.mean(dim=0))
                        g = g + w * r_m.mean(dim=0)
                        w *= gamma
                    tail = w * value(s)
                    if method == "steve":
                        # uncertainty-weighted target from ensemble variance
                        var = torch.stack(means, dim=0).var(dim=0) + 1e-6
                        weight = 1.0 / var
                        weight = weight / weight.mean()
                        target = g + tail * weight
                        tvars.append(float(var.mean().item()))
                    else:
                        target = g + tail
                        tvars.append(0.0)

                v = value(b_obs)
                v_loss = F.mse_loss(v, target.detach())
                p_loss = -value(b_obs + 0.05 * policy(b_obs)).mean()
                loss = m_loss + v_loss + 0.1 * p_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                mlosses.append(float(m_loss.item()))
                vlosses.append(float(v_loss.item()))

            if done:
                logger.log(
                    step=t,
                    episode=ep,
                    reward=ep_r,
                    model_loss=float(np.mean(mlosses[-20:])) if mlosses else 0.0,
                    value_loss=float(np.mean(vlosses[-20:])) if vlosses else 0.0,
                    target_var=float(np.mean(tvars[-20:])) if tvars else 0.0,
                )
                ep += 1
                ep_r = 0.0
                obs, _ = env.reset()

    env.close()
    torch.save({"ensemble": ensemble.state_dict(), "policy": policy.state_dict(), "value": value.state_dict(), "method": method}, model_path)
    console.print(f"[green]✓ Saved → {model_path}[/green]")
    console.print(f"[green]✓ Log   → {log_path}[/green]")
    return model_path, log_path

