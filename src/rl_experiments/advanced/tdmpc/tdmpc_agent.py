from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.rule import Rule

from rl_experiments.advanced.common.envs import make_state_env
from rl_experiments.advanced.common.models import MLPPolicy, MLPValue, EnsembleDynamics
from rl_experiments.advanced.common.replay import TransitionReplay
from rl_experiments.advanced.common.planning import cem_plan
from rl_experiments.advanced.training.cem_score import cem_score_ensemble_dynamics
from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id

console = Console()


def train_tdmpc(env_id: str = "Pendulum-v1", n_steps: int = 40_000, seed: int = 0, variant: str = "tdmpc", run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold blue]{variant.upper()} · {env_id}")
    env = make_state_env(env_id, seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = get_device(verbose=False)

    dyn = EnsembleDynamics(obs_dim, action_dim, ensemble=5).to(device)
    policy = MLPPolicy(obs_dim, action_dim).to(device)
    value = MLPValue(obs_dim).to(device)
    opt = torch.optim.Adam(list(dyn.parameters()) + list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    replay = TransitionReplay(200_000)
    horizon = 8 if variant == "tdmpc2" else 5
    gamma = 0.995 if variant == "tdmpc2" else 0.99

    log_path = build_log_path(variant, env_id, seed, run_id)
    model_path = build_custom_model_path(variant, env_id, seed, run_id, "pt")
    fields = ["step", "episode", "reward", "loss", "horizon", "wall_time"]
    obs, _ = env.reset(seed=seed)
    ep, ep_r = 0, 0.0
    losses = []

    with ExperimentLogger(log_path, fields) as logger:
        for t in range(1, n_steps + 1):
            if len(replay) < 2000:
                action = env.action_space.sample()
            else:
                obs_np = np.asarray(obs, dtype=np.float32)

                def score_fn(action_seq):
                    return cem_score_ensemble_dynamics(
                        dyn, obs_np, obs_dim, device, action_seq, gamma, value_fn=value
                    )

                a0 = cem_plan(score_fn, horizon=horizon, action_dim=action_dim, n_samples=64, n_iters=3)
                action = np.clip(a0, env.action_space.low, env.action_space.high)

            nobs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            replay.add(obs, action, reward, nobs, done)
            ep_r += reward
            obs = nobs

            if len(replay) >= 1024 and t % 2 == 0:
                b_obs, b_act, b_rew, b_nobs, b_done = replay.sample(256)
                b_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act = torch.tensor(b_act, dtype=torch.float32, device=device)
                b_rew = torch.tensor(b_rew, dtype=torch.float32, device=device)
                b_nobs = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_done = torch.tensor(b_done, dtype=torch.float32, device=device)

                pred = dyn(b_obs, b_act)
                ds = pred[:, :, :obs_dim]
                rr = pred[:, :, -1]
                dyn_loss = F.mse_loss(ds, (b_nobs - b_obs).unsqueeze(0)) + F.mse_loss(rr, b_rew.unsqueeze(0))
                with torch.no_grad():
                    target_v = b_rew + (1 - b_done) * gamma * value(b_nobs)
                v = value(b_obs)
                v_loss = F.mse_loss(v, target_v)
                a_pred = policy(b_obs)
                a_loss = -value(b_obs + 0.05 * a_pred).mean()
                loss = dyn_loss + v_loss + 0.1 * a_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            if done:
                logger.log(step=t, episode=ep, reward=ep_r, loss=float(np.mean(losses[-20:])) if losses else 0.0, horizon=horizon)
                ep += 1
                ep_r = 0.0
                obs, _ = env.reset()

    env.close()
    torch.save({"dyn": dyn.state_dict(), "policy": policy.state_dict(), "value": value.state_dict(), "variant": variant}, model_path)
    console.print(f"[green]✓ Saved → {model_path}[/green]")
    console.print(f"[green]✓ Log   → {log_path}[/green]")
    return model_path, log_path

