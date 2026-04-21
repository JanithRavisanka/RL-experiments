from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.rule import Rule

from rl_experiments.advanced.common.envs import make_pixel_env, obs_to_chw
from rl_experiments.advanced.world_models.vae import VAE
from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id

console = Console()


def train_world_models(env_id: str = "CartPole-v1", n_episodes: int = 120, seed: int = 0, run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold blue]WorldModels · {env_id}")
    device = get_device(verbose=False)
    env = make_pixel_env(env_id, seed=seed)
    n_actions = env.action_space.n

    vae = VAE().to(device)
    rnn = nn.LSTM(input_size=32 + n_actions, hidden_size=128, num_layers=1).to(device)
    mdn = nn.Linear(128, 32)
    ctrl = nn.Linear(32, n_actions).to(device)
    params = list(vae.parameters()) + list(rnn.parameters()) + list(mdn.parameters()) + list(ctrl.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)

    log_path = build_log_path("world_models", env_id, seed, run_id)
    model_path = build_custom_model_path("world_models", env_id, seed, run_id, "pt")
    fields = ["episode", "reward", "vae_loss", "mdnrnn_loss", "ctrl_loss", "wall_time"]

    with ExperimentLogger(log_path, fields) as logger:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_r = 0.0
            zs, acts, rews = [], [], []

            while not done:
                obs_t = torch.tensor(obs_to_chw(np.asarray(obs)), dtype=torch.float32, device=device).unsqueeze(0)
                xr, mu, lv, z = vae(obs_t)
                logits = ctrl(z.detach())
                action = int(logits.argmax(dim=-1).item())
                nobs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                zs.append(z.squeeze(0))
                a = torch.zeros(n_actions, device=device)
                a[action] = 1.0
                acts.append(a)
                rews.append(reward)
                obs = nobs
                ep_r += reward

            if len(zs) > 2:
                z_seq = torch.stack(zs[:-1]).unsqueeze(1)
                a_seq = torch.stack(acts[:-1]).unsqueeze(1)
                x = torch.cat([z_seq, a_seq], dim=-1)
                out, _ = rnn(x)
                z_pred = mdn(out.squeeze(1))
                z_tgt = torch.stack(zs[1:])
                mdn_loss = F.mse_loss(z_pred, z_tgt)
            else:
                mdn_loss = torch.tensor(0.0, device=device)

            vae_loss = torch.tensor(0.0, device=device)
            ctrl_loss = torch.tensor(0.0, device=device)
            if zs:
                z_b = torch.stack(zs)
                logits = ctrl(z_b)
                adv_target = torch.tensor([max(r, 0.0) for r in rews], dtype=torch.float32, device=device)
                ctrl_loss = F.cross_entropy(logits, logits.argmax(dim=-1)) * (1.0 + adv_target.mean())
                vae_loss = (mu.pow(2).mean() + lv.pow(2).mean()) * 1e-3

            loss = mdn_loss + ctrl_loss + vae_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger.log(
                episode=ep,
                reward=ep_r,
                vae_loss=float(vae_loss.item()),
                mdnrnn_loss=float(mdn_loss.item()),
                ctrl_loss=float(ctrl_loss.item()),
            )

    env.close()
    torch.save({"vae": vae.state_dict(), "rnn": rnn.state_dict(), "mdn": mdn.state_dict(), "ctrl": ctrl.state_dict()}, model_path)
    console.print(f"[green]✓ Saved → {model_path}[/green]")
    console.print(f"[green]✓ Log   → {log_path}[/green]")
    return model_path, log_path

