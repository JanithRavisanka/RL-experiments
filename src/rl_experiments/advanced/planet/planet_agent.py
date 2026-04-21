from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from rl_experiments.advanced.common.envs import make_pixel_env, obs_to_chw
from rl_experiments.advanced.common.replay import SequenceReplay
from rl_experiments.advanced.common.planning import cem_plan
from rl_experiments.advanced.common.models import PixelEncoder
from rl_experiments.advanced.planet.rssm import RSSMCore
from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id

console = Console()


def train_planet(env_id: str = "CartPole-v1", n_episodes: int = 100, seed: int = 0, run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold blue]PlaNet · {env_id}")
    env = make_pixel_env(env_id, seed=seed, frame_stack=4)
    device = get_device(verbose=False)
    n_actions = env.action_space.n
    latent_dim = 64

    enc = PixelEncoder(in_ch=4, latent_dim=latent_dim).to(device)
    rssm = RSSMCore(latent_dim, n_actions).to(device)
    dec = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(rssm.parameters()) + list(dec.parameters()), lr=3e-4)
    replay = SequenceReplay(2000)

    log_path = build_log_path("planet", env_id, seed, run_id)
    model_path = build_custom_model_path("planet", env_id, seed, run_id, "pt")
    fields = ["episode", "reward", "loss", "planner_score", "wall_time"]

    with ExperimentLogger(log_path, fields) as logger:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            trans = []
            ep_reward = 0.0

            while not done:
                # latent-space CEM planning over discrete actions approximated by one-hot vectors
                obs_chw = obs_to_chw(np.asarray(obs))
                obs_t = torch.tensor(obs_chw, dtype=torch.float32, device=device).unsqueeze(0)
                z0 = enc(obs_t)

                def score_fn(action_seq):
                    h = torch.zeros_like(z0)
                    z = z0
                    total = 0.0
                    for a in action_seq:
                        aid = int(np.argmax(a))
                        one_hot = torch.zeros((1, n_actions), device=device)
                        one_hot[0, aid] = 1.0
                        h, _, _, qm, _ = rssm.step(h, z, one_hot, emb=z)
                        z = qm
                        total += float(dec(z).item())
                    return total

                a0 = cem_plan(score_fn, horizon=8, action_dim=n_actions, n_samples=64, n_iters=2)
                action = int(np.argmax(a0))
                nobs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                trans.append((obs, action, reward, nobs, done))
                ep_reward += reward
                obs = nobs

            replay.add_episode(trans)

            seqs = replay.sample_sequences(batch_size=8, seq_len=20)
            losses = []
            for seq in seqs:
                h = torch.zeros((1, latent_dim), device=device)
                z = torch.zeros((1, latent_dim), device=device)
                loss = 0.0
                for (o, a, r, _no, _d) in seq:
                    o_t = torch.tensor(obs_to_chw(np.asarray(o)), dtype=torch.float32, device=device).unsqueeze(0)
                    emb = enc(o_t)
                    a_oh = torch.zeros((1, n_actions), device=device)
                    a_oh[0, int(a)] = 1.0
                    h, pm, pv, qm, qv = rssm.step(h, z, a_oh, emb=emb)
                    z = qm
                    kl = 0.5 * torch.mean((qm - pm) ** 2 + torch.exp(qv - pv) - 1 + (pv - qv))
                    rew = F.mse_loss(dec(z).squeeze(-1), torch.tensor([r], dtype=torch.float32, device=device))
                    loss = loss + kl + rew
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            logger.log(episode=ep, reward=ep_reward, loss=float(np.mean(losses)) if losses else 0.0, planner_score=ep_reward)

    env.close()
    torch.save({"encoder": enc.state_dict(), "rssm": rssm.state_dict(), "reward_head": dec.state_dict()}, model_path)
    console.print(f"[green]✓ Saved → {model_path}[/green]")
    console.print(f"[green]✓ Log   → {log_path}[/green]")
    return model_path, log_path

