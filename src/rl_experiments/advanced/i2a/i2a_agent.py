from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.rule import Rule

from rl_experiments.advanced.common.envs import make_state_env
from rl_experiments.advanced.common.replay import TransitionReplay
from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id

console = Console()


class EnvModel(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim + 1),
        )
        self.n_actions = n_actions

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        y = self.net(x)
        return y[:, :-1], y[:, -1]


class I2APolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.rollout_encoder = nn.GRU(input_size=obs_dim + 1, hidden_size=hidden, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(obs_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs, rollout_feats):
        x = torch.cat([obs, rollout_feats], dim=-1)
        return self.policy_head(x)


def train_i2a(env_id: str = "CartPole-v1", n_steps: int = 60_000, seed: int = 0, run_id: str | None = None):
    run_id = run_id or make_run_id()
    console.rule(f"[bold blue]I2A · {env_id}")
    env = make_state_env(env_id, seed=seed)
    device = get_device(verbose=False)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = EnvModel(obs_dim, n_actions).to(device)
    policy = I2APolicy(obs_dim, n_actions).to(device)
    value = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    replay = TransitionReplay(100_000)

    log_path = build_log_path("i2a", env_id, seed, run_id)
    model_path = build_custom_model_path("i2a", env_id, seed, run_id, "pt")
    fields = ["step", "episode", "reward", "model_loss", "policy_loss", "wall_time"]

    obs, _ = env.reset(seed=seed)
    ep, ep_r = 0, 0.0
    model_losses, pol_losses = [], []

    with ExperimentLogger(log_path, fields) as logger:
        for t in range(1, n_steps + 1):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            # imagine one-step outcomes for each action
            imag_pairs = []
            for a in range(n_actions):
                a_oh = torch.zeros((1, n_actions), device=device)
                a_oh[0, a] = 1.0
                nobs_pred, r_pred = model(obs_t, a_oh)
                imag_pairs.append(torch.cat([nobs_pred, r_pred.unsqueeze(-1)], dim=-1))
            imag_seq = torch.stack(imag_pairs, dim=1)  # (1, A, obs+1)
            _, h = policy.rollout_encoder(imag_seq)
            logits = policy(obs_t, h.squeeze(0))
            action = int(torch.distributions.Categorical(logits=logits).sample().item())

            nobs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            replay.add(obs, action, reward, nobs, done)
            obs = nobs
            ep_r += reward

            if len(replay) >= 512 and t % 2 == 0:
                b_obs, b_act, b_rew, b_nobs, b_done = replay.sample(256)
                b_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
                b_act = torch.tensor(b_act, dtype=torch.int64, device=device)
                b_rew = torch.tensor(b_rew, dtype=torch.float32, device=device)
                b_nobs = torch.tensor(b_nobs, dtype=torch.float32, device=device)
                b_done = torch.tensor(b_done, dtype=torch.float32, device=device)

                a_oh = torch.zeros((b_obs.size(0), n_actions), device=device)
                a_oh.scatter_(1, b_act.unsqueeze(1), 1.0)
                pred_nobs, pred_r = model(b_obs, a_oh)
                m_loss = F.mse_loss(pred_nobs, b_nobs) + F.mse_loss(pred_r, b_rew)

                # policy/value update
                v = value(b_obs).squeeze(-1)
                with torch.no_grad():
                    target = b_rew + (1.0 - b_done) * 0.99 * value(b_nobs).squeeze(-1)
                    adv = target - v
                # reuse fresh imagination in batch form
                imag_feats = []
                for a in range(n_actions):
                    a_oh_b = torch.zeros((b_obs.size(0), n_actions), device=device)
                    a_oh_b[:, a] = 1.0
                    p_no, p_r = model(b_obs, a_oh_b)
                    imag_feats.append(torch.cat([p_no, p_r.unsqueeze(-1)], dim=-1))
                imag_seq_b = torch.stack(imag_feats, dim=1)
                _, h_b = policy.rollout_encoder(imag_seq_b)
                logits_b = policy(b_obs, h_b.squeeze(0))
                logp = torch.log_softmax(logits_b, dim=-1).gather(1, b_act.unsqueeze(1)).squeeze(1)
                p_loss = -(logp * adv.detach()).mean() + 0.5 * F.mse_loss(v, target)

                loss = m_loss + p_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                model_losses.append(float(m_loss.item()))
                pol_losses.append(float(p_loss.item()))

            if done:
                logger.log(
                    step=t,
                    episode=ep,
                    reward=ep_r,
                    model_loss=float(np.mean(model_losses[-20:])) if model_losses else 0.0,
                    policy_loss=float(np.mean(pol_losses[-20:])) if pol_losses else 0.0,
                )
                ep += 1
                ep_r = 0.0
                obs, _ = env.reset()

    env.close()
    torch.save({"env_model": model.state_dict(), "policy": policy.state_dict(), "value": value.state_dict()}, model_path)
    console.print(f"[green]✓ Saved → {model_path}[/green]")
    console.print(f"[green]✓ Log   → {log_path}[/green]")
    return model_path, log_path

