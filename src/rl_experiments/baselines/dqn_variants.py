"""
baselines/dqn_variants.py
─────────────────────────
PyTorch implementations of DQN-family algorithms for discrete-control tasks.

Implemented variants:
- Double DQN (van Hasselt et al., 2015)
- Prioritized Experience Replay DQN (Schaul et al., 2015)
- Rainbow DQN (Hessel et al., 2018): Double + Dueling + PER + n-step + Noisy Nets + C51

The architecture follows the cited papers, adapted to low-dimensional state inputs
used in this repository (CartPole/LunarLander), replacing Atari CNN encoders with MLPs.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from rich.console import Console
from rich.rule import Rule

from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id

console = Console()


@dataclass
class VariantConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    buffer_size: int = 100_000
    batch_size: int = 128
    learning_starts: int = 1_000
    target_update_interval: int = 1_000
    train_freq: int = 4
    grad_clip: float = 10.0
    hidden_dim: int = 256
    eps_start: float = 1.0
    eps_final: float = 0.05
    eps_decay_fraction: float = 0.2
    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200_000
    per_eps: float = 1e-5
    # n-step
    n_step: int = 1
    # C51
    v_min: float = -200.0
    v_max: float = 200.0
    n_atoms: int = 51
    # Noisy nets
    noisy_std: float = 0.5


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def update(self, idx: int, priority: float):
        tree_idx = idx + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += change

    def total(self) -> float:
        return float(self.tree[1])

    def get(self, value: float) -> tuple[int, float]:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        self.tree = SumTree(capacity)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)

        p = self.max_priority ** self.alpha
        self.tree.update(i, p)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        total = self.tree.total()
        segment = total / batch_size
        idxs = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p = self.tree.get(s)
            idxs.append(idx)
            priorities.append(max(p, 1e-8))

        probs = np.array(priorities, dtype=np.float32) / max(total, 1e-8)
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        idxs = np.array(idxs, dtype=np.int64)
        batch = (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
            idxs,
            weights.astype(np.float32),
        )
        return batch

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray, eps: float):
        for idx, err in zip(idxs, td_errors):
            p = (abs(float(err)) + eps) ** self.alpha
            self.tree.update(int(idx), p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.size


class NoisyLinear(nn.Module):
    # Factorized gaussian noise from Fortunato et al. 2017.
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        sigma = self.sigma0 / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(sigma)

    def reset_noise(self):
        eps_in = self._scaled_noise(self.in_features)
        eps_out = self._scaled_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    @staticmethod
    def _scaled_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int, dueling: bool, noisy: bool):
        super().__init__()
        linear = NoisyLinear if noisy else nn.Linear
        self.noisy = noisy
        self.dueling = dueling

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if dueling:
            self.adv1 = linear(hidden_dim, hidden_dim)
            self.adv2 = linear(hidden_dim, n_actions)
            self.val1 = linear(hidden_dim, hidden_dim)
            self.val2 = linear(hidden_dim, 1)
        else:
            self.head1 = linear(hidden_dim, hidden_dim)
            self.head2 = linear(hidden_dim, n_actions)

    def forward(self, x):
        h = self.backbone(x)
        if self.dueling:
            adv = F.relu(self.adv1(h))
            adv = self.adv2(adv)
            val = F.relu(self.val1(h))
            val = self.val2(val)
            return val + adv - adv.mean(dim=1, keepdim=True)
        q = F.relu(self.head1(h))
        return self.head2(q)

    def reset_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class C51Network(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, n_atoms: int, hidden_dim: int, noisy: bool):
        super().__init__()
        linear = NoisyLinear if noisy else nn.Linear
        self.noisy = noisy
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Dueling distributional heads.
        self.adv1 = linear(hidden_dim, hidden_dim)
        self.adv2 = linear(hidden_dim, n_actions * n_atoms)
        self.val1 = linear(hidden_dim, hidden_dim)
        self.val2 = linear(hidden_dim, n_atoms)

    def dist(self, x):
        h = self.backbone(x)
        adv = F.relu(self.adv1(h))
        adv = self.adv2(adv).view(-1, self.n_actions, self.n_atoms)
        val = F.relu(self.val1(h))
        val = self.val2(val).view(-1, 1, self.n_atoms)
        logits = val + adv - adv.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1).clamp(min=1e-6, max=1.0)

    def q_values(self, x, support: torch.Tensor):
        d = self.dist(x)
        return (d * support.view(1, 1, -1)).sum(dim=-1)

    def reset_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


def _check_discrete_env(env_id: str):
    env = gym.make(env_id)
    import gymnasium.spaces as spaces
    ok = isinstance(env.action_space, spaces.Discrete)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n if ok else 0
    env.close()
    return ok, obs_dim, n_actions


def _epsilon(frame: int, total_timesteps: int, eps_start: float, eps_final: float, frac: float):
    decay_steps = max(1, int(total_timesteps * frac))
    ratio = min(frame / decay_steps, 1.0)
    return eps_start + ratio * (eps_final - eps_start)


def _beta(frame: int, beta_start: float, beta_frames: int):
    return min(1.0, beta_start + frame * (1.0 - beta_start) / max(beta_frames, 1))


def _nstep_push(nstep_buf: deque, gamma: float, n_step: int):
    rew, next_obs, done = 0.0, nstep_buf[-1][3], nstep_buf[-1][4]
    for i, (_, _, r, ns, d) in enumerate(nstep_buf):
        rew += (gamma ** i) * r
        next_obs = ns
        done = d
        if d:
            break
    s, a = nstep_buf[0][0], nstep_buf[0][1]
    return s, a, rew, next_obs, done


def train_dqn_variant(
    variant_name: str,
    env_id: str,
    total_timesteps: int = 300_000,
    seed: int = 0,
    run_id: str | None = None,
):
    """
    Train one DQN variant.
    Supported variant_name: "double_dqn", "per_dqn", "rainbow".
    """
    run_id = run_id or make_run_id()
    console.rule(f"[bold blue]{variant_name.upper()} · {env_id}")
    ok, obs_dim, n_actions = _check_discrete_env(env_id)
    if not ok:
        console.print(f"[red]✗ {variant_name} requires Discrete action space[/red]")
        return None

    cfg = VariantConfig()
    is_double = variant_name in {"double_dqn", "rainbow"}
    is_per = variant_name in {"per_dqn", "rainbow"}
    is_rainbow = variant_name == "rainbow"
    use_noisy = is_rainbow
    use_c51 = is_rainbow
    dueling = is_rainbow
    n_step = 3 if is_rainbow else 1
    if is_rainbow:
        cfg.lr = 6.25e-5
        cfg.batch_size = 32
        cfg.hidden_dim = 256

    device = get_device(verbose=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)

    if use_c51:
        net = C51Network(obs_dim, n_actions, cfg.n_atoms, cfg.hidden_dim, noisy=use_noisy).to(device)
        tgt = C51Network(obs_dim, n_actions, cfg.n_atoms, cfg.hidden_dim, noisy=use_noisy).to(device)
        support = torch.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms, device=device)
        delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)
    else:
        net = QNetwork(obs_dim, n_actions, cfg.hidden_dim, dueling=dueling, noisy=use_noisy).to(device)
        tgt = QNetwork(obs_dim, n_actions, cfg.hidden_dim, dueling=dueling, noisy=use_noisy).to(device)
        support = None
        delta_z = None
    tgt.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    replay = PrioritizedReplayBuffer(obs_dim, cfg.buffer_size, alpha=cfg.per_alpha if is_per else 0.0)
    nstep_buf = deque(maxlen=n_step)

    log_path = build_log_path(variant_name, env_id, seed, run_id)
    model_path = build_custom_model_path(variant_name, env_id, seed, run_id, extension="pt")
    Path("logs/tensorboard").mkdir(parents=True, exist_ok=True)

    fields = ["timestep", "episode", "reward", "loss", "epsilon", "beta", "wall_time"]
    episode_reward = 0.0
    episode_idx = 0
    losses = []

    with ExperimentLogger(log_path, fields) as logger:
        for t in range(1, total_timesteps + 1):
            eps = 0.0 if use_noisy else _epsilon(
                t, total_timesteps, cfg.eps_start, cfg.eps_final, cfg.eps_decay_fraction
            )
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    if use_c51:
                        q = net.q_values(s, support)
                    else:
                        q = net(s)
                    action = int(q.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            nstep_buf.append((obs, action, reward, next_obs, done))
            if len(nstep_buf) == n_step:
                replay.add(*_nstep_push(nstep_buf, cfg.gamma, n_step))
            if done:
                while nstep_buf:
                    replay.add(*_nstep_push(nstep_buf, cfg.gamma, n_step))
                    nstep_buf.popleft()
                obs, _ = env.reset()
                logger.log(
                    timestep=t,
                    episode=episode_idx,
                    reward=episode_reward,
                    loss=float(np.mean(losses[-20:])) if losses else 0.0,
                    epsilon=eps,
                    beta=_beta(t, cfg.per_beta_start, cfg.per_beta_frames),
                )
                episode_idx += 1
                episode_reward = 0.0
            else:
                obs = next_obs

            if t < cfg.learning_starts or t % cfg.train_freq != 0 or len(replay) < cfg.batch_size:
                continue

            beta = _beta(t, cfg.per_beta_start, cfg.per_beta_frames) if is_per else 1.0
            b_obs, b_act, b_rew, b_nobs, b_done, idxs, isw = replay.sample(cfg.batch_size, beta=beta)

            b_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
            b_act = torch.tensor(b_act, dtype=torch.int64, device=device)
            b_rew = torch.tensor(b_rew, dtype=torch.float32, device=device)
            b_nobs = torch.tensor(b_nobs, dtype=torch.float32, device=device)
            b_done = torch.tensor(b_done, dtype=torch.float32, device=device)
            isw_t = torch.tensor(isw, dtype=torch.float32, device=device)

            if use_noisy:
                net.reset_noise()
                tgt.reset_noise()

            if use_c51:
                dist = net.dist(b_obs)
                act_dist = dist[torch.arange(cfg.batch_size, device=device), b_act]

                with torch.no_grad():
                    if is_double:
                        next_q_online = net.q_values(b_nobs, support)
                        next_actions = next_q_online.argmax(dim=1)
                    else:
                        next_q_tgt = tgt.q_values(b_nobs, support)
                        next_actions = next_q_tgt.argmax(dim=1)

                    next_dist = tgt.dist(b_nobs)[torch.arange(cfg.batch_size, device=device), next_actions]
                    gamma_n = cfg.gamma ** n_step
                    tz = b_rew.unsqueeze(1) + (1.0 - b_done.unsqueeze(1)) * gamma_n * support.unsqueeze(0)
                    tz = tz.clamp(cfg.v_min, cfg.v_max)
                    b = (tz - cfg.v_min) / delta_z
                    l = b.floor().long()
                    u = b.ceil().long()

                    proj = torch.zeros_like(next_dist)
                    offset = torch.arange(cfg.batch_size, device=device).unsqueeze(1) * cfg.n_atoms
                    proj.view(-1).index_add_(
                        0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                    )
                    proj.view(-1).index_add_(
                        0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                    )

                per_sample_loss = -(proj * act_dist.log()).sum(dim=1)
                loss = (isw_t * per_sample_loss).mean()
                td_err = per_sample_loss.detach().cpu().numpy()
            else:
                q = net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if is_double:
                        next_a = net(b_nobs).argmax(dim=1, keepdim=True)
                        next_q = tgt(b_nobs).gather(1, next_a).squeeze(1)
                    else:
                        next_q = tgt(b_nobs).max(dim=1).values
                    target = b_rew + (1.0 - b_done) * (cfg.gamma ** n_step) * next_q
                td = target - q
                per_sample_loss = F.smooth_l1_loss(q, target, reduction="none")
                loss = (isw_t * per_sample_loss).mean()
                td_err = td.detach().abs().cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

            if is_per:
                replay.update_priorities(idxs, td_err, cfg.per_eps)

            if t % cfg.target_update_interval == 0:
                tgt.load_state_dict(net.state_dict())

    env.close()
    torch.save(
        {
            "variant": variant_name,
            "env_id": env_id,
            "seed": seed,
            "model": net.state_dict(),
            "config": cfg.__dict__,
        },
        model_path,
    )
    console.print(f"  [green]✓ Saved model → {model_path}[/green]")
    console.print(f"  [green]✓ Saved log   → {log_path}[/green]")
    return model_path, log_path
