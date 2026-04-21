"""
advanced/dreamer/dreamer_agent.py
─────────────────────────────────────────────────────────────────────────────
Dreamer Agent — full training loop
Hafner et al. 2019 / 2021 / 2023

Training algorithm (Algorithm 1 from Hafner et al. 2019):

  Initialise world model θ, actor φ, critic ψ
  Initialise replay buffer D
  for each episode:
    1. COLLECTION: interact with env using actor, appending to D
       - At each step: update RSSM state h,z with last obs+action
       - Action: one-hot argmax of actor output (discrete ε-greedy)
    2. WORLD MODEL UPDATE (C_wm gradient steps):
       - Sample chunk-batch from D
       - Compute L_WM = L_rec + β·L_KL + L_rew + L_cont
       - Update θ
    3. BEHAVIOUR LEARNING (C_ac gradient steps):
       - Sample posterior states (h,z) from chunk-batch
       - Roll out H=15 imagination steps
       - Compute λ-returns
       - Update actor φ, critic ψ
       - Update target critic (slow EMA)

Mac GPU: uses device="mps" on Apple Silicon.

Hyperparameters are scaled for low-dim / discrete control; see ``docs/algorithm_fidelity.md``.
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id
from .world_model   import WorldModel
from .actor_critic  import Actor, Critic, actor_critic_loss
from .replay_buffer import EpisodeReplayBuffer, Episode

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — from Appendix B of Hafner et al. 2019
# (scaled down for CartPole / low-dim envs)
# ─────────────────────────────────────────────────────────────────────────────

DREAMER_CONFIG = {
    # World Model
    "embed_dim":        64,
    "h_dim":            200,
    "z_dim":            30,
    "hidden_dim":       200,
    "kl_free_bits":     1.0,
    "kl_weight":        1.0,
    "wm_lr":            6e-4,   # World model Adam lr

    # Actor-Critic
    "actor_lr":         8e-5,
    "critic_lr":        8e-5,
    "actor_hidden":     300,
    "actor_layers":     4,
    "horizon":          15,     # Imagination horizon H
    "gamma":            0.99,
    "lam":              0.95,
    "ent_coef":         3e-4,
    "target_ema":       0.98,   # Slow target-critic EMA α

    # Training loop
    "prefill":          1000,   # Random steps before training
    "batch_size":       32,
    "chunk_len":        50,
    "wm_grad_steps":    1,      # World model updates per env step
    "ac_grad_steps":    1,      # AC updates per env step
    "max_episodes":     500,
    "eval_interval":    10,     # Log metrics every N episodes
    "grad_clip":        100.0,  # Gradient norm clipping
}


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class DreamerAgent:
    def __init__(self, env_id: str, config: dict = DREAMER_CONFIG, seed: int = 0):
        self.env_id = env_id
        self.cfg    = config
        self.seed   = seed
        self.device = get_device(verbose=False)

        # ── Environment ──────────────────────────────────────────────────────
        self._env   = gym.make(env_id)
        self._env.reset(seed=seed)
        obs_dim     = self._env.observation_space.shape[0]
        n_actions   = self._env.action_space.n      # Discrete only (CartPole)
        action_dim  = n_actions                     # one-hot size

        console.print(f"  [cyan]Dreamer Agent · {env_id}[/cyan]")
        console.print(f"  obs_dim={obs_dim}  action_dim={action_dim}  device={self.device}")

        # ── World model ──────────────────────────────────────────────────────
        feat_dim = config["h_dim"] + config["z_dim"]
        self.wm  = WorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embed_dim=config["embed_dim"],
            h_dim=config["h_dim"],
            z_dim=config["z_dim"],
            hidden_dim=config["hidden_dim"],
            kl_free_bits=config["kl_free_bits"],
            kl_weight=config["kl_weight"],
        ).to(self.device)

        # ── Actor + Critic ────────────────────────────────────────────────────
        self.actor  = Actor(
            feat_dim=feat_dim,
            action_dim=action_dim,
            hidden_dim=config["actor_hidden"],
            n_layers=config["actor_layers"],
            continuous=False,
        ).to(self.device)

        self.critic = Critic(
            feat_dim=feat_dim,
            hidden_dim=config["actor_hidden"],
            n_layers=config["actor_layers"],
        ).to(self.device)

        # Slow EMA target critic (DreamerV3)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # ── Optimisers ───────────────────────────────────────────────────────
        self.wm_opt     = torch.optim.Adam(self.wm.parameters(),     lr=config["wm_lr"])
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=config["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

        # ── Replay + Logger ──────────────────────────────────────────────────
        self.buffer = EpisodeReplayBuffer(
            max_episodes=config["max_episodes"],
            chunk_len=config["chunk_len"],
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        self.n_actions = n_actions
        self.obs_dim   = obs_dim
        self.action_dim = action_dim

    # ─────────────────────────────────────────────────────────────────────────
    # Environment interaction
    # ─────────────────────────────────────────────────────────────────────────

    def _one_hot(self, action: int) -> np.ndarray:
        oh = np.zeros(self.n_actions, dtype=np.float32)
        oh[action] = 1.0
        return oh

    @torch.no_grad()
    def _select_action(self, obs_np, h, z, explore: bool = True) -> tuple:
        """
        Select action given current RSSM state.
        Returns (action_int, action_onehot, h_new, z_new)
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        embed = self.wm.encoder(obs_t)

        # Update h with GRU, then get posterior z
        prev_act = torch.zeros(1, self.action_dim, device=self.device)
        gru_in   = torch.cat([z, prev_act], dim=-1)
        h_new    = self.wm.rssm.gru(gru_in, h)

        post_in  = torch.cat([h_new, embed], dim=-1)
        post_feat = self.wm.rssm.post_mlp(post_in)
        z_new, _ = self.wm.rssm.post_head(post_feat)

        feat     = self.wm.feat(h_new, z_new)
        act      = self.actor(feat)           # one-hot (soft straight-through)
        act_idx  = act.argmax(dim=-1).item()
        return act_idx, self._one_hot(act_idx), h_new, z_new

    def _collect_episode(self, random: bool = False) -> tuple:
        """Run one episode, return (episode, total_reward)."""
        ep  = Episode()
        obs, _ = self._env.reset()
        h, z   = self.wm.rssm.initial_state(1, self.device)
        total_r = 0.0

        while True:
            if random:
                action_idx = self._env.action_space.sample()
                action_oh  = self._one_hot(action_idx)
            else:
                action_idx, action_oh, h, z = self._select_action(obs, h, z)

            next_obs, reward, terminated, truncated, _ = self._env.step(action_idx)
            done = terminated or truncated

            ep.observations.append(obs.astype(np.float32))
            ep.actions.append(action_oh)
            ep.rewards.append(float(reward))
            ep.dones.append(done)

            total_r += reward
            obs = next_obs
            if done:
                break

        return ep, total_r

    # ─────────────────────────────────────────────────────────────────────────
    # World model training step
    # ─────────────────────────────────────────────────────────────────────────

    def _update_world_model(self) -> dict:
        batch  = self.buffer.sample(self.cfg["batch_size"], self.device)
        losses = self.wm.compute_loss(
            obs=batch["obs"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            dones=batch["dones"],
            device=self.device,
        )
        self.wm_opt.zero_grad()
        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg["grad_clip"])
        self.wm_opt.step()
        return losses

    # ─────────────────────────────────────────────────────────────────────────
    # Actor-Critic training step
    # ─────────────────────────────────────────────────────────────────────────

    def _update_actor_critic(self) -> dict:
        # Sample (h, z) starting states from posterior over a random batch
        batch = self.buffer.sample(self.cfg["batch_size"], self.device)

        with torch.no_grad():
            _, _, _, hs, zs = self.wm.observe_sequence(
                batch["obs"], batch["actions"], self.device
            )
        # Use the midpoint of the sequence as start states (middle of chunk)
        T    = hs.shape[0]
        t_s  = T // 2
        h_st = hs[t_s].detach()
        z_st = zs[t_s].detach()

        # Compute losses
        al, cl, info = actor_critic_loss(
            world_model=self.wm,
            actor=self.actor,
            critic=self.critic,
            target_critic=self.target_critic,
            h_start=h_st,
            z_start=z_st,
            horizon=self.cfg["horizon"],
            gamma=self.cfg["gamma"],
            lam=self.cfg["lam"],
            ent_coef=self.cfg["ent_coef"],
        )

        # Actor update
        self.actor_opt.zero_grad()
        al.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg["grad_clip"])
        self.actor_opt.step()

        # Critic update
        self.critic_opt.zero_grad()
        cl.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg["grad_clip"])
        self.critic_opt.step()

        # Slow target critic EMA
        alpha = self.cfg["target_ema"]
        for tp, cp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(alpha).add_((1 - alpha) * cp.data)

        return {"actor_loss": al.item(), "critic_loss": cl.item(), **info}

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, n_episodes: int = 500, log_path: str | None = None, run_id: str | None = None):
        run_id = run_id or make_run_id()
        if log_path is None:
            log_path = build_log_path("dreamer", self.env_id, self.seed, run_id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        console.rule(f"[bold blue]Dreamer · {self.env_id}")

        total_steps = 0
        fields = ["episode", "reward", "wm_loss", "kl", "rew_loss",
                  "actor_loss", "critic_loss", "total_steps", "wall_time"]

        with ExperimentLogger(log_path, fields) as logger:
            # ── Pre-fill with random data ────────────────────────────────────
            console.print(f"  [yellow]Pre-filling buffer ({self.cfg['prefill']} steps)...[/yellow]")
            while total_steps < self.cfg["prefill"]:
                ep, _ = self._collect_episode(random=True)
                self.buffer.add_episode(ep)
                total_steps += len(ep)
            console.print("  [green]✓ Buffer pre-filled[/green]")

            # ── Training episodes ────────────────────────────────────────────
            pbar = tqdm(range(n_episodes), desc="Dreamer", unit="ep")
            ep_rewards = []

            for episode_idx in pbar:
                # 1. Collect one episode with current policy
                ep, ep_reward = self._collect_episode(random=False)
                self.buffer.add_episode(ep)
                total_steps += len(ep)
                ep_rewards.append(ep_reward)

                # 2. World model updates
                wm_info = {}
                for _ in range(self.cfg["wm_grad_steps"]):
                    wm_info = self._update_world_model()

                # 3. Actor-Critic updates
                ac_info = {}
                for _ in range(self.cfg["ac_grad_steps"]):
                    ac_info = self._update_actor_critic()

                # 4. Logging
                if episode_idx % self.cfg["eval_interval"] == 0:
                    mean_r = np.mean(ep_rewards[-20:]) if len(ep_rewards) >= 20 else np.mean(ep_rewards)
                    logger.log(
                        episode=episode_idx,
                        reward=ep_reward,
                        wm_loss=wm_info.get("loss_total", 0).item() if hasattr(wm_info.get("loss_total", 0), "item") else wm_info.get("loss_total", 0),
                        kl=wm_info.get("loss_kl", 0),
                        rew_loss=wm_info.get("loss_rew", 0),
                        actor_loss=ac_info.get("actor_loss", 0),
                        critic_loss=ac_info.get("critic_loss", 0),
                        total_steps=total_steps,
                    )
                    pbar.set_postfix({
                        "reward": f"{mean_r:.1f}",
                        "wm":     f"{wm_info.get('loss_rec', 0):.3f}",
                        "kl":     f"{wm_info.get('loss_kl', 0):.3f}",
                    })

        # Save weights
        model_path = build_custom_model_path(
            algorithm="dreamer",
            env_id=self.env_id,
            seed=self.seed,
            run_id=run_id,
            extension="pt",
        )
        torch.save({
            "world_model":  self.wm.state_dict(),
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
        }, model_path)
        console.print(f"\n  [green]✓ Saved → {model_path}[/green]")
        console.print(f"  [green]✓ Log   → {log_path}[/green]")

        return ep_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs",    exist_ok=True)
    os.makedirs("results", exist_ok=True)

    agent = DreamerAgent("CartPole-v1")
    rewards = agent.train(n_episodes=300)
    console.print(f"\nFinal mean reward (last 20): {np.mean(rewards[-20:]):.1f}")
