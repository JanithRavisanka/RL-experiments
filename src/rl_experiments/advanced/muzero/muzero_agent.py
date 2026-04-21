"""
advanced/muzero/muzero_agent.py
─────────────────────────────────────────────────────────────────────────────
MuZero Agent — complete training loop
Schrittwieser et al. 2020 (https://www.nature.com/articles/s41586-020-03051-4)

Training Algorithm (Algorithm 1 from the paper):

  Initialise networks: Representation f_θ, Dynamics g_θ, Prediction p_θ
  Initialise replay buffer R

  for each episode:
    1. SELF-PLAY:
       - Encode obs at t=0: h_0 = f_θ(o_0)
       - At each step t:
           Run MCTS from h_t → improved policy π̂_t and action a_t
           Execute a_t → get reward r_t, obs o_{t+1}
           Compute next hidden state: h_{t+1} = g_θ(h_t, a_t)
       - Store (o_t, a_t, r_t, π̂_t) in episode buffer → add to R

    2. TRAINING (after each episode):
       Sample batch of (o_s, a_{s:s+K-1}, targets) from R
       For each unroll step k = 0..K:
         h_0 = f_θ(o_s)              [k=0: representation]
         h_k, r̂_k = g_θ(h_{k-1}, a_{s+k-1})  [k>0: dynamics]
         p_k, v_k = p_θ(h_k)         [prediction]

       Loss = L_v + L_p + L_r  (all K steps)
         L_v = MSE(v_k, z_k)          value loss
         L_p = CE(p_k, π̂_k)          policy loss (KL/cross-entropy)
         L_r = MSE(r̂_k, u_k)         reward loss

Mac GPU: device="mps" for Apple Silicon.

Networks and MCTS budget are reduced vs full-scale MuZero; see ``docs/algorithm_fidelity.md``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

from rl_experiments.utils.device_utils import get_device
from rl_experiments.utils.metrics import ExperimentLogger
from rl_experiments.utils.run_paths import build_custom_model_path, build_log_path, make_run_id
from .networks       import RepresentationNetwork, DynamicsNetwork, PredictionNetwork, to_one_hot
from .mcts           import MCTS
from .replay_buffer  import MuZeroReplayBuffer, GameTrajectory, Transition

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — from Appendix B of Schrittwieser et al. 2020
# (adapted / scaled for CartPole)
# ─────────────────────────────────────────────────────────────────────────────

MUZERO_CONFIG = {
    # Networks
    "hidden_dim":       64,         # 256 in paper; 64 here for speed
    "n_layers":         2,          # Residual layers

    # MCTS
    "n_simulations":    50,         # Paper: 50 (Atari), 800 (Go)
    "c_init":           1.25,       # PUCT c_init
    "c_base":           19652.0,    # PUCT c_base
    "temperature":      1.0,        # Action selection temperature (training)
    "temp_threshold":   30,         # Episode steps after which temp → 0

    # Training
    "lr":               3e-4,       # Paper: 0.05 / 5e-4 (schedule)
    "weight_decay":     1e-4,
    "batch_size":       128,
    "K":                5,          # Unroll steps (paper: 5)
    "n_step":           20,         # n-step return (paper: 10 Atari)
    "gamma":            0.997,      # Discount (paper: 0.997 Atari)
    "grad_clip":        1.0,        # Gradient clipping

    # Loop
    "max_games":        1000,
    "prefill_games":    50,        # Random games before training
    "updates_per_ep":   20,        # Gradient steps after each episode
    "eval_interval":    5,
}


# ─────────────────────────────────────────────────────────────────────────────
# MuZero Agent
# ─────────────────────────────────────────────────────────────────────────────

class MuZeroAgent:
    def __init__(self, env_id: str, config: dict = MUZERO_CONFIG, seed: int = 0):
        self.env_id = env_id
        self.cfg    = config
        self.seed   = seed
        self.device = get_device(verbose=False)

        # ── Environment ──────────────────────────────────────────────────────
        env           = gym.make(env_id)
        obs_dim       = env.observation_space.shape[0]
        n_actions     = env.action_space.n
        env.close()

        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        console.print(f"  [cyan]MuZero Agent · {env_id}[/cyan]")
        console.print(f"  obs_dim={obs_dim}  n_actions={n_actions}  device={self.device}")

        # ── Networks ─────────────────────────────────────────────────────────
        h_dim = config["hidden_dim"]
        self.repr_net = RepresentationNetwork(obs_dim, h_dim, config["n_layers"]).to(self.device)
        self.dyn_net  = DynamicsNetwork(h_dim, n_actions, config["n_layers"]).to(self.device)
        self.pred_net = PredictionNetwork(h_dim, n_actions, config["n_layers"]).to(self.device)

        # Shared optimiser for all three networks
        all_params = (
            list(self.repr_net.parameters()) +
            list(self.dyn_net.parameters()) +
            list(self.pred_net.parameters())
        )
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # ── MCTS ─────────────────────────────────────────────────────────────
        self.mcts = MCTS(
            n_actions=n_actions,
            dynamics_fn=self._dynamics_fn,
            predict_fn=self._predict_fn,
            n_simulations=config["n_simulations"],
            gamma=config["gamma"],
            c_init=config["c_init"],
            c_base=config["c_base"],
            device=self.device,
        )

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.buffer = MuZeroReplayBuffer(
            max_games=config["max_games"],
            obs_dim=obs_dim,
            n_actions=n_actions,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Network function wrappers (for MCTS callbacks)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _dynamics_fn(self, h: torch.Tensor, action_oh: torch.Tensor):
        return self.dyn_net(h, action_oh)

    @torch.no_grad()
    def _predict_fn(self, h: torch.Tensor):
        return self.pred_net(h)

    @torch.no_grad()
    def _value_fn(self, obs: torch.Tensor) -> torch.Tensor:
        """Bootstrap value for n-step return target construction."""
        h = self.repr_net(obs)
        _, v = self.pred_net(h)
        return v

    # ─────────────────────────────────────────────────────────────────────────
    # Self-play — collect one episode
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _play_episode(self, env: gym.Env, random: bool = False) -> tuple:
        """
        Play one episode, recording MCTS search statistics.

        Returns: (GameTrajectory, total_reward)
        """
        obs, _ = env.reset()
        game   = GameTrajectory()
        total_r = 0.0
        step    = 0

        while True:
            if random:
                action_idx   = env.action_space.sample()
                policy_target = np.ones(self.n_actions) / self.n_actions
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                h_t   = self.repr_net(obs_t)

                # Temperature schedule: high during early steps
                temp = self.cfg["temperature"] if step < self.cfg["temp_threshold"] else 0
                policy_target, action_idx = self.mcts.improved_policy(h_t, temperature=temp)

            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated

            game.append(Transition(
                observation   = obs.astype(np.float32),
                action        = action_idx,
                reward        = float(reward),
                policy_target = policy_target,
                done          = done,
            ))
            total_r += reward
            obs     = next_obs
            step   += 1

            if done:
                break

        return game, total_r

    # ─────────────────────────────────────────────────────────────────────────
    # Training step — one gradient update
    # ─────────────────────────────────────────────────────────────────────────

    def _update(self) -> dict:
        """
        Sample a batch and compute the MuZero training loss:
          L = Σ_{k=0}^{K} (L_v_k + L_p_k) + Σ_{k=1}^{K} L_r_k

        L_v_k = MSE(v_k, z_k)      value target z_k from n-step bootstrap
        L_p_k = CE(softmax(p_k), π̂_k)  cross-entropy with MCTS policy
        L_r_k = MSE(r̂_k, u_k)     one-step reward target
        """
        K = self.cfg["K"]

        batch = self.buffer.sample_batch(
            batch_size=self.cfg["batch_size"],
            K=K,
            n=self.cfg["n_step"],
            gamma=self.cfg["gamma"],
            value_fn=self._value_fn,
            device=self.device,
        )

        obs_b   = batch["observations"]      # (B, obs_dim)
        acts_b  = batch["actions"]           # (B, K)
        vals_b  = batch["value_targets"]     # (B, K+1)
        pols_b  = batch["policy_targets"]    # (B, K+1, n_actions)
        rews_b  = batch["reward_targets"]    # (B, K)

        # ── Initial state: Representation ─────────────────────────────────────
        h = self.repr_net(obs_b)      # (B, hidden_dim)

        policy_logits, value = self.pred_net(h)   # prediction at k=0
        total_loss = 0.0

        # Value loss k=0
        total_loss += F.mse_loss(value.squeeze(-1), vals_b[:, 0])
        # Policy loss k=0
        total_loss += F.cross_entropy(policy_logits, pols_b[:, 0])

        # ── Unroll K steps ────────────────────────────────────────────────────
        for k in range(K):
            action_oh = to_one_hot(acts_b[:, k], self.n_actions)  # (B, n_actions)
            h, r_hat  = self.dyn_net(h, action_oh)

            # Prediction at k+1
            policy_logits, value = self.pred_net(h)

            # Losses at k+1
            total_loss += F.mse_loss(value.squeeze(-1), vals_b[:, k + 1])
            total_loss += F.cross_entropy(policy_logits, pols_b[:, k + 1])
            total_loss += F.mse_loss(r_hat.squeeze(-1), rews_b[:, k])

        # Normalise by (K+1)
        total_loss = total_loss / (K + 1)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.repr_net.parameters()) +
            list(self.dyn_net.parameters()) +
            list(self.pred_net.parameters()),
            self.cfg["grad_clip"],
        )
        self.optimizer.step()

        return {"loss": total_loss.item()}

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, n_episodes: int = 500, log_path: str | None = None, run_id: str | None = None):
        run_id = run_id or make_run_id()
        if log_path is None:
            log_path = build_log_path("muzero", self.env_id, self.seed, run_id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        console.rule(f"[bold blue]MuZero · {self.env_id}")

        env    = gym.make(self.env_id)
        env.reset(seed=self.seed)
        fields = ["episode", "reward", "loss", "total_games", "wall_time"]

        with ExperimentLogger(log_path, fields) as logger:
            # ── Pre-fill with random episodes ──────────────────────────────────
            console.print(f"  [yellow]Pre-filling with {self.cfg['prefill_games']} random episodes...[/yellow]")
            for _ in range(self.cfg["prefill_games"]):
                game, _ = self._play_episode(env, random=True)
                self.buffer.add_game(game)
            console.print("  [green]✓ Buffer pre-filled[/green]")

            # ── Training ───────────────────────────────────────────────────────
            pbar     = tqdm(range(n_episodes), desc="MuZero", unit="ep")
            rewards  = []
            ep_idx   = 0

            for ep_idx in pbar:
                # Self-play with MCTS
                game, ep_reward = self._play_episode(env, random=False)
                self.buffer.add_game(game)
                rewards.append(ep_reward)

                # Gradient updates
                loss_val = 0.0
                for _ in range(self.cfg["updates_per_ep"]):
                    info     = self._update()
                    loss_val = info["loss"]

                # Logging
                if ep_idx % self.cfg["eval_interval"] == 0:
                    mean_r = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
                    logger.log(
                        episode=ep_idx,
                        reward=ep_reward,
                        loss=loss_val,
                        total_games=len(self.buffer),
                    )
                    pbar.set_postfix({
                        "reward": f"{mean_r:.1f}",
                        "loss":   f"{loss_val:.4f}",
                    })

        env.close()

        # Save
        model_path = build_custom_model_path(
            algorithm="muzero",
            env_id=self.env_id,
            seed=self.seed,
            run_id=run_id,
            extension="pt",
        )
        torch.save({
            "repr_net": self.repr_net.state_dict(),
            "dyn_net":  self.dyn_net.state_dict(),
            "pred_net": self.pred_net.state_dict(),
        }, model_path)

        console.print(f"\n  [green]✓ Saved → {model_path}[/green]")
        console.print(f"  [green]✓ Log   → {log_path}[/green]")

        return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("logs",    exist_ok=True)
    os.makedirs("results", exist_ok=True)

    agent   = MuZeroAgent("CartPole-v1")
    rewards = agent.train(n_episodes=300)
    console.print(f"\nFinal mean reward (last 20): {np.mean(rewards[-20:]):.1f}")
