"""
advanced/dreamer/rssm.py
─────────────────────────────────────────────────────────────────────────────
RSSM — Recurrent State Space Model (core of Dreamer)
Hafner et al., "Dream to Control", 2019 (DreamerV1)
Hafner et al., "Mastering Atari with Discrete World Models", 2021 (DreamerV2)
Hafner et al., "Mastering Diverse Domains with World Models", 2023 (DreamerV3)

The RSSM maintains a hybrid world state:
  h_t  = recurrent (deterministic) GRU hidden state
  z_t  = stochastic latent state  (sampled from a distribution)

For low-dim vector observations (not pixels) we use:
  Encoder  : MLP(obs) → e_t
  Prior    : p(z_t | h_t)           ← used during imagination
  Posterior: q(z_t | h_t, e_t)      ← used during training on real data
  Decoder  : MLP(h_t, z_t) → obs_hat
  Reward   : MLP(h_t, z_t) → r_hat
  Continue : MLP(h_t, z_t) → γ_hat  (DreamerV2+)

DreamerV3 improvements implemented here:
  • Categorical latent (32 categories × 32 classes) instead of Gaussian
  • symlog predictions
  • Free bits KL (minimum KL = 1 free bit per dim)
  • Straight-through gradients through discrete latent

For simplicity / Mac compatibility we use:
  • Gaussian latent (DreamerV2 style) — easier to implement and debug
  • MPS-compatible PyTorch ops

Mac GPU: All modules accept device from caller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Standard MLP with LayerNorm + ELU (DreamerV2/V3 default)."""

    def __init__(self, in_dim: int, hidden_dims, out_dim: int, activation=nn.ELU):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianHead(nn.Module):
    """
    Maps features → (mean, log_std) → Normal distribution.
    Used for stochastic z_t in RSSM.
    """

    def __init__(self, in_dim: int, z_dim: int, min_std: float = 0.1):
        super().__init__()
        self.fc_mean    = nn.Linear(in_dim, z_dim)
        self.fc_log_std = nn.Linear(in_dim, z_dim)
        self.min_std    = min_std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        mean    = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std     = F.softplus(log_std) + self.min_std
        dist    = Independent(Normal(mean, std), 1)
        sample  = dist.rsample()   # reparameterisation trick
        return sample, dist


# ─────────────────────────────────────────────────────────────────────────────
# RSSM
# ─────────────────────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    """
    Recurrent State Space Model.

    State:
      h : deterministic hidden  (GRU hidden)
      z : stochastic latent     (Gaussian sample)

    The full model state is called a "representation": (h, z)
    and the concatenated feature vector feat = cat(h, z).

    For a sequence of length T:
      for t in 1..T:
        h_t = GRU(cat(z_{t-1}, a_{t-1}), h_{t-1})
        z_t ~ posterior q(z|h_t, e_t)         [training]
        z_t ~ prior    p(z|h_t)                [imagination]

    Args:
        obs_dim    : Observation dimension (after encoder)
        action_dim : Action dimension (one-hot or continuous scalar)
        h_dim      : Deterministic hidden size (GRU units)
        z_dim      : Stochastic latent size
        hidden_dim : Hidden layer size in MLPs
    """

    def __init__(
        self,
        embed_dim:  int,
        action_dim: int,
        h_dim:      int = 200,
        z_dim:      int = 30,
        hidden_dim: int = 200,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        # ── GRU: takes (z_{t-1}, a_{t-1}) → updates h_t ────────────────────
        self.gru = nn.GRUCell(z_dim + action_dim, h_dim)

        # ── Prior p(z_t | h_t) ───────────────────────────────────────────────
        self.prior_mlp = MLP(h_dim, [hidden_dim], hidden_dim)
        self.prior_head = GaussianHead(hidden_dim, z_dim)

        # ── Posterior q(z_t | h_t, e_t) ─────────────────────────────────────
        self.post_mlp  = MLP(h_dim + embed_dim, [hidden_dim], hidden_dim)
        self.post_head = GaussianHead(hidden_dim, z_dim)

    def initial_state(self, batch: int, device: torch.device):
        """Return (h_0, z_0) = zeros."""
        h = torch.zeros(batch, self.h_dim, device=device)
        z = torch.zeros(batch, self.z_dim, device=device)
        return h, z

    def observe(
        self,
        embed:   torch.Tensor,   # (T, B, embed_dim) — encoded obs sequence
        actions: torch.Tensor,   # (T, B, action_dim)
        h_prev:  torch.Tensor,   # (B, h_dim)
        z_prev:  torch.Tensor,   # (B, z_dim)
    ):
        """
        Run RSSM on a sequence of real observations.
        Returns posterior z samples (training uses q, imagination uses p).

        Returns:
          hs    : (T, B, h_dim) deterministic states
          zs    : (T, B, z_dim) stochastic states (posterior samples)
          priors: list of T prior Normal distributions
          posts : list of T posterior Normal distributions
        """
        T = embed.shape[0]
        hs, zs, priors, posts = [], [], [], []

        h, z = h_prev, z_prev
        for t in range(T):
            # 1. Deterministic step
            gru_in = torch.cat([z, actions[t]], dim=-1)
            h      = self.gru(gru_in, h)

            # 2. Prior (for KL divergence computation)
            prior_feat    = self.prior_mlp(h)
            z_prior, p_prior = self.prior_head(prior_feat)

            # 3. Posterior (conditioned on embedding)
            post_in       = torch.cat([h, embed[t]], dim=-1)
            post_feat     = self.post_mlp(post_in)
            z_post, p_post = self.post_head(post_feat)

            z = z_post  # use posterior sample for next step (during training)

            hs.append(h)
            zs.append(z)
            priors.append(p_prior)
            posts.append(p_post)

        hs = torch.stack(hs)   # (T, B, h_dim)
        zs = torch.stack(zs)   # (T, B, z_dim)
        return hs, zs, priors, posts

    def imagine(
        self,
        action_fn,          # callable: (h, z) → action  (actor policy)
        h_init: torch.Tensor,
        z_init: torch.Tensor,
        horizon: int,
    ):
        """
        Roll out RSSM using the prior (no real observations).
        Used for actor-critic training inside the world model's head.

        Returns:
          hs      : (H, B, h_dim)
          zs      : (H, B, z_dim)
          actions : (H, B, action_dim)
        """
        h, z = h_init, z_init
        hs, zs, acts = [], [], []

        for _ in range(horizon):
            a = action_fn(h.detach(), z.detach())    # stop gradient (paper §A)
            gru_in = torch.cat([z, a], dim=-1)
            h      = self.gru(gru_in, h)

            prior_feat = self.prior_mlp(h)
            z, _       = self.prior_head(prior_feat)

            hs.append(h)
            zs.append(z)
            acts.append(a)

        return (torch.stack(hs), torch.stack(zs), torch.stack(acts))

    @staticmethod
    def kl_loss(priors, posts, free_bits: float = 1.0) -> torch.Tensor:
        """
        KL divergence: KL[posterior ‖ prior]
        DreamerV3: min KL to 'free_bits' (lower bound to avoid posterior collapse).

        kl = max(KL[q‖p], free_bits)
        """
        if not priors:
            return torch.tensor(0.0)
        # Accumulate on the same device as KL terms (avoid torch.zeros(1) on CPU vs MPS/CUDA).
        kl_sum = None
        for prior, post in zip(priors, posts):
            kl_step = torch.distributions.kl_divergence(post, prior).sum(-1)  # (B,)
            part = kl_step.clamp(min=free_bits).mean()
            kl_sum = part if kl_sum is None else kl_sum + part
        return kl_sum / len(priors)
