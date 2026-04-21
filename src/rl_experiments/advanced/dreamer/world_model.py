"""
advanced/dreamer/world_model.py
─────────────────────────────────────────────────────────────────────────────
Dreamer World Model
Hafner et al., 2019–2023 (DreamerV1/V2/V3)

Components:
  ┌─────────────────────────────────────────────────────────┐
  │  Encoder:    obs_t   → embed_t   (MLP for low-dim obs)  │
  │  RSSM:       embed_t → (h_t, z_t)                       │
  │  Decoder:    (h_t,z_t) → obs_hat  [reconstruction]      │
  │  Reward:     (h_t,z_t) → r_hat   [reward prediction]    │
  │  Continue:   (h_t,z_t) → γ_hat   [discount prediction]  │
  └─────────────────────────────────────────────────────────┘

Loss (total world model loss):
  L_WM = L_rec + β_KL · L_KL + L_rew + L_cont

  where  β_KL = 1.0 (DreamerV3 sets β=1)
         L_rec = -E[log p(obs | h, z)]   MSE for continuous obs
         L_KL  = KL[q(z|h,e) ‖ p(z|h)]  with free bits
         L_rew = -E[log p(r  | h, z)]    MSE for reward
         L_cont= -E[log p(γ  | h, z)]    BCE for continue flag
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rssm import RSSM, MLP


class WorldModel(nn.Module):
    """
    Complete Dreamer World Model for low-dimensional (vector) observations.

    Args:
        obs_dim     : Dimensionality of environment observations
        action_dim  : Dimensionality of action (one-hot for discrete)
        embed_dim   : Encoder output size
        h_dim       : RSSM deterministic hidden size
        z_dim       : RSSM stochastic latent size
        hidden_dim  : Width of MLP hidden layers
        kl_free_bits: Free bits for KL (DreamerV3, default 1.0)
        kl_weight   : β_KL coefficient
    """

    def __init__(
        self,
        obs_dim:      int,
        action_dim:   int,
        embed_dim:    int = 64,
        h_dim:        int = 200,
        z_dim:        int = 30,
        hidden_dim:   int = 200,
        kl_free_bits: float = 1.0,
        kl_weight:    float = 1.0,
    ):
        super().__init__()
        self.obs_dim      = obs_dim
        self.action_dim   = action_dim
        self.h_dim        = h_dim
        self.z_dim        = z_dim
        self.kl_free_bits = kl_free_bits
        self.kl_weight    = kl_weight

        # ── Encoder: obs → embed ─────────────────────────────────────────────
        self.encoder = MLP(obs_dim, [hidden_dim], embed_dim)

        # ── RSSM ────────────────────────────────────────────────────────────
        self.rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=action_dim,
            h_dim=h_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
        )

        feat_dim = h_dim + z_dim  # concatenated feature for decoders

        # ── Decoder: (h,z) → obs  (reconstruction) ──────────────────────────
        self.decoder = MLP(feat_dim, [hidden_dim, hidden_dim], obs_dim)

        # ── Reward predictor: (h,z) → r ─────────────────────────────────────
        self.reward_head = MLP(feat_dim, [hidden_dim], 1)

        # ── Continue predictor: (h,z) → Bernoulli(γ) ────────────────────────
        # DreamerV2/V3: predicts whether episode continues (1 - done flag)
        self.continue_head = MLP(feat_dim, [hidden_dim], 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def feat(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate deterministic and stochastic states."""
        return torch.cat([h, z], dim=-1)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward: process a sequence of real transitions
    # ─────────────────────────────────────────────────────────────────────────

    def observe_sequence(
        self,
        obs:     torch.Tensor,    # (T, B, obs_dim)
        actions: torch.Tensor,    # (T, B, action_dim)
        device:  torch.device,
    ):
        """
        Process a batch of real trajectories through the world model.

        Returns:
          feats  : (T, B, h_dim+z_dim) — features for downstream losses
          priors : list[T] of prior distributions
          posts  : list[T] of posterior distributions
        """
        T, B, _ = obs.shape

        # Encode all observations
        embeds = self.encoder(obs.reshape(T * B, -1)).reshape(T, B, -1)

        h0, z0 = self.rssm.initial_state(B, device)
        hs, zs, priors, posts = self.rssm.observe(embeds, actions, h0, z0)

        feats = self.feat(hs, zs)   # (T, B, feat_dim)
        return feats, priors, posts, hs, zs

    # ─────────────────────────────────────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        obs:      torch.Tensor,    # (T, B, obs_dim)
        actions:  torch.Tensor,    # (T, B, action_dim)
        rewards:  torch.Tensor,    # (T, B, 1)
        dones:    torch.Tensor,    # (T, B, 1)  boolean / float
        device:   torch.device,
    ) -> dict:
        """
        Compute all world model losses for one sequence batch.

        Returns a dict of named losses including the total loss.
        """
        feats, priors, posts, hs, zs = self.observe_sequence(obs, actions, device)

        T, B = obs.shape[:2]

        # ── Reconstruction loss ──────────────────────────────────────────────
        obs_hat = self.decoder(feats)   # (T, B, obs_dim)
        loss_rec = F.mse_loss(obs_hat, obs)

        # ── KL divergence: posterior vs prior ───────────────────────────────
        loss_kl = RSSM.kl_loss(priors, posts, free_bits=self.kl_free_bits)

        # ── Reward prediction loss ───────────────────────────────────────────
        r_hat   = self.reward_head(feats).squeeze(-1)   # (T, B)
        loss_rew = F.mse_loss(r_hat, rewards.squeeze(-1))

        # ── Continue prediction (BCE) ─────────────────────────────────────────
        cont_hat = self.continue_head(feats).squeeze(-1)  # (T, B) logit
        cont_target = 1.0 - dones.float().squeeze(-1)     # 1 if not done
        loss_cont = F.binary_cross_entropy_with_logits(cont_hat, cont_target)

        # ── Total loss ───────────────────────────────────────────────────────
        loss_total = loss_rec + self.kl_weight * loss_kl + loss_rew + loss_cont

        return {
            "loss_total":    loss_total,
            "loss_rec":      loss_rec.item(),
            "loss_kl":       loss_kl.item(),
            "loss_rew":      loss_rew.item(),
            "loss_cont":     loss_cont.item(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Imagination rollout — used for actor-critic training
    # ─────────────────────────────────────────────────────────────────────────

    def imagine_ahead(
        self,
        h_init:    torch.Tensor,
        z_init:    torch.Tensor,
        actor,
        horizon:   int = 15,
    ):
        """
        Roll out the world model using the actor policy for `horizon` steps.

        Returns:
          feats   : (H, B, feat_dim)  — latent features at each imagined step
          r_hats  : (H, B)            — predicted rewards
          cont_hats: (H, B)           — predicted continue probabilities
        """
        def action_fn(h, z):
            return actor(self.feat(h, z))

        hs, zs, actions = self.rssm.imagine(
            action_fn=action_fn,
            h_init=h_init,
            z_init=z_init,
            horizon=horizon,
        )

        feats     = self.feat(hs, zs)          # (H, B, feat_dim)
        r_hats    = self.reward_head(feats).squeeze(-1)
        cont_hats = torch.sigmoid(self.continue_head(feats).squeeze(-1))

        return feats, r_hats, cont_hats
