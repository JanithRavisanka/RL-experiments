"""
advanced/dreamer/actor_critic.py
─────────────────────────────────────────────────────────────────────────────
Dreamer Actor-Critic trained entirely inside the world model's imagination.

Architecture — Hafner et al. 2019 / 2021 / 2023:

  Actor  π_φ(a | h, z):
    • Input: feat = cat(h, z)
    • MLP 4×300 ELU (DreamerV3 uses 4 layers × 512 units; reduced here for speed)
    • Continuous: outputs mean + std of a squashed Gaussian
    • Discrete:   outputs logits over action categories

  Critic V_ψ(h, z):
    • Input: feat = cat(h, z)
    • Same architecture as actor
    • Scalar output: predicted value

Training in imagination (Dreamer §3):
  • Roll out H=15 imagined steps from a start state (h, z)
  • Compute λ-return (TD-λ with discount γ=0.99, λ=0.95):
      R_t^λ = r_t + γ·((1-λ)·V_t+1 + λ·R_{t+1}^λ)
  • Actor loss:  -E[Σ_t R_t^λ]  (maximise returns)
  • Critic loss:  E[Σ_t (R_t^λ - V_t)²]  (MSE regression)
  • DreamerV3 adds: percentile return normalisation (drop for simplicity)
  • Entropy regularisation on actor: +η·H[π]

DreamerV3 target critic (stop-gradient slow-update copy) also included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Categorical
from .rssm import MLP


# ─────────────────────────────────────────────────────────────────────────────
# Actor
# ─────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Dreamer Actor π_φ(a | feat).

    For discrete actions:   outputs softmax over action_dim categories.
    For continuous actions: outputs squashed Gaussian (tanh + Normal).
    """

    def __init__(
        self,
        feat_dim:      int,
        action_dim:    int,
        hidden_dim:    int = 300,
        n_layers:      int = 4,
        continuous:    bool = False,
        min_std:       float = 0.1,
        init_std:      float = 5.0,
    ):
        super().__init__()
        self.continuous  = continuous
        self.action_dim  = action_dim
        self.min_std     = min_std

        hidden_dims = [hidden_dim] * (n_layers - 1)
        out_dim     = action_dim * 2 if continuous else action_dim
        self.net    = MLP(feat_dim, hidden_dims, out_dim, activation=nn.ELU)

        if continuous:
            # Trainable initial log_std parameter (DreamerV2 trick)
            self.raw_init_std = torch.tensor(
                [torch.log(torch.exp(torch.tensor(init_std)) - 1.0)] * action_dim
            )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Returns a sampled action (for environment interaction or imagination).
        Gradients flow through reparameterised samples.
        """
        out = self.net(feat)

        if self.continuous:
            mean, raw_std = out.chunk(2, dim=-1)
            std  = F.softplus(raw_std) + self.min_std
            dist = Independent(Normal(mean, std), 1)
            action = dist.rsample()
            action = torch.tanh(action)  # squash to (-1, 1)
        else:
            # Straight-through gradient for discrete actions
            logits = out
            probs  = torch.softmax(logits, dim=-1)
            dist   = Categorical(probs=probs)
            action = dist.sample()
            # one-hot for straight-through
            one_hot = F.one_hot(action, self.action_dim).float()
            action  = one_hot + (probs - probs.detach())  # straight-through

        return action

    def distribution(self, feat: torch.Tensor):
        """Return the action distribution (for entropy computation)."""
        out = self.net(feat)
        if self.continuous:
            mean, raw_std = out.chunk(2, dim=-1)
            std = F.softplus(raw_std) + self.min_std
            return Independent(Normal(mean, std), 1)
        else:
            return Categorical(logits=out)


# ─────────────────────────────────────────────────────────────────────────────
# Critic (value function)
# ─────────────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Dreamer Critic V_ψ(feat).
    Predicts scalar value.  Uses ELU MLPs same as actor.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 300, n_layers: int = 4):
        super().__init__()
        hidden_dims = [hidden_dim] * (n_layers - 1)
        self.net    = MLP(feat_dim, hidden_dims, 1, activation=nn.ELU)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)  # (B,) scalar values


# ─────────────────────────────────────────────────────────────────────────────
# λ-return computation
# ─────────────────────────────────────────────────────────────────────────────

def lambda_returns(
    rewards:   torch.Tensor,   # (H, B)   imagined rewards
    values:    torch.Tensor,   # (H+1, B) predicted values (includes bootstrap)
    continues: torch.Tensor,   # (H, B)   continue probability
    gamma:     float = 0.99,
    lam:       float = 0.95,
) -> torch.Tensor:
    """
    Compute λ-returns backwards through the imagined trajectory.

    R_t^λ = r_t + γ·c_t·[(1−λ)·V_{t+1} + λ·R_{t+1}^λ]

    where c_t is the predicted continue probability.

    Returns:
      targets: (H, B)   TD-λ targets for critic regression
    """
    H = rewards.shape[0]
    # Bootstrap with last value
    last_val  = values[-1]          # (B,)
    returns   = []
    R         = last_val

    for t in reversed(range(H)):
        R = rewards[t] + gamma * continues[t] * ((1 - lam) * values[t + 1] + lam * R)
        returns.insert(0, R)

    return torch.stack(returns, dim=0)  # (H, B)


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic training step
# ─────────────────────────────────────────────────────────────────────────────

def actor_critic_loss(
    world_model,
    actor:         Actor,
    critic:        Critic,
    target_critic: Critic,       # slow EMA copy (DreamerV3)
    h_start:       torch.Tensor, # (B, h_dim)
    z_start:       torch.Tensor, # (B, z_dim)
    horizon:       int   = 15,
    gamma:         float = 0.99,
    lam:           float = 0.95,
    ent_coef:      float = 3e-4,  # entropy coefficient
):
    """
    Compute actor and critic losses from imagination rollout.

    Returns:
      actor_loss, critic_loss, info_dict
    """
    # ── Imagine horizon steps ────────────────────────────────────────────────
    with torch.no_grad():
        feats, r_hats, cont_hats = world_model.imagine_ahead(
            h_start, z_start, actor, horizon=horizon
        )

    # ── Critic values (slow target for λ-return stability) ───────────────────
    with torch.no_grad():
        # values shape: (H+1, B) — include h_start/z_start as first value
        feat0  = world_model.feat(h_start, z_start).unsqueeze(0)  # (1, B, F)
        all_f  = torch.cat([feat0, feats], dim=0)                  # (H+1, B, F)
        values = target_critic(all_f.detach())                     # (H+1, B)

    # ── λ-returns ─────────────────────────────────────────────────────────────
    targets = lambda_returns(r_hats, values, cont_hats, gamma=gamma, lam=lam)
    # targets: (H, B)

    # ── Actor loss: -E[returns] + entropy bonus ───────────────────────────────
    # Re-compute actor distributions over imagined feats (with gradients)
    actor_feats = feats.detach()                       # stop WM gradients
    dist        = actor.distribution(actor_feats)      # for entropy
    entropy     = dist.entropy().unsqueeze(-1) if actor_feats.dim() == 3 else dist.entropy()
    # Normalise returns (DreamerV3 §B: percentile normalisation, simplified here)
    norm_targets = (targets - targets.mean()) / (targets.std() + 1e-8)

    actor_loss   = -norm_targets.detach().mean() - ent_coef * entropy.mean()

    # ── Critic loss: MSE or Huber vs λ-returns ────────────────────────────────
    critic_vals  = critic(feats.detach())               # (H, B), gradients OK
    critic_loss  = F.huber_loss(critic_vals, targets.detach())

    info = {
        "mean_return":  targets.mean().item(),
        "mean_reward":  r_hats.mean().item(),
        "entropy":      entropy.mean().item(),
    }

    return actor_loss, critic_loss, info
