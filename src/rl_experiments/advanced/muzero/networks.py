"""
advanced/muzero/networks.py
─────────────────────────────────────────────────────────────────────────────
MuZero Networks — Schrittwieser et al. 2020
https://www.nature.com/articles/s41586-020-03051-4

MuZero learns THREE networks that together enable planning WITHOUT a
pre-defined environment model:

  1. Representation  h = f_θ(o)
     Maps raw observation o_t → abstract hidden state h_t
     (similar to an encoder)

  2. Dynamics  (r̂, h') = g_θ(h, a)
     Predicts the NEXT hidden state h' and REWARD r̂
     given current hidden state h and a one-hot action a.
     This is the "imagination" engine — allows multi-step planning.

  3. Prediction  (p, v) = p_θ(h)
     Given hidden state h, predicts:
       p : policy logits over all actions (PRIOR for MCTS)
       v : scalar value estimate

These three networks are used together in MCTS to plan a sequence of
imagined actions before choosing the best real action.

Architecture (from Appendix A of the paper):
  - For board games: residual convolutional networks
  - For Atari/simple envs: fully-connected MLPs are used
  - Hidden state size: 256 units (paper), reduced to 64 here for speed

Mac GPU: all modules accept a PyTorch device (mps/cpu).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared building block
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Simple residual block: x → Linear → ReLU → Linear + skip → ReLU.
    Used in all three MuZero networks.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)


class MLP(nn.Module):
    """Flexible MLP with optional residual blocks."""

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        out_dim:    int,
        n_layers:   int = 2,
        use_residual: bool = True,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            if use_residual:
                layers.append(ResidualBlock(hidden_dim))
            else:
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Representation Network  h = f_θ(o)
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationNetwork(nn.Module):
    """
    f_θ: observation → hidden state

    Paper §C: For board games, 16 conv blocks.
              For simple envs, a stack of fully-connected layers.

    Output is normalised to [0,1] per dimension (min-max scale).
    This normalisation from MuZero Appendix A is important for stability.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, hidden_dim, n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.net(obs)
        # Min-max normalise hidden state to [0, 1] (MuZero Appendix A)
        h_min = h.min(dim=-1, keepdim=True).values
        h_max = h.max(dim=-1, keepdim=True).values
        h_norm = (h - h_min) / (h_max - h_min + 1e-8)
        return h_norm


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dynamics Network  (r̂, h') = g_θ(h, a)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicsNetwork(nn.Module):
    """
    g_θ: (hidden state, action) → (next hidden state, reward)

    Action is encoded as a one-hot vector and concatenated with h.

    Paper: Same number of residual blocks as Representation and Prediction.
    The reward head predicts a scalar (MSE loss during training).

    The next hidden state is also min-max normalised (Appendix A).
    """

    def __init__(
        self,
        hidden_dim:  int,
        action_dim:  int,   # number of discrete actions (one-hot size)
        n_layers:    int = 2,
        reward_support: int = 1,  # 1 = scalar; >1 = categorical (MuZero paper uses bins)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Dynamics MLP: cat(h, one_hot(a)) → h'
        self.state_net  = MLP(hidden_dim + action_dim, hidden_dim, hidden_dim, n_layers)

        # Reward head
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, reward_support),
        )

    def forward(
        self, h: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          h      : (..., hidden_dim)
          action : (..., action_dim) one-hot encoded
        Returns:
          h_next : (..., hidden_dim)  min-max normalised
          r_hat  : (..., 1)           scalar reward prediction
        """
        x      = torch.cat([h, action], dim=-1)
        h_next = self.state_net(x)

        # Normalise next state
        h_min  = h_next.min(dim=-1, keepdim=True).values
        h_max  = h_next.max(dim=-1, keepdim=True).values
        h_next = (h_next - h_min) / (h_max - h_min + 1e-8)

        r_hat  = self.reward_net(h_next)
        return h_next, r_hat


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prediction Network  (p, v) = p_θ(h)
# ─────────────────────────────────────────────────────────────────────────────

class PredictionNetwork(nn.Module):
    """
    p_θ: hidden state → (policy logits, value)

    Policy logits: used as MCTS prior (softmax to get probabilities)
    Value: scalar, used to initialise leaf nodes in tree search
    """

    def __init__(
        self,
        hidden_dim:  int,
        n_actions:   int,   # number of discrete actions
        n_layers:    int = 2,
    ):
        super().__init__()
        self.trunk = MLP(hidden_dim, hidden_dim, hidden_dim, n_layers - 1)

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # Value head (scalar)
        self.value_head  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # value in (-1, 1) — matches MuZero for normalised rewards
        )

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          policy_logits : (..., n_actions)
          value         : (..., 1)
        """
        feat   = self.trunk(h)
        policy = self.policy_head(feat)
        value  = self.value_head(feat)
        return policy, value


# ─────────────────────────────────────────────────────────────────────────────
# Utility: one-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

def to_one_hot(actions: torch.Tensor, n_actions: int) -> torch.Tensor:
    """
    Convert integer action tensor to one-hot encoding.
    actions: (...,) int64
    Returns: (..., n_actions) float32
    """
    return F.one_hot(actions.long(), num_classes=n_actions).float()
