"""
advanced/muzero/mcts.py
─────────────────────────────────────────────────────────────────────────────
MCTS — Monte Carlo Tree Search for MuZero
Schrittwieser et al. 2020

MuZero's MCTS differs from AlphaZero's classical MCTS in key ways:
  1. Uses LEARNED dynamics (not a real environment simulator)
  2. Uses LEARNED reward and value (not game outcomes)
  3. Policy prior comes from the Prediction network
  4. Uses PUCT (Predictor + UCB for Trees) as the selection criterion

MCTS Algorithm (Algorithm 2 from the paper):

  1. SELECTION:
     Traverse tree from root, selecting child that maximises PUCT score:
       PUCT(s,a) = Q(s,a) + C(s) · P(s,a) · √N(s) / (1 + N(s,a))
     where C(s) = c₁ · log((N(s)+c₂+1)/c₂) + c_init
     (c₁=1.25, c₂=19652, c_init depends on min/max tracking)

  2. EXPANSION:
     At a leaf node: use Dynamics g_θ to expand → get r, h', then
     use Prediction p_θ to get policy p and value v.

  3. BACKUP:
     Back up the value through the tree (tracked per node).
     Value targets use the BOOTSTRAPPED value from the network:
     G = r + γ · v(h')

  Values are normalised to [0,1] using min-max across the tree (paper §B).
"""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Tree node
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    """A single node in the MCTS tree."""
    prior:          float           # P(s, a) from Prediction network
    h:              Optional[torch.Tensor] = None  # Hidden state at this node

    visit_count:    int   = 0
    value_sum:      float = 0.0
    reward:         float = 0.0     # Predicted reward to reach this node
    children:       Dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def expand(self, n_actions: int, policy_probs: np.ndarray, h: torch.Tensor):
        """Expand this node: create all children with policy priors."""
        self.h = h
        for a in range(n_actions):
            self.children[a] = Node(prior=float(policy_probs[a]))


# ─────────────────────────────────────────────────────────────────────────────
# Min-max value tracker for normalisation
# ─────────────────────────────────────────────────────────────────────────────

class MinMaxStats:
    """Tracks min/max Q-values for normalisation to [0,1] in PUCT."""

    def __init__(self):
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float):
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def normalize(self, value: float) -> float:
        if self.max > self.min:
            return (value - self.min) / (self.max - self.min)
        return value


# ─────────────────────────────────────────────────────────────────────────────
# MCTS
# ─────────────────────────────────────────────────────────────────────────────

class MCTS:
    """
    Monte Carlo Tree Search as described in Schrittwieser et al. 2020.

    Args:
        n_actions   : Number of discrete actions
        dynamics_fn : g_θ(h, a) → (h_next, r)
        predict_fn  : p_θ(h) → (policy_logits, value)
        n_simulations: Number of MCTS simulations per step (paper: 50 Atari, 800 Go)
        gamma       : Discount factor
        c_init      : PUCT exploration constant (paper: 1.25)
        c_base      : PUCT base constant      (paper: 19652)
        device      : PyTorch device
    """

    def __init__(
        self,
        n_actions:     int,
        dynamics_fn,
        predict_fn,
        n_simulations: int   = 50,
        gamma:         float = 0.99,
        c_init:        float = 1.25,
        c_base:        float = 19652.0,
        device:        torch.device = torch.device("cpu"),
    ):
        self.n_actions    = n_actions
        self.dynamics_fn  = dynamics_fn
        self.predict_fn   = predict_fn
        self.n_sims       = n_simulations
        self.gamma        = gamma
        self.c_init       = c_init
        self.c_base       = c_base
        self.device       = device

    # ─────────────────────────────────────────────────────────────────────────
    # PUCT selection criterion
    # ─────────────────────────────────────────────────────────────────────────

    def _ucb_score(
        self,
        parent:   Node,
        child:    Node,
        minmax:   MinMaxStats,
    ) -> float:
        """
        PUCT score (Predictor + UCB for Trees):

          PUCT = Q_norm + P(s,a) · C(s) · √N(s) / (1 + N(s,a))

          C(s) = log((N(s) + c_base + 1) / c_base) + c_init
        """
        pb_c = (
            math.log((parent.visit_count + self.c_base + 1) / self.c_base)
            + self.c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = minmax.normalize(child.value) if child.visit_count > 0 else 0.0
        return prior_score + value_score

    # ─────────────────────────────────────────────────────────────────────────
    # Single simulation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _simulate(self, root: Node, minmax: MinMaxStats) -> None:
        """Run one simulation: Selection → Expansion → Backup."""
        node       = root
        path: List[Tuple[Node, int]] = []  # (parent, action) pairs
        search_depth = 0

        # 1. SELECTION: traverse down until an unexpanded node
        while node.is_expanded():
            scores = {
                a: self._ucb_score(node, child, minmax)
                for a, child in node.children.items()
            }
            action = max(scores, key=scores.__getitem__)
            path.append((node, action))
            node   = node.children[action]
            search_depth += 1

        # 2. EXPANSION: use dynamics and prediction networks
        parent, action = path[-1] if path else (None, None)

        if parent is not None:
            # Dynamics: (parent_h, action) → (h_next, reward)
            parent_h = parent.h
            a_onehot = torch.zeros(1, self.n_actions, device=self.device)
            a_onehot[0, action] = 1.0
            h_next, r_hat = self.dynamics_fn(parent_h, a_onehot)
            node.reward = r_hat.squeeze().item()
        else:
            h_next = node.h  # root node already has h

        # Prediction: h → (policy, value)
        policy_logits, value = self.predict_fn(h_next)
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
        value_scalar = value.squeeze().item()

        # Expand node
        node.expand(self.n_actions, policy_probs, h_next)

        # 3. BACKUP: propagate value up the tree
        self._backup(path, node, value_scalar, minmax)

    def _backup(
        self,
        path: List[Tuple[Node, int]],
        leaf: Node,
        leaf_value: float,
        minmax: MinMaxStats,
    ) -> None:
        """
        Back-prop bootstrapped value up the path.
        G_t = r_{t+1} + γ · G_{t+1}
        """
        G = leaf_value
        for parent, action in reversed(path):
            child    = parent.children[action]
            G        = child.reward + self.gamma * G
            child.visit_count  += 1
            child.value_sum    += G
            minmax.update(child.value)

        # Update root
        leaf.visit_count += 1
        leaf.value_sum   += G

    # ─────────────────────────────────────────────────────────────────────────
    # Public: run MCTS from an initial hidden state
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, root_h: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        Run MCTS for n_simulations starting from hidden state root_h.

        Returns:
          visit_counts : (n_actions,) array — for constructing improved target policy
          best_action  : int — action with highest visit count (greedy)
        """
        # Initialise root node
        policy_logits, value = self.predict_fn(root_h)
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()

        root = Node(prior=0.0)
        root.expand(self.n_actions, policy_probs, root_h)

        # Add Dirichlet noise to root priors (for exploration — paper §C)
        alpha = 0.25   # Dirichlet α (paper uses 0.3 for Go, 0.15 for chess)
        frac  = 0.25   # fraction of noise to mix in
        noise = np.random.dirichlet([alpha] * self.n_actions)
        for a, child in root.children.items():
            child.prior = (1 - frac) * child.prior + frac * noise[a]

        minmax = MinMaxStats()

        # Run simulations
        root.visit_count = 1
        for _ in range(self.n_sims):
            self._simulate(root, minmax)

        # Extract visit counts
        visits = np.array([
            root.children[a].visit_count for a in range(self.n_actions)
        ], dtype=np.float32)

        best_action = int(visits.argmax())
        return visits, best_action

    def improved_policy(self, root_h: torch.Tensor, temperature: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Run MCTS and return softmax-temperature-adjusted visit counts as
        the improved policy target π̂(a|s).

        temperature=1.0 during training; 0 (argmax) for evaluation.
        """
        visits, best_action = self.run(root_h)
        if temperature == 0:
            probs = np.zeros_like(visits)
            probs[best_action] = 1.0
        else:
            v = visits ** (1.0 / temperature)
            probs = v / v.sum()
        return probs, best_action
