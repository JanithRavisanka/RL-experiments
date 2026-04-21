"""
advanced/muzero/replay_buffer.py
─────────────────────────────────────────────────────────────────────────────
MuZero Replay Buffer (Schrittwieser et al. 2020, §B.2)

MuZero stores entire game/episode trajectories.
During training, it samples a (s, a, r) subsequence of length K (unroll steps)
along with the MCTS policy target π̂ and value target z.

Key differences from DQN buffer:
  1. Stores MCTS search statistics (visit counts → target policy π̂)
  2. Bootstrap value targets using n-step returns + value network
  3. Samples are (observation, action sequence [K], policy targets [K],
     value targets [K], reward targets [K])
  4. Priority-based sampling supported (PER) — simplified to uniform here

Paper: K = 5 unroll steps, n = 10 for n-step value bootstrap (Atari)
       Simplified here to K=5, n=20 for CartPole
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    observation:    np.ndarray   # env observation at this step
    action:         int          # action taken
    reward:         float        # env reward received
    policy_target:  np.ndarray   # π̂ from MCTS (search policy)
    done:           bool


@dataclass
class GameTrajectory:
    """Stores one complete episode of MuZero data."""
    transitions: List[Transition] = field(default_factory=list)

    def __len__(self):
        return len(self.transitions)

    def append(self, t: Transition):
        self.transitions.append(t)

    def make_target(
        self,
        pos:     int,
        K:       int,
        n:       int,
        gamma:   float,
        value_fn,   # callable(obs) → float  (bootstrap value)
        device:  torch.device,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[float], List[np.ndarray]]:
        """
        Construct training targets for MuZero at position `pos`.

        Returns:
          observation   : obs at `pos`
          actions       : [K] actions following pos
          value_targets : [K+1] n-step value targets z_t
          policy_targets: [K+1] MCTS improved policies π̂_t
          reward_targets: [K]   one-step rewards r_{t+1}
        """
        T = len(self.transitions)
        observation = self.transitions[pos].observation

        actions       = []
        value_targets = []
        policy_targets = []
        reward_targets = []

        for k in range(K + 1):
            t = pos + k
            if t >= T:
                # Absorbing state — fill with zeros
                actions.append(0)
                value_targets.append(0.0)
                policy_targets.append(np.ones(len(self.transitions[0].policy_target)) /
                                      len(self.transitions[0].policy_target))
                reward_targets.append(0.0)
                continue

            # n-step return for value target
            G = 0.0
            for i in range(n):
                ti = t + i
                if ti >= T:
                    break
                G += (gamma ** i) * self.transitions[ti].reward

            # Bootstrap with value network at t+n
            tn = t + n
            if tn < T and not self.transitions[tn].done:
                obs_boot = self.transitions[tn].observation
                obs_t    = torch.tensor(obs_boot, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    v_boot = value_fn(obs_t).item()
                G += (gamma ** n) * v_boot

            value_targets.append(G)
            policy_targets.append(self.transitions[t].policy_target)

            if k < K:
                actions.append(self.transitions[t].action)
                reward_targets.append(self.transitions[t].reward)

        return observation, actions, value_targets, policy_targets, reward_targets


class MuZeroReplayBuffer:
    """
    Stores game trajectories; supports uniform sampling of (obs, targets) tuples.
    """

    def __init__(self, max_games: int = 1000, obs_dim: int = 4, n_actions: int = 2):
        self.max_games  = max_games
        self.obs_dim    = obs_dim
        self.n_actions  = n_actions
        self._games: List[GameTrajectory] = []
        self._ptr = 0

    def add_game(self, game: GameTrajectory):
        if len(self._games) < self.max_games:
            self._games.append(game)
        else:
            self._games[self._ptr % self.max_games] = game
        self._ptr += 1

    def __len__(self):
        return len(self._games)

    def total_steps(self):
        return sum(len(g) for g in self._games)

    def sample_batch(
        self,
        batch_size:  int,
        K:           int,
        n:           int,
        gamma:       float,
        value_fn,
        device:      torch.device,
    ):
        """
        Sample a batch of (observation, unroll targets) for MuZero training.

        Returns dicts of tensors.
        """
        eligible = [g for g in self._games if len(g) > 1]
        if not eligible:
            raise RuntimeError("Buffer has no games with >1 steps.")

        obs_batch      = []
        act_batch      = []
        val_batch      = []
        pol_batch      = []
        rew_batch      = []

        for _ in range(batch_size):
            game = eligible[np.random.randint(len(eligible))]
            pos  = np.random.randint(0, len(game))

            obs, acts, vals, pols, rews = game.make_target(
                pos=pos, K=K, n=n, gamma=gamma,
                value_fn=value_fn, device=device,
            )
            obs_batch.append(obs)
            act_batch.append(acts[:K])
            val_batch.append(vals[:K + 1])
            pol_batch.append(np.stack(pols[:K + 1]))
            rew_batch.append(rews[:K])

        return {
            "observations":    torch.tensor(np.array(obs_batch),  dtype=torch.float32, device=device),
            "actions":         torch.tensor(np.array(act_batch),  dtype=torch.long,    device=device),
            "value_targets":   torch.tensor(np.array(val_batch),  dtype=torch.float32, device=device),
            "policy_targets":  torch.tensor(np.array(pol_batch),  dtype=torch.float32, device=device),
            "reward_targets":  torch.tensor(np.array(rew_batch),  dtype=torch.float32, device=device),
        }
