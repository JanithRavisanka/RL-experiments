"""Shared imagined-rollout objective for CEM planning (PETS, TD-MPC)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from rl_experiments.advanced.common.models import EnsembleDynamics


def cem_score_ensemble_dynamics(
    dynamics: EnsembleDynamics,
    obs_np: np.ndarray,
    obs_dim: int,
    device: torch.device,
    action_seq: list,
    discount: float,
    value_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    """
    Return discounted sum of predicted rewards under ensemble dynamics,
    optionally bootstrapping with ``value_fn(s)`` at the final state (TD-MPC).
    """
    s = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    ret = 0.0
    g = 1.0
    for a in action_seq:
        a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
        pred = dynamics(s, a_t).mean(dim=0)
        ds = pred[:, :obs_dim]
        r = pred[:, -1]
        s = s + ds
        ret += g * float(r.item())
        g *= discount
    if value_fn is not None:
        ret += g * float(value_fn(s).item())
    return ret
