"""Lightweight RSSM core for PlaNet-style training."""

from __future__ import annotations

import torch
import torch.nn as nn


class RSSMCore(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.prior = nn.Linear(latent_dim, latent_dim * 2)
        self.post = nn.Linear(latent_dim * 2, latent_dim * 2)
        self.rew = nn.Linear(latent_dim, 1)

    def step(self, h, z, a, emb=None):
        h = self.gru(torch.cat([z, a], dim=-1), h)
        pm, pv = self.prior(h).chunk(2, dim=-1)
        if emb is None:
            return h, pm, pv, pm, pv
        qm, qv = self.post(torch.cat([h, emb], dim=-1)).chunk(2, dim=-1)
        return h, pm, pv, qm, qv
