"""VAE for World Models (Ha & Schmidhuber-style pixel pipeline)."""

from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 9 * 9, z_dim)
        self.fc_lv = nn.Linear(64 * 9 * 9, z_dim)
        self.dec_fc = nn.Linear(z_dim, 64 * 9 * 9)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 4, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.enc(x / 255.0)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        d = self.dec_fc(z).view(-1, 64, 9, 9)
        xr = self.dec(d)
        return xr, mu, lv, z
