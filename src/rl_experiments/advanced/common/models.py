from __future__ import annotations

import torch
import torch.nn as nn


class PixelEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 9 * 9, latent_dim)

    def forward(self, x):
        h = self.conv(x / 255.0)
        h = h.flatten(1)
        return self.fc(h)


class EnsembleDynamics(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, ensemble: int = 5):
        super().__init__()
        self.models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, obs_dim + 1),
                )
                for _ in range(ensemble)
            ]
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        outs = [m(x) for m in self.models]
        return torch.stack(outs, dim=0)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class MLPValue(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

