from __future__ import annotations

import numpy as np


def cem_plan(score_fn, horizon: int, action_dim: int, n_samples: int = 256, n_iters: int = 5, elite_frac: float = 0.1):
    mu = np.zeros((horizon, action_dim), dtype=np.float32)
    sigma = np.ones((horizon, action_dim), dtype=np.float32)
    n_elite = max(1, int(n_samples * elite_frac))

    for _ in range(n_iters):
        samples = np.random.randn(n_samples, horizon, action_dim).astype(np.float32) * sigma + mu
        scores = np.asarray([score_fn(a) for a in samples], dtype=np.float32)
        elite_ids = np.argsort(scores)[-n_elite:]
        elite = samples[elite_ids]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-4
    return mu[0]

