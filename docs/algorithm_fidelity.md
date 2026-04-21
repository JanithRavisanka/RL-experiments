# Algorithm fidelity vs original papers

This document records how each method in this repository relates to its canonical reference. **None of the code paths reproduce full large-scale benchmarks** (e.g. Atari 200M frames, DMControl from pixels) without adaptation: the focus is **correct algorithmic structure** on **low-dimensional / classic** Gymnasium tasks, with hyperparameters scaled for speed and stability.

Use this as the authoritative statement of **what is paper-aligned** versus **what is intentionally simplified**.

For shared MDP notation and the **model-based theory appendix** (compounding error, ensembles), see [`algorithms/00_notation_and_conventions.md`](algorithms/00_notation_and_conventions.md) and [`algorithms/theoretical_appendix_model_based.md`](algorithms/theoretical_appendix_model_based.md).

---

## Summary table

| Algorithm | Primary reference | Implementation basis | Fidelity notes |
|-----------|-------------------|----------------------|----------------|
| PPO | Schulman et al., 2017 | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) `PPO` | Core hyperparameters match common OpenAI-style defaults (clip 0.2, GAE λ=0.95, vf_coef 0.5, 64×64 tanh MLP). Vectorized envs (4×) are an implementation choice, not from the paper. |
| SAC | Haarnoja et al., 2018 | SB3 `SAC` | Table 1–style settings: 256×256 ReLU, lr 3e-4, buffer 1M, Polyak τ=0.005, automatic entropy temperature. |
| DQN | Mnih et al., 2015 (Nature) | SB3 `DQN` | **Adapted for vector obs:** MLP encoder, Adam (paper: RMSProp), buffer 50k (paper: 1M for Atari), ε schedule aligned with SB3 defaults. |
| Double DQN | van Hasselt et al., 2015 | Custom in `dqn_variants.py` | **Decoupled target:** `argmax` on online net, value from target net for non-distributional Q; Huber loss. MLP not CNN. |
| PER-DQN | Schaul et al., 2015 | Custom | Sum-tree priorities, importance sampling with β annealing; same Q backbone as Double when applicable. |
| Rainbow | Hessel et al., 2018 | Custom | Combines Double + Dueling + PER + n-step (3) + Noisy Nets + C51; lr 6.25e-5 and batch 32 match paper order of magnitude for Rainbow. **No Atari frame stack / CNN.** |
| Dreamer | Hafner et al., 2019 / Dreamer-style RSSM | `dreamer_agent.py` | RSSM-style world model + imagination actor–critic; **Appendix B–style dims scaled** for CartPole-class tasks, not full image Dreamer. |
| MuZero | Schrittwieser et al., 2020 | `muzero_agent.py` | Representation / dynamics / prediction + MCTS + unrolled losses; **hidden width, simulations, and schedules reduced** vs Nature (see config comments). |
| PETS | Chua et al., 2018 | `pets_agent.py` | Ensemble dynamics + CEM planning; ensemble size and horizons are **smaller** than full paper runs. |
| MBPO | Janner et al., 2019 | `mbpo_agent.py` | Learned dynamics + short model rollouts + SAC; **not** a line-for-line copy of reference hyperparameters. |
| PlaNet | Hafner et al., 2019 (PlaNet) | `planet_agent.py` | RSSM-like core + pixel encoder path; **simplified planner and data regime** vs full DMControl-from-pixels. |
| TD-MPC / TD-MPC2 | Hansen et al. | `tdmpc_agent.py` | Latent planning flavor; **reference-scale networks and task budgets** not claimed. |
| World Models | Ha & Schmidhuber, 2018 | `world_models_agent.py` | VAE + MDN-RNN pipeline **simplified** for small envs. |
| I2A | Weber et al., 2017 | `i2a_agent.py` | Imagination-augmented structure **without** full-scale Atari stack. |
| MVE / STEVE | Feinberg et al. 2018 / Buckman et al. 2018 | `mve_steve_agent.py` | Model-based value expansion ideas **implemented in reduced form**. |

---

## SB3 baselines (`ppo_experiment.py`, `sac_experiment.py`, `dqn_experiment.py`)

- **Trust boundary:** Training dynamics are those of **Stable-Baselines3**, which implements standard objectives (clipped surrogate for PPO, twin Q and entropy for SAC, Huber TD for DQN).
- **Documented deviations in code:** DQN uses an MLP and Adam for vector environments; Nature DQN used CNN + RMSProp + large replay for Atari.
- **PPO:** `ent_coef=0.0` is a valid choice for many discrete toy domains; the PPO paper discusses entropy bonus for exploration—raising `ent_coef` (e.g. 0.01) is optional if you want closer alignment with entropy-regularized runs.

---

## Custom DQN family (`dqn_variants.py`)

Verified logic (high level):

- **Double DQN (scalar Q):** `next_a = argmax_a Q_online(s')`; `y = r + γ^n Q_target(s', next_a)` — matches van Hasselt et al.
- **Rainbow (C51):** Double-style action selection on **mean Q** from categorical distribution; **categorical projection** onto fixed atoms (Bellemare et al., 2017) implemented for the bellman target; cross-entropy loss with prioritized weights.
- **Noisy Nets:** Factorized Gaussian noise (Fortunato et al., 2017); ε-greedy disabled when noisy is on.

**Not claimed:** Full Rainbow data pipeline (sticky actions, full Atari preprocessing, 200M frames).

---

## Dreamer (`dreamer_agent.py`)

- References **Hafner et al.** world-model + imagination training.
- Uses **discrete actions** and **low-dim** observations; **not** the full pixel encoder + DreamerV2/V3 training stack from DMControl benchmarks.
- Hyperparameters in `DREAMER_CONFIG` are **scaled** (see file comments): treat as educational, not a reproduction table.

---

## MuZero (`muzero_agent.py`)

- Loss structure follows the paper sketch: unroll representation / dynamics / prediction; MCTS for policy improvement targets.
- `MUZERO_CONFIG` explicitly shrinks **hidden size**, **simulation count**, and training budget vs large-scale MuZero. **Learning-rate schedules** in Nature are not fully replicated.

---

## Model-based / planning agents (PETS, MBPO, PlaNet, TD-MPC, World Models, I2A, MVE/STEVE)

These modules **encode the right qualitative loop** (ensemble or latent model, planning horizon, policy learning) but use **smaller networks, shorter horizons, fewer particles / samples, and lighter replay** than full paper experiments. Treat them as **research teaching code**, not leaderboard submissions.

---

## How to tighten alignment (if you need it)

1. **Baselines:** Compare your SB3 version to [SB3 docs](https://stable-baselines3.readthedocs.io/) and bump buffer sizes / network depth toward paper or benchmark scripts (e.g. Atari runners).
2. **Rainbow / DQN variants:** Switch to CNN + frame stack and paper replay size when using image observations.
3. **Dreamer / MuZero:** Increase `hidden_dim`, `chunk_len`, episodes, and (for MuZero) `n_simulations` toward published appendices; add image observations if matching DMControl results.
4. **Always** log seeds and match evaluation protocol (deterministic vs stochastic policy) when comparing to published numbers.

---

## Changelog

- Added with repository audit: clarifies that “paper-faithful” in file comments means **structural alignment and cited hyperparameter families**, not guaranteed numerical match to every benchmark in the original work.
