# Modern RL Exploration on Apple Silicon

Welcome to the **RL Experiments** documentation. This project is a comprehensive toolkit for training, viewing, and analyzing various Reinforcement Learning (RL) algorithms natively on macOS, heavily utilizing PyTorch's MPS (Metal Performance Shaders) backend for Apple Silicon GPUs.

## Project Structure

The codebase is organized into four main phases, modeled as different modules:

- **`baselines/`** (Phase 1 & 2)
  Contains baseline implementations using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
  - **PPO** (Proximal Policy Optimization) - Stable, on-policy model-free RL.
  - **SAC** (Soft Actor-Critic) - Off-policy, entropy-regularized continuous control.
  - **DQN** (Deep Q-Network) - Off-policy, value-based discrete control.

- **`advanced/`** (Phase 3)
  Contains custom, from-scratch PyTorch implementations of advanced, state-of-the-art algorithms optimized for MPS.
  - **Dreamer** (Model-Based RL): Learns a latent world model (RSSM) and trains the actor-critic in "imagination".
  - **MuZero** (Planning-Based RL): Uses a learned dynamics model coupled with Monte Carlo Tree Search (MCTS) to plan ahead without needing the real environment rules.

- **`experiments/`**
  Scripts to run large-scale comparative multi-seed experiments.
  - `compare_phase1.py` - Runs variance analysis for baselines.
  - `run_advanced.py` - Runs Dreamer/MuZero experiments.

- **`analysis/`**
  Houses scripts to generate publication-quality dark-themed plots comparing learning curves, sample efficiency, and paradigm differences.

- **`utils/`**
  Common helpers, custom CSV loggers, and hardware device auto-detection logic (`device_utils.py`).

## Quick Navigation

- [Setup Instructions](setup.md)
- [How to Run & View Agents](usage.md)
- [Baseline Algorithms (Model-Free)](baselines.md)
- [Advanced Algorithms (Model-Based & Planning)](advanced.md)
- [Analysis & Plotting](analysis.md)
