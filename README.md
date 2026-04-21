# RL Experiments on Apple Silicon

Train, compare, and visualize modern Reinforcement Learning algorithms with a clean phase-based workflow.

This repository includes:

- Baseline model-free methods (`PPO`, `SAC`, `DQN`) and DQN-family variants (`Double DQN`, `PER-DQN`, `Rainbow`)
- Advanced custom PyTorch agents (`Dreamer`, `MuZero`, `PETS`, `MBPO`, `PlaNet`, `TD-MPC/TD-MPC2`, `World Models`, `I2A`, `MVE`, `STEVE`)
- Experiment orchestration (`run_all.py` / `rl-experiments` CLI)
- Agent playback (`view_agent.py` / `rl-view`, `play_model.py` / `rl-play`)
- Plotting and analysis scripts
- Unified Python API: `TrainConfig` + `train()` for every algorithm ([`src/rl_experiments/api/`](src/rl_experiments/api/))

**Paper alignment:** Training code is structured to match canonical algorithms (SB3 for baselines; custom modules for advanced methods), but hyperparameters and architectures are often **scaled for small envs** rather than full Atari / DMControl benchmarks. See [`docs/algorithm_fidelity.md`](docs/algorithm_fidelity.md) for a concise audit.

**Deep algorithm docs:** Academic-style notes (intuition, equations, architecture, and code anchors) for every registered algorithm are in [`docs/algorithms/index.md`](docs/algorithms/index.md), with Mermaid diagrams and citations.

## Features

- Phase-based experiment runner (train, compare, advanced, plots)
- Apple Silicon friendly setup (PyTorch MPS-aware device selection)
- Per-run artifact isolation in `logs/<run_id>/` and `results/<run_id>/`
- CSV metric logging and TensorBoard logging
- Interactive model discovery and replay viewer

## Repository Structure

```text
RL-experiments/
├── src/rl_experiments/ # Installable package (baselines, advanced, utils, analysis, api, cli, playback)
├── tests/              # Pytest smoke tests
├── docs/               # Extended documentation
├── run_all.py          # Thin wrapper → rl_experiments.cli.run_all
├── view_agent.py       # Thin wrapper → rl_experiments.cli.view_agent
├── play_model.py       # Thin wrapper → rl_experiments.cli.play_model
├── pyproject.toml      # Package metadata + optional dev deps
└── requirements.txt    # Legacy flat list (optional; prefer pyproject)
```

## Setup

1) Clone and enter the repo:

```bash
git clone <your-repo-url>
cd RL-experiments
```

1) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

1) Install the package in editable mode (recommended):

```bash
pip install -e ".[dev]"
```

This installs console scripts `rl-experiments`, `rl-play`, and `rl-view`, and adds `rl_experiments` to your Python path so you do not need `PYTHONPATH` hacks.

Alternatively, install dependencies only:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run a quick smoke test (reduced timesteps):

```bash
python run_all.py --phase 1 --quick --seeds 0
# same: rl-experiments --phase 1 --quick --seeds 0
# same: python -m rl_experiments --phase 1 --quick --seeds 0
```

Run full phase workflows:

```bash
# Phase 1: baseline training (PPO, SAC, DQN)
python run_all.py --phase 1

# Phase 2: baseline behavioral analysis/comparison
python run_all.py --phase 2

# Phase 3: advanced algorithms (default: Dreamer, MuZero)
python run_all.py --phase 3

# Phase 3: specific model-based algorithms
python run_all.py --phase 3 --algorithms pets mbpo planet tdmpc tdmpc2 world_models i2a mve steve

# Phase 3: full model-based suite
python run_all.py --phase 3 --all-model-based

# Phase 4: plotting and final comparisons
python run_all.py --phase 4
```

Run all phases:

```bash
python run_all.py
```

Additional options (same flags work with `rl-experiments` / `run_all.py`):

- `--phases 1 3` — run only phases 1 and 3
- `--phase1-include ppo sac dqn` — subset of Phase 1 baselines
- `--strict-phase3` — do not default to Dreamer+MuZero when `--algorithms` is empty

## Viewing Trained Agents

List discovered models:

```bash
python view_agent.py --list
```

Open interactive picker:

```bash
python view_agent.py
```

Replay a specific model:

```bash
python view_agent.py --model "results/<run_id>/ppo_cartpole_v1_seed0_<run_id>.zip" --episodes 5
```

Export replay:

```bash
python view_agent.py --model "results/<run_id>/ppo_cartpole_v1_seed0_<run_id>.zip" --gif
python view_agent.py --model "results/<run_id>/ppo_cartpole_v1_seed0_<run_id>.zip" --video
```

## Output Layout (Per Run)

Each training invocation gets a timestamp `run_id` to avoid overwrites.

- Logs:
  - `logs/<run_id>/<algo>_<env>_seed<seed>_<run_id>.csv`
- Baseline models (SB3):
  - `results/<run_id>/<algo>_<env>_seed<seed>_<run_id>.zip`
- Advanced checkpoints:
  - `results/<run_id>/<algo>/<env>/<algo>_seed<seed>_<run_id>.pt`
- Run metadata:
  - `results/<run_id>/metadata.json` and `logs/<run_id>/metadata.json`

## Main Algorithms

- **PPO**: on-policy clipped policy gradient baseline
- **SAC**: entropy-regularized off-policy actor-critic for continuous control
- **DQN**: value-based method for discrete action spaces
- **Double DQN**: reduced Q overestimation via decoupled action selection/evaluation
- **PER-DQN**: TD-error-prioritized replay with importance-sampling correction
- **Rainbow DQN**: Double + Dueling + PER + n-step + Noisy Nets + C51
- **Dreamer**: model-based latent world model + imagination actor-critic
- **MuZero**: learned dynamics with planning via MCTS
- **PETS**: probabilistic ensemble dynamics with MPC/CEM planning
- **MBPO**: model-based policy optimization with short synthetic rollouts
- **PlaNet**: latent RSSM world model with planning in latent space
- **TD-MPC / TD-MPC2**: latent dynamics with planning + value backup
- **World Models**: VAE + MDN-RNN + controller pipeline
- **I2A**: imagination-augmented policy with learned rollout model
- **MVE / STEVE**: model-based value expansion and uncertainty-weighted targets

## Programmatic training (unified API)

```python
from rl_experiments.api.training import TrainConfig, train

train(TrainConfig(
    "ppo",
    "CartPole-v1",
    seed=0,
    run_id="20260101_120000",
    budget_steps=200_000,
))
```

Use `train()` with any registered algorithm id (`ppo`, `sac`, `dqn`, `dreamer`, `pets`, …). See [`src/rl_experiments/api/registry.py`](src/rl_experiments/api/registry.py).

## Notes

- Some environments (e.g., Box2D variants) may require local system packages.
- `view_agent` / `rl-view` discovers `.zip` models recursively under `results/`.
- Advanced Dreamer/MuZero checkpoints are `.pt` files; use `play_model` / `rl-play` for those.

## Documentation

See the `docs/` folder for more details:

- `docs/index.md`
- `docs/usage.md`
- `docs/baselines.md`
- `docs/advanced.md`

## License

This project is licensed under the MIT License. See `LICENSE`.
