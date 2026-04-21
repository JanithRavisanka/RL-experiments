# RL Experiments on Apple Silicon

Train, compare, and visualize modern Reinforcement Learning algorithms with a clean phase-based workflow.

This repository includes:

- Baseline model-free methods using Stable-Baselines3 (`PPO`, `SAC`, `DQN`)
- Advanced custom PyTorch agents (`Dreamer`, `MuZero`)
- Experiment orchestration (`run_all.py`)
- Agent playback (`view_agent.py`)
- Plotting and analysis scripts

## Features

- Phase-based experiment runner (train, compare, advanced, plots)
- Apple Silicon friendly setup (PyTorch MPS-aware device selection)
- Per-run artifact isolation in `logs/<run_id>/` and `results/<run_id>/`
- CSV metric logging and TensorBoard logging
- Interactive model discovery and replay viewer

## Repository Structure

```text
RL-experiments/
├── baselines/          # PPO, SAC, DQN experiment scripts
├── advanced/           # Dreamer and MuZero implementations
├── experiments/        # Comparison experiments
├── analysis/           # Plotting and generated figures
├── utils/              # Device helpers, metrics, run path utilities
├── docs/               # Extended documentation
├── run_all.py          # Master runner
└── view_agent.py       # Model viewer / replay exporter
```

## Setup

1) Clone and enter the repo:

```bash
git clone <your-repo-url>
cd RL-experiments
```

2) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3) Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a quick smoke test (reduced timesteps):

```bash
python run_all.py --phase 1 --quick --seeds 0
```

Run full phase workflows:

```bash
# Phase 1: baseline training (PPO, SAC, DQN)
python run_all.py --phase 1

# Phase 2: baseline behavioral analysis/comparison
python run_all.py --phase 2

# Phase 3: advanced algorithms (Dreamer, MuZero)
python run_all.py --phase 3

# Phase 4: plotting and final comparisons
python run_all.py --phase 4
```

Run all phases:

```bash
python run_all.py
```

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

## Main Algorithms

- **PPO**: on-policy clipped policy gradient baseline
- **SAC**: entropy-regularized off-policy actor-critic for continuous control
- **DQN**: value-based method for discrete action spaces
- **Dreamer**: model-based latent world model + imagination actor-critic
- **MuZero**: learned dynamics with planning via MCTS

## Notes

- Some environments (e.g., Box2D variants) may require local system packages.
- `view_agent.py` discovers `.zip` models recursively under `results/`.
- Advanced Dreamer/MuZero checkpoints are `.pt` files and are not loaded by `view_agent.py`.

## Documentation

See the `docs/` folder for more details:

- `docs/index.md`
- `docs/usage.md`
- `docs/baselines.md`
- `docs/advanced.md`

## License

This project is licensed under the MIT License. See `LICENSE`.
