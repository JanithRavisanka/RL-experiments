# Usage Guide

This project features two main entry points for running training sequences and viewing results.

## 1. `run_all.py` - Master Training Runner

`run_all.py` acts as the command center for running structured experiments. It builds out the logs, saves the models, and coordinates different phases.

Each training invocation now gets a timestamp-based run ID in artifact names, so repeated runs do not overwrite previous outputs.
Example pattern:
- models: `results/<run_id>/<algo>_<env>_seed<seed>_<run_id>.zip` (SB3) or `results/<run_id>/<algo>/<env>/<algo>_seed<seed>_<run_id>.pt` (Dreamer/MuZero)
- logs: `logs/<run_id>/<algo>_<env>_seed<seed>_<run_id>.csv`

**Basic Usage:**
```bash
python run_all.py
```
*(This runs the entire suite, which can take hours!)*

**Targeted execution:**
You can run specific phases using flags:
```bash
# Phase 1 only (Train standard baseline algorithms: PPO, SAC, DQN)
python run_all.py --phase 1

# Phase 2 only (Generate Behavioral Analysis multi-seed plots)
python run_all.py --phase 2

# Phase 3 only (Train Advanced algorithms: Dreamer & MuZero)
python run_all.py --phase 3

# Phase 4 only (Generate Final Paradigm Comparison plots)
python run_all.py --phase 4
```

**Quick Smoke-Test:**
To ensure everything works without waiting for full convergence, use the `--quick` flag. It scales down the number of timesteps/episodes.
```bash
python run_all.py --phase 1 --quick
```

## 2. `view_agent.py` - Watch the Agents Play!

Once models are trained and saved to `results/`, you can use `view_agent.py` to watch them perform inside a PyGame window or save their gameplay to video files.

**Interactive Mode:**
Simply run the script with no arguments. It will list all available generated `.zip` models and prompt you to pick one.
```bash
python view_agent.py
```

**Direct Mode:**
Target a specific model directly.
```bash
python view_agent.py --model results/ppo_cartpole_v1_seed0.zip
```

**Run specific number of episodes:**
```bash
python view_agent.py --model results/ppo_cartpole_v1_seed0.zip --episodes 5
```

**Export to GIF or MP4:**
If you want to save the output instead of opening a live window (useful for presentations):
```bash
python view_agent.py --model results/ppo_cartpole_v1_seed0.zip --gif
python view_agent.py --model results/ppo_cartpole_v1_seed0.zip --video
```
*Note: Video export requires `imageio[ffmpeg]`. GIFs use standard `Pillow`.* Outputs are saved to `analysis/`.
