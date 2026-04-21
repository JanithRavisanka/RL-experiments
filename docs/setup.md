# Setup & Environment

This project is tailored to work out-of-the-box on macOS (M1/M2/M3/M4) devices.

## Requirements

- macOS 12.3+ (for PyTorch MPS support)
- Python 3.10+
- Homebrew (for installing native dependencies)

## Installation

### 1. Install SWIG
Swig is required to compile Box2D, which is the physics engine behind the `LunarLander` environments.
```bash
brew install swig
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
Update pip and install everything:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio  # PyTorch 2.x
pip install "gymnasium[classic-control,box2d]"
pip install stable-baselines3 pandas matplotlib seaborn rich tqdm tensorboard
```

*(Alternatively, use the provided `setup.sh` script to automate everything!)*
```bash
bash setup.sh
```

## Verifying Apple Silicon GPU (MPS)

Once installed, you can verify that PyTorch is correctly detecting your GPU:
```bash
source venv/bin/activate
python run_all.py --device
```

You should see output indicating that the device is set to `mps`. The project will automatically use this device for all neural network training.
