#!/bin/bash
# setup.sh — Environment setup for RL-experiments on macOS
# ─────────────────────────────────────────────────────────
# Supports Apple Silicon (M1/M2/M3/M4) MPS GPU acceleration.
# Run this once before first use: bash setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   RL Experiments — macOS Setup                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check Python ──────────────────────────────────────────
PYTHON=$(which python3)
PY_VER=$($PYTHON --version 2>&1)
echo "✓ Python: $PY_VER  ($PYTHON)"

# ── Check pip ─────────────────────────────────────────────
PIP=$(which pip3)
echo "✓ pip:    $($PIP --version)"
echo ""

# ── Install PyTorch (MPS support needs >= 2.x) ───────────
echo "📦 Installing PyTorch with MPS support..."
pip3 install --upgrade torch torchvision torchaudio

# ── Install remaining requirements ────────────────────────
echo ""
echo "📦 Installing remaining requirements..."
pip3 install --upgrade -r requirements.txt

# ── Box2D (LunarLander) ───────────────────────────────────
echo ""
echo "📦 Installing Box2D for LunarLander environments..."
# Install swig if not present (needed to compile Box2D)
if ! command -v swig &>/dev/null; then
    echo "  swig not found. Install it with: brew install swig"
    echo "  Then re-run: pip install gymnasium[box2d]"
else
    pip3 install gymnasium[box2d] --quiet
    echo "  ✓ Box2D / LunarLander installed"
fi

# ── Verify MPS ────────────────────────────────────────────
echo ""
echo "🔍 Checking Mac GPU (MPS) availability..."
$PYTHON - <<'PYEOF'
import torch
print(f"  PyTorch version : {torch.__version__}")
if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps")
    print(f"  ✅ MPS GPU ACTIVE — tensor on {x.device}")
else:
    print("  ⚠️  MPS not available — will run on CPU")
    print("     (Requires Apple Silicon + PyTorch >= 2.0 + macOS >= 12.3)")
PYEOF

# ── Done ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✓ Setup complete!                                   ║"
echo "║                                                      ║"
echo "║  Quick start:                                        ║"
echo "║    python run_all.py --device          # check GPU  ║"
echo "║    python run_all.py --quick --phase 1 # smoke test ║"
echo "║    python run_all.py --phase 1         # Phase 1    ║"
echo "║    python run_all.py                   # run all    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
