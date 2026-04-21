#!/usr/bin/env python3
"""Thin wrapper to launch the Streamlit training UI (adds ``src/`` if needed)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    try:
        import rl_experiments  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(_SRC))

from rl_experiments.cli.train_ui import main

if __name__ == "__main__":
    main()
