#!/usr/bin/env python3
"""Thin wrapper — prefer ``pip install -e .``; falls back to adding ``src/``."""

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

from rl_experiments.cli.view_agent import main

if __name__ == "__main__":
    main()
