"""Launch the Streamlit training UI (`rl-train-ui`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).resolve().parent.parent / "ui" / "train_app.py"
    if not app.is_file():
        raise SystemExit(f"Missing UI file: {app}")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))


if __name__ == "__main__":
    main()
