#!/usr/bin/env python3
"""Play / evaluate a saved checkpoint (.zip SB3 or Dreamer/MuZero .pt)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_experiments.playback.rollout import play_checkpoint


def main():
    p = argparse.ArgumentParser(description="Play / evaluate a saved RL checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .zip or .pt")
    p.add_argument("--env", type=str, default=None, help="Gymnasium env id (inferred from path if possible)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument(
        "--render",
        type=str,
        default="human",
        choices=["human", "none"],
        help="human = pygame window; none = no display (faster)",
    )
    args = p.parse_args()
    play_checkpoint(Path(args.checkpoint), args.env, args.episodes, args.render)


if __name__ == "__main__":
    main()
