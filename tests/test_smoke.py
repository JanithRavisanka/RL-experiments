"""Smoke tests — imports and CLI entry points."""

import subprocess
import sys


def test_import_package():
    import rl_experiments
    from rl_experiments.api.training import TrainConfig, TrainResult, train

    assert rl_experiments.__version__
    assert TrainConfig("ppo", "CartPole-v1", 0, run_id="test").algorithm == "ppo"


def test_run_all_help():
    r = subprocess.run(
        [sys.executable, "-m", "rl_experiments.cli.run_all", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    assert "phase" in r.stdout.lower()


def test_play_help():
    r = subprocess.run(
        [sys.executable, "-m", "rl_experiments.cli.play_model", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
