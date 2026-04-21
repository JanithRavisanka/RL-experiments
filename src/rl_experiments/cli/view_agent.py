#!/usr/bin/env python3
"""Watch a trained SB3 agent — live window or saved GIF/MP4."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from rl_experiments.playback.checkpoints import discover_models, load_sb3_model, parse_model_file

console = Console()


def run_live(model, env_id: str, n_episodes: int = 3, algo: str = "PPO"):
    """Open a pygame window and watch the agent."""
    import gymnasium as gym

    console.print("\n  [bold green]Opening live window[/bold green] — close it to stop\n")
    env = gym.make(env_id, render_mode="human")

    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated
            time.sleep(0.01)

        total_rewards.append(ep_reward)
        console.print(
            f"  Episode {ep + 1}/{n_episodes} — reward: [bold]{ep_reward:.1f}[/bold]  steps: {step}"
        )

    env.close()
    console.print(f"\n  [cyan]Mean reward over {n_episodes} episodes: {np.mean(total_rewards):.1f}[/cyan]")
    return total_rewards


def run_capture(
    model,
    env_id: str,
    n_episodes: int = 1,
    algo: str = "PPO",
    output_path: str = "analysis/replay.gif",
    as_video: bool = False,
):
    """Capture rgb_array frames and save as GIF or MP4."""
    import gymnasium as gym

    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            frame = env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        console.print(f"  Episode {ep + 1}: reward={ep_reward:.1f}  frames={len(frames)}")

    env.close()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if as_video:
        _save_mp4(frames, output_path.replace(".gif", ".mp4"))
    else:
        _save_gif(frames, output_path)

    return rewards


def _save_gif(frames, path: str, fps: int = 30):
    try:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / fps)
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )
        console.print(f"  [green]✓ GIF saved → {path}  ({len(frames)} frames)[/green]")
    except ImportError:
        console.print("  [yellow]Pillow not installed. Run: pip install Pillow[/yellow]")


def _save_mp4(frames, path: str, fps: int = 30):
    try:
        import imageio

        imageio.mimwrite(path, frames, fps=fps, quality=8)
        console.print(f"  [green]✓ MP4 saved → {path}  ({len(frames)} frames)[/green]")
    except ImportError:
        console.print("  [yellow]imageio not installed. Run: pip install imageio[imageio-ffmpeg][/yellow]")
        console.print("  [dim]Falling back to GIF...[/dim]")
        _save_gif(frames, path.replace(".mp4", ".gif"))


def interactive_menu(models: list) -> dict:
    table = Table(
        title="[bold]Available Trained Models",
        header_style="bold cyan",
        border_style="blue",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Algo", justify="center", style="yellow", width=8)
    table.add_column("Environment", style="white", width=28)
    table.add_column("Seed", justify="center", style="cyan", width=6)
    table.add_column("File", style="dim", width=40)

    for i, m in enumerate(models, 1):
        table.add_row(
            str(i),
            m["algo"],
            m["env_id"],
            str(m["seed"]),
            m["path"].name,
        )

    console.print(table)
    console.print()
    choice = Prompt.ask(
        "  [bold]Enter model number to watch[/bold]",
        choices=[str(i) for i in range(1, len(models) + 1)],
    )
    return models[int(choice) - 1]


def main():
    parser = argparse.ArgumentParser(description="View a trained RL agent")
    parser.add_argument("--model", type=str, default=None, help="Path to .zip model file")
    parser.add_argument("--algo", type=str, default=None, choices=["PPO", "SAC", "DQN"], help="Algorithm")
    parser.add_argument("--env", type=str, default=None, help="Env ID")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--gif", action="store_true", help="Save as GIF")
    parser.add_argument("--video", action="store_true", help="Save as MP4")
    parser.add_argument("--out", type=str, default=None, help="Output path for GIF/MP4")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold white]RL Agent Viewer\n[dim]Watch your trained agents play in real-time[/dim]",
            border_style="cyan",
        )
    )

    models = discover_models("results")
    if not models:
        console.print("[red]No models found in results/. Run experiments first.[/red]")
        return

    if args.list:
        for m in models:
            console.print(f"  {m['algo']:4}  {m['env_id']:30}  seed={m['seed']}  {m['path']}")
        return

    if args.model:
        path = Path(args.model)
        algo, env_id, seed = parse_model_file(path)
        algo = args.algo or algo
        env_id = args.env or env_id
        selected = {"path": path, "algo": algo, "env_id": env_id, "seed": seed}
    else:
        selected = interactive_menu(models)

    algo = args.algo or selected["algo"]
    env_id = args.env or selected["env_id"]
    path = selected["path"]

    console.print(f"\n  [bold]Model  :[/bold] {path}")
    console.print(f"  [bold]Algo   :[/bold] [yellow]{algo}[/yellow]")
    console.print(f"  [bold]Env    :[/bold] [cyan]{env_id}[/cyan]")
    console.print(f"  [bold]Episodes:[/bold] {args.episodes}")
    console.print()

    console.print("  Loading model...", end=" ")
    model = load_sb3_model(str(path), algo)
    console.print("[green]✓[/green]")

    if args.gif or args.video:
        stem = f"{algo.lower()}_{env_id.replace('-', '_').lower()}_seed{selected['seed']}"
        ext = ".mp4" if args.video else ".gif"
        outpath = args.out or f"analysis/{stem}_replay{ext}"
        console.print(f"  [yellow]Capturing {args.episodes} episode(s) → {outpath}[/yellow]\n")
        run_capture(model, env_id, args.episodes, algo, outpath, as_video=args.video)
    else:
        run_live(model, env_id, args.episodes, algo)

    console.print("\n  [bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
