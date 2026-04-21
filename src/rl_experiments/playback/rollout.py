"""Roll out policies from checkpoints (SB3 zip, Dreamer/MuZero pt)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import DQN, PPO, SAC

from rl_experiments.playback.checkpoints import infer_env_id_from_path, load_sb3_model, parse_model_file


def play_sb3_zip(path: Path, env_id: str | None, episodes: int, render: str) -> None:
    algo, env_from_name, _ = parse_model_file(path)
    env_id = env_id or env_from_name
    if not env_id:
        raise SystemExit("Could not infer --env; pass explicitly, e.g. --env CartPole-v1")
    if not algo:
        for name, cls in [("PPO", PPO), ("SAC", SAC), ("DQN", DQN)]:
            try:
                cls.load(str(path), device="cpu")
                algo = name
                break
            except Exception:
                continue
        else:
            raise SystemExit("Could not load checkpoint as PPO, SAC, or DQN.")

    model = load_sb3_model(str(path), algo)
    import gymnasium as gym

    mode = "human" if render == "human" else "rgb_array"
    env = gym.make(env_id, render_mode=mode)
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            ep_r += reward
            steps += 1
            done = term or trunc
            if render == "human":
                time.sleep(0.01)
        print(f"  Episode {ep + 1}/{episodes}  reward={ep_r:.1f}  steps={steps}")
    env.close()


def play_dreamer(path: Path, env_id: str, episodes: int, render: str) -> None:
    import gymnasium as gym
    from rl_experiments.advanced.dreamer.dreamer_agent import DreamerAgent

    agent = DreamerAgent(env_id, seed=0)
    try:
        ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=agent.device)
    agent.wm.load_state_dict(ckpt["world_model"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.wm.eval()
    agent.actor.eval()
    agent.critic.eval()

    agent._env.close()
    mode = "human" if render == "human" else None
    agent._env = gym.make(env_id, render_mode=mode)

    for ep in range(episodes):
        obs, _ = agent._env.reset()
        h, z = agent.wm.rssm.initial_state(1, agent.device)
        total_r = 0.0
        steps = 0
        done = False
        while not done:
            action_idx, _, h, z = agent._select_action(obs, h, z, explore=False)
            obs, reward, term, trunc, _ = agent._env.step(action_idx)
            total_r += reward
            steps += 1
            done = term or trunc
            if render == "human":
                time.sleep(0.01)
        print(f"  Episode {ep + 1}/{episodes}  reward={total_r:.1f}  steps={steps}")

    agent._env.close()


def play_muzero(path: Path, env_id: str, episodes: int, render: str) -> None:
    import gymnasium as gym
    from rl_experiments.advanced.muzero.muzero_agent import MuZeroAgent

    agent = MuZeroAgent(env_id, seed=0)
    try:
        ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=agent.device)
    agent.repr_net.load_state_dict(ckpt["repr_net"])
    agent.dyn_net.load_state_dict(ckpt["dyn_net"])
    agent.pred_net.load_state_dict(ckpt["pred_net"])
    agent.repr_net.eval()
    agent.dyn_net.eval()
    agent.pred_net.eval()

    mode = "human" if render == "human" else None
    env = gym.make(env_id, render_mode=mode)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_r = 0.0
        step = 0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            h_t = agent.repr_net(obs_t)
            temp = agent.cfg["temperature"] if step < agent.cfg["temp_threshold"] else 0
            _, action_idx = agent.mcts.improved_policy(h_t, temperature=temp)
            obs, reward, term, trunc, _ = env.step(int(action_idx))
            total_r += reward
            step += 1
            done = term or trunc
            if render == "human":
                time.sleep(0.01)
        print(f"  Episode {ep + 1}/{episodes}  reward={total_r:.1f}  steps={step}")

    env.close()


def play_checkpoint(path: Path, env_id: str | None, episodes: int, render: str) -> None:
    """Dispatch by file type and path hints."""
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    env_resolved = env_id or infer_env_id_from_path(path)

    if path.suffix.lower() == ".zip":
        play_sb3_zip(path, env_resolved, episodes, render)
        return

    if path.suffix.lower() != ".pt":
        raise SystemExit("Expected .zip (SB3) or .pt (PyTorch)")

    s = str(path).lower()
    if "dreamer" in s:
        if not env_resolved:
            raise SystemExit("Pass --env (e.g. CartPole-v1) or use .../dreamer/<env_slug>/")
        play_dreamer(path, env_resolved, episodes, render)
    elif "muzero" in s:
        if not env_resolved:
            raise SystemExit("Pass --env or use path .../muzero/<env_slug>/")
        play_muzero(path, env_resolved, episodes, render)
    else:
        print(
            "This checkpoint type is not wired for playback yet.\n"
            "  • MBPO policy: use the .zip under .../mbpo/.../mbpo_seed*.zip.\n"
            "  • Dreamer/MuZero: path should contain 'dreamer' or 'muzero'.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
