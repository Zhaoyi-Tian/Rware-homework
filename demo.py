"""录制模型 Demo 视频。"""
import argparse, os, warnings
import gymnasium as gym
import rware  # noqa: F401
import torch
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from config import Config
from algorithms.iac import IAC
from algorithms.snac import SNAC
from algorithms.seac import SEAC
from algorithms.seac_pooled import SEACPooled
from algorithms.seac_cc import SEACCC
from algorithms.seac_sil import SEACSIL
from algorithms.iac_sil import IACSIL

warnings.filterwarnings("ignore")
ALGO_MAP = {"iac": IAC, "snac": SNAC, "seac": SEAC,
            "seac_pooled": SEACPooled, "seac_cc": SEACCC,
            "seac_sil": SEACSIL, "iac_sil": IACSIL}


def parse_checkpoint(path):
    path = path.rstrip("/")
    ckpt = os.path.join(path, "model.pt") if not path.endswith(".pt") else path
    name = path.split("/")[-1].replace(".pt", "")
    for algo in sorted(ALGO_MAP, key=lambda x: -len(x)):
        if name.startswith(algo):
            env = name[len(algo) + 1:].rsplit("_seed", 1)[0]
            return algo, env, ckpt
    raise ValueError(f"Cannot parse: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=str, default="./videos")
    args = parser.parse_args()

    algo_name, env_name, ckpt_path = parse_checkpoint(args.checkpoint)
    print(f"Algorithm: {algo_name}, Environment: {env_name}")

    config = Config()
    tmp_env = gym.make(env_name)
    obs_dim = tmp_env.observation_space[0].shape[0]
    n_actions = int(tmp_env.action_space[0].n)
    n_agents = tmp_env.unwrapped.n_agents
    tmp_env.close()

    algo = ALGO_MAP[algo_name](config, obs_dim, n_actions, n_agents)
    algo.load(ckpt_path)
    os.makedirs(args.output, exist_ok=True)

    for ep in range(args.episodes):
        env = gym.make(env_name, render_mode="rgb_array")
        obs, info = env.reset(seed=42 + ep)
        done, truncated = False, False
        total_reward, frames = 0.0, []

        while not (done or truncated):
            frames.append(env.render())
            actions = []
            for i, ob in enumerate(obs):
                ob_tensor = torch.from_numpy(ob).float().unsqueeze(0)
                with torch.inference_mode():
                    _, action, _ = algo.networks[i].act(ob_tensor, deterministic=True)
                actions.append(action.item())
            obs, rewards, done, truncated, info = env.step(actions)
            total_reward += sum(rewards)

        try: env.close()
        except: pass

        path = os.path.join(args.output, f"{algo_name}_{ep + 1}.mp4")
        ImageSequenceClip(frames, fps=args.fps).write_videofile(path, logger=None)
        print(f"Episode {ep + 1}: reward = {total_reward} -> {path}")


if __name__ == "__main__":
    main()
