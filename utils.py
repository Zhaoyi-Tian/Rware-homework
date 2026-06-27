"""环境创建、随机种子、评估。"""

import random
import numpy as np
import torch
import gymnasium as gym
import rware  # noqa: F401


class RecordEpisodeStats(gym.Wrapper):
    """记录 episode 奖励/步数，写入 info dict。"""

    def __init__(self, env):
        super().__init__(env)
        self.n_agents = env.unwrapped.n_agents
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        return obs, info

    def step(self, action):
        obs, rewards, done, truncated, info = self.env.step(action)
        self.episode_reward += np.array(rewards)
        self.episode_length += 1
        if done or truncated:
            info["episode_reward"] = self.episode_reward.sum()
            info["episode_length"] = self.episode_length
        return obs, rewards, done, truncated, info


def make_env(env_name: str) -> gym.Env:
    env = gym.make(env_name)
    max_ep = env.unwrapped.max_steps or 500
    env.unwrapped.max_steps = None
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_ep)
    env = RecordEpisodeStats(env)
    return env


def make_envs(env_name: str, n: int) -> list:
    return [make_env(env_name) for _ in range(n)]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def evaluate(agent, env_name: str, n_episodes: int = 10,
             seed: int = 42, render: bool = False) -> float:
    """评估策略：跑 n_episodes 局，返回平均奖励（确定性动作）。"""
    env = make_env(env_name)
    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 999 + ep)
        done, truncated = False, False
        total_reward = 0.0

        while not (done or truncated):
            actions = []
            for i, ob in enumerate(obs):
                ob_tensor = torch.from_numpy(ob).float().unsqueeze(0)
                with torch.inference_mode():
                    _, action, _ = agent.networks[i].act(ob_tensor, deterministic=True)
                actions.append(action.item())

            obs, rewards, done, truncated, info = env.step(actions)
            total_reward += sum(rewards)
            if render:
                env.render()

        episode_rewards.append(total_reward)

    env.close()
    return np.mean(episode_rewards)
