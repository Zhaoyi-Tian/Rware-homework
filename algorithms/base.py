"""A2C 算法抽象基类。"""

from abc import ABC, abstractmethod
import os
import warnings
import numpy as np
import torch
from buffer import BatchedRolloutBuffer
from config import Config

warnings.filterwarnings("ignore", message=".*reward returned by.*")


class BaseAlgorithm(ABC):
    """A2C 算法基类。子类实现 update()。"""

    def __init__(self, config: Config, obs_dim: int, n_actions: int, n_agents: int):
        self.config = config
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = torch.device("cpu")

    @abstractmethod
    def update(self, rollouts: list) -> dict:
        ...

    def collect_rollout(self, envs: list) -> tuple:
        """从多个环境收集 num_steps 步经验。
        返回 (buffers_per_agent, completed_rewards)。
        buffers_per_agent[agent_i] = BatchedRolloutBuffer([T, N, ...])
        """
        n_envs = len(envs)
        if not hasattr(self, '_current_obs'):
            self._current_obs = []
            for e_idx, env in enumerate(envs):
                obs, info = env.reset(seed=self.config.seed + e_idx)
                self._current_obs.append(obs)

        buffers = [
            BatchedRolloutBuffer(self.obs_dim, self.config.num_steps, n_envs)
            for _ in range(self.n_agents)
        ]
        completed_rewards = []

        _zeros = torch.zeros(n_envs, 1)
        _ones = torch.ones(n_envs, 1)
        _rew = torch.zeros(n_envs, self.n_agents, 1)

        for step in range(self.config.num_steps):
            obs_batches = [
                torch.from_numpy(np.stack([
                    self._current_obs[e_idx][i] for e_idx in range(n_envs)
                ])).float()
                for i in range(self.n_agents)
            ]
            batched_actions = []
            for i, obs_batch in enumerate(obs_batches):
                with torch.inference_mode():
                    value, action, log_prob = self.networks[i].act(obs_batch)
                buffers[i].insert(
                    obs=obs_batch, action=action,
                    reward=_zeros, log_prob=log_prob, value=value,
                    mask=_ones, bad_mask=_ones,
                )
                batched_actions.append(action)

            _rew.zero_()
            _masks = _ones.clone()
            _bad_masks = _ones.clone()
            next_obs = [None] * n_envs

            for e_idx, env in enumerate(envs):
                actions = [int(batched_actions[i][e_idx].item()) for i in range(self.n_agents)]
                obs, rewards, done, truncated, info = env.step(actions)

                if "episode_reward" in info:
                    completed_rewards.append(info["episode_reward"])

                mask = 0.0 if (done or truncated) else 1.0
                bad_mask = 0.0 if truncated else 1.0
                for i in range(self.n_agents):
                    _rew[e_idx, i, 0] = rewards[i]
                _masks[e_idx, 0] = mask
                _bad_masks[e_idx, 0] = bad_mask

                if done or truncated:
                    obs, info = env.reset()
                next_obs[e_idx] = obs

            self._current_obs = next_obs
            for i in range(self.n_agents):
                buffers[i].rewards[step] = _rew[:, i]
                buffers[i].masks[step] = _masks
                buffers[i].bad_masks[step] = _bad_masks

        for i in range(self.n_agents):
            obs_batch = torch.from_numpy(np.stack([
                self._current_obs[e_idx][i] for e_idx in range(n_envs)
            ])).float()
            with torch.inference_mode():
                next_value = self.networks[i].forward(obs_batch)[0]
            buffers[i].compute_returns(next_value, self.config.gamma)

        return buffers, completed_rewards

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(
            {f"network_{i}": net.state_dict() for i, net in enumerate(self.networks)},
            os.path.join(path, "model.pt"),
        )

    def load(self, path: str):
        checkpoint = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        for i, net in enumerate(self.networks):
            net.load_state_dict(checkpoint[f"network_{i}"])
