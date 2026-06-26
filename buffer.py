"""Rollout 经验存储。"""

import torch


class BatchedRolloutBuffer:

    def __init__(self, obs_dim: int, num_steps: int, num_envs: int):
        self.obs_dim = obs_dim
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.num_steps, self.num_envs, self.obs_dim)
        self.actions = torch.zeros(self.num_steps, self.num_envs, 1, dtype=torch.long)
        self.rewards = torch.zeros(self.num_steps, self.num_envs, 1)
        self.log_probs = torch.zeros(self.num_steps, self.num_envs, 1)
        self.values = torch.zeros(self.num_steps, self.num_envs, 1)
        self.masks = torch.ones(self.num_steps, self.num_envs, 1)
        self.bad_masks = torch.ones(self.num_steps, self.num_envs, 1)
        self.returns = torch.zeros(self.num_steps, self.num_envs, 1)
        self.step = 0

    def insert(self, obs, action, reward, log_prob, value, mask, bad_mask):
        self.obs[self.step] = obs
        self.actions[self.step, :, 0] = action
        self.rewards[self.step] = reward
        self.log_probs[self.step, :, 0] = log_prob
        self.values[self.step] = value
        self.masks[self.step] = mask
        self.bad_masks[self.step] = bad_mask
        self.step += 1

    def compute_returns(self, next_value: torch.Tensor, gamma: float):
        R = next_value
        for step in reversed(range(self.num_steps)):
            R = self.rewards[step] + gamma * R * self.masks[step]
            R = R * self.bad_masks[step] + (1 - self.bad_masks[step]) * self.values[step]
            self.returns[step] = R

    def get_batch(self):
        return self.obs, self.actions, self.log_probs, self.returns, self.values
