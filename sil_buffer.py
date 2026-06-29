"""SIL (Self-Imitation Learning) 经验缓冲。"""

import random
from collections import deque

import numpy as np
import torch


class SILBuffer:
    """FIFO buffer，存 (obs, action, return, old_log_prob)。

    每个 agent 独立维护一个 buffer，避免 INDIVIDUAL reward 下跨 agent 混淆。
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.obs = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.returns = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)

    def add(self, obs, action, ret, log_prob):
        """存入一个样本。obs 应为 numpy 或 tensor。"""
        self.obs.append(obs if isinstance(obs, np.ndarray) else np.array(obs))
        self.actions.append(action)
        self.returns.append(ret)
        self.log_probs.append(log_prob)

    def sample(self, batch_size: int):
        """随机采样一个 batch，返回 tensor。"""
        idx = np.random.choice(len(self.obs), size=batch_size, replace=False)
        obs = torch.tensor(np.stack([self.obs[i] for i in idx]), dtype=torch.float32)
        actions = torch.tensor([self.actions[i] for i in idx], dtype=torch.long)
        returns = torch.tensor([self.returns[i] for i in idx], dtype=torch.float32).unsqueeze(-1)
        log_probs = torch.tensor([self.log_probs[i] for i in idx], dtype=torch.float32).unsqueeze(-1)
        return obs, actions, returns, log_probs

    def __len__(self):
        return len(self.obs)
