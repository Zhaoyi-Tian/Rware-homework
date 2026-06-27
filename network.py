"""Actor-Critic 网络。

与 SEAC 原版一致：actor 和 critic 各有一份独立 MLP，不共享参数。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init_orth(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    return m


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            _init_orth(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            _init_orth(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            _init_orth(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            _init_orth(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.critic_head = _init_orth(nn.Linear(hidden_dim, 1))
        self.actor_head = _init_orth(nn.Linear(hidden_dim, n_actions), gain=0.01)

    def forward(self, obs):
        return self.critic_head(self.critic(obs)), self.actor_head(self.actor(obs))

    def act(self, obs, deterministic=False):
        value, logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return value, action, dist.log_prob(action)

    def evaluate(self, obs, action):
        value, logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return value, dist.log_prob(action), dist.entropy().mean()
