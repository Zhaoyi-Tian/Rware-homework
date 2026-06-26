"""Actor-Critic 网络。"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init_layer(m):
    """正交初始化：gain=√2，bias=0。"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)


class ActorCritic(nn.Module):
    """MLP Actor-Critic：
       obs → Linear(hidden) → ReLU → Linear(hidden) → ReLU
                                        ├─→ actor: Linear(hidden, n_actions)
                                        └─→ critic: Linear(hidden, 1)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(_init_layer)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0)

    def forward(self, obs):
        feat = self.shared(obs)
        return self.critic(feat), self.actor(feat)

    def act(self, obs, deterministic=False):
        value, logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return value, action, dist.log_prob(action)

    def evaluate(self, obs, action):
        value, logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return value, dist.log_prob(action), dist.entropy().mean()
