"""Actor-Critic 网络。共享 backbone，actor/critic 各一个 head。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_orth(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    return m


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            _init_orth(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            _init_orth(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.actor = _init_orth(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.critic = _init_orth(nn.Linear(hidden_dim, 1))

    def forward(self, obs):
        feat = self.shared(obs)
        return self.critic(feat), self.actor(feat)

    def act(self, obs, deterministic=False):
        value, logits = self.forward(obs)
        log_probs_full = F.log_softmax(logits, dim=-1)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.multinomial(log_probs_full.exp(), 1).squeeze(-1)
        log_prob = log_probs_full.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return value, action, log_prob

    def evaluate(self, obs, action):
        value, logits = self.forward(obs)
        log_probs_full = F.log_softmax(logits, dim=-1)
        log_prob = log_probs_full.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        probs = log_probs_full.exp()
        entropy = -(probs * log_probs_full).sum(-1).mean()
        return value, log_prob, entropy
