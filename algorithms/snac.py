"""SNAC: Shared Network Actor-Critic。"""

import torch
import torch.optim as optim
from algorithms.base import BaseAlgorithm
from network import ActorCritic
from config import Config


class SNAC(BaseAlgorithm):
    """所有智能体共享一个网络 + 一个优化器。梯度跨 agent 累积后一次 step。"""

    def __init__(self, config: Config, obs_dim: int, n_actions: int, n_agents: int):
        super().__init__(config, obs_dim, n_actions, n_agents)
        shared = ActorCritic(obs_dim, n_actions, config.hidden_dim).to(self.device)
        self.networks = torch.nn.ModuleList([shared] * n_agents)
        self.optimizers = [optim.Adam(shared.parameters(), lr=config.lr, eps=config.adam_eps)]

    def update(self, rollouts: list) -> dict:
        self.optimizers[0].zero_grad()

        all_obs, all_actions, all_returns = [], [], []
        for buf in rollouts:
            obs, actions, _, returns, _ = buf.get_batch()
            all_obs.append(obs.reshape(-1, self.obs_dim))
            all_actions.append(actions.reshape(-1))
            all_returns.append(returns.reshape(-1, 1))

        obs = torch.cat(all_obs)
        actions = torch.cat(all_actions)
        returns = torch.cat(all_returns)

        values, log_probs, entropy = self.networks[0].evaluate(obs, actions)
        advantages = returns - values

        policy_loss = -(log_probs.unsqueeze(-1) * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks[0].parameters(), self.config.max_grad_norm)
        self.optimizers[0].step()

        return {
            "agent0/policy_loss": policy_loss.item(),
            "agent0/value_loss": value_loss.item(),
            "agent0/entropy": entropy.item(),
            "agent0/total_loss": loss.item(),
        }
