"""SEAC-Pooled: SEAC 改进版 — 所有 agent 经验合并为一个池，重要性加权。

与 SEAC 原版的关键区别：
  SEAC 原版: own_loss + seac_loss，熵只来自 own_loss
  Pooled:    所有 agent 的 (s,a) 合并 → 一次 evaluate → 一个 loss
             自己的 ratio=1，别人的 ratio=π_i/π_j，熵自然统一。
"""

import torch
import torch.optim as optim
from algorithms.base import BaseAlgorithm
from network import ActorCritic
from config import Config


class SEACPooled(BaseAlgorithm):
    """独立网络 + 经验池化 + 重要性加权。"""

    def __init__(self, config: Config, obs_dim: int, n_actions: int, n_agents: int):
        super().__init__(config, obs_dim, n_actions, n_agents)
        self.networks = torch.nn.ModuleList([
            ActorCritic(obs_dim, n_actions, config.hidden_dim)
            for _ in range(n_agents)
        ]).to(self.device)
        self.optimizers = [
            optim.Adam(net.parameters(), lr=config.lr, eps=config.adam_eps)
            for net in self.networks
        ]

    def update(self, rollouts: list) -> dict:
        loss_stats = {}

        for i, (net, opt) in enumerate(zip(self.networks, self.optimizers)):
            all_values, all_log_probs, all_returns, all_weights, all_entropy = [], [], [], [], []

            for j, buf in enumerate(rollouts):
                obs, actions, old_log_probs, returns, _ = buf.get_batch()
                obs = obs.reshape(-1, self.obs_dim)
                actions = actions.reshape(-1)
                returns = returns.reshape(-1, 1)
                n_j = returns.shape[0]

                # 一次 evaluate，同时拿到 ratio 和 loss 需要的 log_probs
                values_j, log_probs_j, entropy_j = net.evaluate(obs, actions)

                if j == i:
                    weight = torch.ones(n_j, 1)
                else:
                    old_log_probs = old_log_probs.reshape(-1, 1)
                    weight = (log_probs_j.unsqueeze(-1).exp()
                              / (old_log_probs.exp() + 1e-7)).detach()

                all_values.append(values_j)
                all_log_probs.append(log_probs_j)
                all_returns.append(returns)
                all_weights.append(weight)
                all_entropy.append(entropy_j * n_j)

            values = torch.cat(all_values).unsqueeze(-1)
            log_probs = torch.cat(all_log_probs).unsqueeze(-1)
            returns = torch.cat(all_returns)
            weights = torch.cat(all_weights)
            entropy = sum(all_entropy) / returns.shape[0]

            advantages = returns - values
            policy_loss = -(weights * log_probs * advantages.detach()).mean()
            value_loss = (weights * advantages.pow(2)).mean()

            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.config.max_grad_norm)
            opt.step()

            loss_stats[f"agent{i}/policy_loss"] = policy_loss.item()
            loss_stats[f"agent{i}/value_loss"] = value_loss.item()
            loss_stats[f"agent{i}/entropy"] = entropy.item()
            loss_stats[f"agent{i}/total_loss"] = loss.item()

        return loss_stats
