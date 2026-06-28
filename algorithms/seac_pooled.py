"""SEAC-Pooled: SEAC 改进版 — 所有 agent 经验合并为一个池，重要性加权。

与 SEAC 原版的关键区别：
  SEAC 原版: own_loss + seac_loss，熵只来自 own_loss
  Pooled:    所有 agent 的 (s,a) 合并 → 一次 evaluate → 一个 loss
             自己的 ratio=1，别人的 ratio=π_i/π_j，熵自然统一。
"""

import torch
import torch.nn.functional as F
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
            all_obs, all_actions, all_returns, all_weights = [], [], [], []

            for j, buf in enumerate(rollouts):
                obs, actions, old_log_probs, returns, _ = buf.get_batch()
                obs = obs.reshape(-1, self.obs_dim)
                actions = actions.reshape(-1)
                returns = returns.reshape(-1, 1)

                if j == i:
                    weight = torch.ones(returns.shape[0], 1)
                else:
                    old_log_probs = old_log_probs.reshape(-1, 1)
                    _, log_probs_ij, _ = net.evaluate(obs, actions)
                    weight = (log_probs_ij.unsqueeze(-1).exp()
                              / (old_log_probs.exp() + 1e-7)).detach()

                all_obs.append(obs)
                all_actions.append(actions)
                all_returns.append(returns)
                all_weights.append(weight)

            obs = torch.cat(all_obs)
            actions = torch.cat(all_actions)
            returns = torch.cat(all_returns)
            weights = torch.cat(all_weights)

            values, log_probs, _ = net.evaluate(obs, actions)
            advantages = returns - values

            policy_loss = -(weights * log_probs.unsqueeze(-1) * advantages.detach()).mean()
            value_loss = (weights * advantages.pow(2)).mean()

            # 加权熵：每个样本的熵乘以 importance weight，与 policy/value loss 一致
            _, logits = net.forward(obs)
            full_log_probs = F.log_softmax(logits, -1)
            probs = full_log_probs.exp()
            per_sample_entropy = -(probs * full_log_probs).sum(-1)
            weighted_entropy = (weights.squeeze() * per_sample_entropy).mean()

            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * weighted_entropy
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.config.max_grad_norm)
            opt.step()

            loss_stats[f"agent{i}/policy_loss"] = policy_loss.item()
            loss_stats[f"agent{i}/value_loss"] = value_loss.item()
            loss_stats[f"agent{i}/entropy"] = weighted_entropy.item()
            loss_stats[f"agent{i}/total_loss"] = loss.item()

        return loss_stats
