"""Selective-SEAC: 基于信息量筛选跨 agent 经验。

与 SEAC 原版的区别：重要性比率 ratio 乘以 |advantage|，
只学那些"实际带来价值变化"的他人经验，忽略无信息的平凡帧。
"""

import torch
import torch.optim as optim
from algorithms.base import BaseAlgorithm
from network import ActorCritic
from config import Config


class SEACSelective(BaseAlgorithm):
    """独立网络 + 信息选择性跨 agent 经验共享。"""

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
        n_agents = len(rollouts)

        for i, (net, opt, buf_i) in enumerate(zip(self.networks, self.optimizers, rollouts)):
            obs_i, actions_i, _, returns_i, _ = buf_i.get_batch()
            obs_i = obs_i.reshape(-1, self.obs_dim)
            actions_i = actions_i.reshape(-1)
            returns_i = returns_i.reshape(-1, 1)

            values_i, log_probs_i, entropy = net.evaluate(obs_i, actions_i)
            advantages_i = returns_i - values_i

            policy_loss = -(log_probs_i.unsqueeze(-1) * advantages_i.detach()).mean()
            value_loss = advantages_i.pow(2).mean()

            other_ids = [j for j in range(n_agents) if j != i]
            seac_policy_loss = 0.0
            seac_value_loss = 0.0

            for j in other_ids:
                obs_j, actions_j, old_log_probs_j, returns_j, _ = rollouts[j].get_batch()
                obs_j = obs_j.reshape(-1, self.obs_dim)
                actions_j = actions_j.reshape(-1)
                returns_j = returns_j.reshape(-1, 1)
                old_log_probs_j = old_log_probs_j.reshape(-1, 1)

                values_j, log_probs_ij, _ = net.evaluate(obs_j, actions_j)
                advantages_j = returns_j - values_j

                ratio = (log_probs_ij.unsqueeze(-1).exp()
                         / (old_log_probs_j.exp() + 1e-7)).detach()

                # 信息选择：ratio × |advantage|，无信息的 transition 权重趋于 0
                weight = (ratio * advantages_j.abs()).detach()

                seac_value_loss += (weight * advantages_j.pow(2)).mean()
                seac_policy_loss += (-weight
                                     * log_probs_ij.unsqueeze(-1)
                                     * advantages_j.detach()).mean()

            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy
                + (seac_policy_loss
                   + self.config.value_loss_coef * seac_value_loss)
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.config.max_grad_norm)
            opt.step()

            loss_stats[f"agent{i}/policy_loss"] = policy_loss.item()
            loss_stats[f"agent{i}/value_loss"] = value_loss.item()
            loss_stats[f"agent{i}/entropy"] = entropy.item()
            loss_stats[f"agent{i}/seac_loss"] = (
                seac_policy_loss + self.config.value_loss_coef * seac_value_loss
            )
            loss_stats[f"agent{i}/total_loss"] = loss.item()

        return loss_stats
