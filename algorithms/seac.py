"""SEAC: Shared Experience Actor-Critic。"""

import torch
import torch.optim as optim
from algorithms.base import BaseAlgorithm
from network import ActorCritic
from config import Config


class SEAC(BaseAlgorithm):
    """独立网络 + 跨 agent 经验共享。
    每个 agent 除了自己的 A2C loss，还额外用其他 agent 的经验做重要性采样训练。
    """

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

            # SEAC: 用其他 agent 的经验训练当前 agent
            other_ids = [j for j in range(n_agents) if j != i]
            seac_policy_loss = 0.0
            seac_value_loss = 0.0

            for j in other_ids:
                obs_j, actions_j, old_log_probs_j, returns_j, _ = rollouts[j].get_batch()
                obs_j = obs_j.reshape(-1, self.obs_dim)
                actions_j = actions_j.reshape(-1)
                returns_j = returns_j.reshape(-1, 1)
                old_log_probs_j = old_log_probs_j.reshape(-1, 1)

                # 用 agent i 的当前网络评价 agent j 的经验
                values_j, log_probs_ij, _ = net.evaluate(obs_j, actions_j)
                advantages_j = returns_j - values_j

                # 重要性采样比率: π_i(a|s) / π_j_old(a|s)
                ratio = (log_probs_ij.unsqueeze(-1).exp()
                         / (old_log_probs_j.exp() + 1e-7)).detach()

                seac_value_loss += (ratio * advantages_j.pow(2)).mean()
                seac_policy_loss += (-ratio
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
