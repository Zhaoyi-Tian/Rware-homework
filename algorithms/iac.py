"""IAC: Independent Actor-Critic。"""

import torch
import torch.optim as optim
from algorithms.base import BaseAlgorithm
from network import ActorCritic
from config import Config


class IAC(BaseAlgorithm):
    """每个智能体独立网络 + 独立优化器。"""

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
        """每个 agent 用自己的数据独立更新。
        rollouts[agent_i] = BatchedRolloutBuffer，包含所有环境的数据。
        """
        loss_stats = {}
        for i, (net, opt, buf) in enumerate(zip(self.networks, self.optimizers, rollouts)):
            obs, actions, _, returns, _ = buf.get_batch()
            obs = obs.reshape(-1, self.obs_dim)
            actions = actions.reshape(-1)
            returns = returns.reshape(-1, 1)

            values, log_probs, entropy = net.evaluate(obs, actions)
            advantages = returns - values

            policy_loss = -(log_probs.unsqueeze(-1) * advantages.detach()).mean()
            value_loss = advantages.pow(2).mean()

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
