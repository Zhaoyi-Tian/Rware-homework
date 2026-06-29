"""SEAC-SIL: SEAC + Self-Imitation Learning。

在 SEAC 基础上增加 SIL buffer，复用自身过去的成功经验。
每个 agent 独立 buffer，使用 importance sampling 修正 off-policy bias。
"""

import torch
import torch.optim as optim
from algorithms.base import BaseAlgorithm
from network import ActorCritic
from sil_buffer import SILBuffer
from config import Config


class SEACSIL(BaseAlgorithm):
    """SEAC 跨 agent 经验共享 + SIL 纵向经验复用。"""

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
        self.sil_buffers = [
            SILBuffer(capacity=config.sil_capacity)
            for _ in range(n_agents)
        ]

    def update(self, rollouts: list) -> dict:
        loss_stats = {}
        n_agents = len(rollouts)

        for i, (net, opt, buf_i) in enumerate(zip(self.networks, self.optimizers, rollouts)):
            obs_i, actions_i, old_log_probs_i, returns_i, values_i = buf_i.get_batch()
            obs_i = obs_i.reshape(-1, self.obs_dim)
            actions_i = actions_i.reshape(-1)
            old_log_probs_i = old_log_probs_i.reshape(-1, 1)
            returns_i = returns_i.reshape(-1, 1)
            values_i = values_i.reshape(-1, 1)

            values_i_eval, log_probs_i, entropy = net.evaluate(obs_i, actions_i)
            advantages_i = returns_i - values_i_eval

            policy_loss = -(log_probs_i.unsqueeze(-1) * advantages_i.detach()).mean()
            value_loss = advantages_i.pow(2).mean()

            # SEAC 跨 agent 经验共享
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

                seac_value_loss += (ratio * advantages_j.pow(2)).mean()
                seac_policy_loss += (-ratio
                                     * log_probs_ij.unsqueeze(-1)
                                     * advantages_j.detach()).mean()

            # SIL: 存入好经验
            with torch.no_grad():
                mask_good = (advantages_i > 0).squeeze()
                o_good = obs_i[mask_good].cpu().numpy()
                a_good = actions_i[mask_good].cpu().numpy()
                r_good = returns_i[mask_good].squeeze(-1).cpu().numpy()
                lp_good = old_log_probs_i[mask_good].squeeze(-1).cpu().numpy()
                for k in range(len(o_good)):
                    self.sil_buffers[i].add(o_good[k], int(a_good[k]), float(r_good[k]), float(lp_good[k]))

            # SIL: 从 buffer 采样并计算 loss（与 SEAC 跨 agent 对称）
            sil_policy_loss = torch.tensor(0.0)
            sil_value_loss = torch.tensor(0.0)
            n_sil_samples = 0
            sil_batch_size = self.config.num_steps * self.config.num_envs * (n_agents - 1)

            if len(self.sil_buffers[i]) >= sil_batch_size:
                obs_sil, act_sil, ret_sil, old_lp_sil = self.sil_buffers[i].sample(sil_batch_size)
                obs_sil = obs_sil.to(self.device)
                act_sil = act_sil.to(self.device)
                ret_sil = ret_sil.to(self.device)
                old_lp_sil = old_lp_sil.to(self.device)

                val_sil, lp_sil, _ = net.evaluate(obs_sil, act_sil)
                adv_sil = ret_sil - val_sil

                ratio_sil = (lp_sil.unsqueeze(-1).exp()
                             / (old_lp_sil.exp() + 1e-7)).detach()

                sil_mask = (adv_sil > 0).float()
                sil_policy_loss = (-ratio_sil * sil_mask
                                   * lp_sil.unsqueeze(-1)
                                   * adv_sil.detach()).mean()
                sil_value_loss = (ratio_sil * sil_mask * adv_sil.pow(2)).mean()
                n_sil_samples = obs_sil.shape[0]

            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy
                + seac_policy_loss
                + self.config.value_loss_coef * seac_value_loss
                + self.config.sil_coef * (sil_policy_loss
                                          + self.config.value_loss_coef * sil_value_loss)
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
            loss_stats[f"agent{i}/sil_loss"] = (
                sil_policy_loss.item() + self.config.value_loss_coef * sil_value_loss.item()
            )
            loss_stats[f"agent{i}/sil_samples"] = float(n_sil_samples)
            loss_stats[f"agent{i}/total_loss"] = loss.item()

        return loss_stats
