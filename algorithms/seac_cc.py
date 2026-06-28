"""SEAC-CC: SEAC with an ego-centered centralized critic."""

import numpy as np
import torch
import torch.optim as optim
from algorithms.base import BaseAlgorithm
from buffer import BatchedRolloutBuffer
from network import CentralizedActorCritic
from config import Config


class SEACCC(BaseAlgorithm):
    """SEAC actor sharing with centralized critics for individual returns."""

    def __init__(self, config: Config, obs_dim: int, n_actions: int, n_agents: int):
        super().__init__(config, obs_dim, n_actions, n_agents)
        self.networks = torch.nn.ModuleList([
            CentralizedActorCritic(obs_dim, n_actions, n_agents, config.hidden_dim)
            for _ in range(n_agents)
        ]).to(self.device)
        self.optimizers = [
            optim.Adam(net.parameters(), lr=config.lr, eps=config.adam_eps)
            for net in self.networks
        ]

    def _ego_joint_obs(self, obs_batches: list, ego_idx: int) -> torch.Tensor:
        ordered = [obs_batches[ego_idx]]
        ordered.extend(obs_batches[j] for j in range(self.n_agents) if j != ego_idx)
        return torch.cat(ordered, dim=-1)

    def _rollout_joint_obs(self, rollouts: list, ego_idx: int) -> torch.Tensor:
        obs_batches = [
            rollouts[j].obs.reshape(-1, self.obs_dim)
            for j in range(self.n_agents)
        ]
        return self._ego_joint_obs(obs_batches, ego_idx)

    def collect_rollout(self, envs: list) -> tuple:
        """Collect rollout with local actors and ego-centered centralized critics."""
        n_envs = len(envs)
        if not hasattr(self, "_current_obs"):
            self._current_obs = []
            for e_idx, env in enumerate(envs):
                obs, info = env.reset(seed=self.config.seed + e_idx)
                self._current_obs.append(obs)

        buffers = [
            BatchedRolloutBuffer(self.obs_dim, self.config.num_steps, n_envs)
            for _ in range(self.n_agents)
        ]
        completed_rewards = []

        _zeros = torch.zeros(n_envs, 1)
        _ones = torch.ones(n_envs, 1)
        _rew = torch.zeros(n_envs, self.n_agents, 1)

        for step in range(self.config.num_steps):
            obs_batches = [
                torch.from_numpy(np.stack([
                    self._current_obs[e_idx][i] for e_idx in range(n_envs)
                ])).float()
                for i in range(self.n_agents)
            ]

            batched_actions = []
            for i, obs_batch in enumerate(obs_batches):
                joint_obs = self._ego_joint_obs(obs_batches, i)
                with torch.inference_mode():
                    value = self.networks[i].forward_critic(joint_obs)
                    _, action, log_prob = self.networks[i].act(obs_batch)
                buffers[i].insert(
                    obs=obs_batch, action=action,
                    reward=_zeros, log_prob=log_prob, value=value,
                    mask=_ones, bad_mask=_ones,
                )
                batched_actions.append(action)

            _rew.zero_()
            _masks = _ones.clone()
            _bad_masks = _ones.clone()
            next_obs = [None] * n_envs

            for e_idx, env in enumerate(envs):
                actions = [
                    int(batched_actions[i][e_idx].item())
                    for i in range(self.n_agents)
                ]
                obs, rewards, done, truncated, info = env.step(actions)

                if "episode_reward" in info:
                    completed_rewards.append(info["episode_reward"])

                mask = 0.0 if (done or truncated) else 1.0
                bad_mask = 0.0 if truncated else 1.0
                for i in range(self.n_agents):
                    _rew[e_idx, i, 0] = rewards[i]
                _masks[e_idx, 0] = mask
                _bad_masks[e_idx, 0] = bad_mask

                if done or truncated:
                    obs, info = env.reset()
                next_obs[e_idx] = obs

            self._current_obs = next_obs
            for i in range(self.n_agents):
                buffers[i].rewards[step] = _rew[:, i]
                buffers[i].masks[step] = _masks
                buffers[i].bad_masks[step] = _bad_masks

        next_obs_batches = [
            torch.from_numpy(np.stack([
                self._current_obs[e_idx][i] for e_idx in range(n_envs)
            ])).float()
            for i in range(self.n_agents)
        ]
        for i in range(self.n_agents):
            joint_obs = self._ego_joint_obs(next_obs_batches, i)
            with torch.inference_mode():
                next_value = self.networks[i].forward_critic(joint_obs)
            buffers[i].compute_returns(next_value, self.config.gamma)

        return buffers, completed_rewards

    def update(self, rollouts: list) -> dict:
        loss_stats = {}
        n_agents = len(rollouts)

        obs_batches = []
        action_batches = []
        old_log_prob_batches = []
        return_batches = []
        for buf in rollouts:
            obs, actions, old_log_probs, returns, _ = buf.get_batch()
            obs_batches.append(obs.reshape(-1, self.obs_dim))
            action_batches.append(actions.reshape(-1))
            old_log_prob_batches.append(old_log_probs.reshape(-1, 1))
            return_batches.append(returns.reshape(-1, 1))

        joint_obs_batches = [
            self._ego_joint_obs(obs_batches, ego_idx)
            for ego_idx in range(n_agents)
        ]

        for i, (net, opt, buf_i) in enumerate(zip(self.networks, self.optimizers, rollouts)):
            obs_i = obs_batches[i]
            actions_i = action_batches[i]
            returns_i = return_batches[i]

            joint_obs_i = joint_obs_batches[i]
            values_i = net.forward_critic(joint_obs_i)
            log_probs_i, entropy = net.evaluate_actor(obs_i, actions_i)
            advantages_i = returns_i - values_i

            policy_loss = -(log_probs_i.unsqueeze(-1) * advantages_i.detach()).mean()
            value_loss = advantages_i.pow(2).mean()

            seac_policy_loss = 0.0
            seac_value_loss = 0.0
            for j in range(n_agents):
                if j == i:
                    continue

                obs_j = obs_batches[j]
                actions_j = action_batches[j]
                old_log_probs_j = old_log_prob_batches[j]
                returns_j = return_batches[j]
                joint_obs_j = joint_obs_batches[j]
                values_j = net.forward_critic(joint_obs_j)
                log_probs_ij, _ = net.evaluate_actor(obs_j, actions_j)
                advantages_j = returns_j - values_j

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
                + seac_policy_loss
                + self.config.value_loss_coef * seac_value_loss
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
            ).item()
            loss_stats[f"agent{i}/total_loss"] = loss.item()

        return loss_stats
