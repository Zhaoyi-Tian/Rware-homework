"""训练入口。"""

import os

# 必须在 import torch/numpy 之前设置，否则库初始化后无效
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(1)

from config import Config
from utils import make_envs, set_seed, evaluate
from algorithms.iac import IAC
from algorithms.snac import SNAC
from algorithms.seac import SEAC
from algorithms.seac_pooled import SEACPooled
from algorithms.seac_selective import SEACSelective


def parse_args():
    parser = argparse.ArgumentParser(description="RWARE A2C Training")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["iac", "snac", "seac", "seac_pooled", "seac_selective"])
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.env is not None: cfg.env_name = args.env
    if args.algorithm is not None: cfg.algorithm = args.algorithm
    if args.total_steps is not None: cfg.total_steps = args.total_steps
    if args.num_envs is not None: cfg.num_envs = args.num_envs
    if args.num_steps is not None: cfg.num_steps = args.num_steps
    if args.lr is not None: cfg.lr = args.lr
    if args.seed is not None: cfg.seed = args.seed
    if args.log_dir is not None: cfg.log_dir = args.log_dir
    if args.checkpoint_dir is not None: cfg.checkpoint_dir = args.checkpoint_dir
    config = cfg
    set_seed(config.seed)

    envs = make_envs(config.env_name, config.num_envs)
    obs_dim = envs[0].observation_space[0].shape[0]
    n_actions = int(envs[0].action_space[0].n)
    n_agents = envs[0].unwrapped.n_agents

    print(f"Environment: {config.env_name}")
    print(f"  Agents: {n_agents}, Envs: {config.num_envs}, Obs dim: {obs_dim}, Actions: {n_actions}")
    print(f"  Device: cpu, Total steps: {config.total_steps:,}")

    if config.algorithm == "iac":
        algo = IAC(config, obs_dim, n_actions, n_agents)
    elif config.algorithm == "snac":
        algo = SNAC(config, obs_dim, n_actions, n_agents)
    elif config.algorithm == "seac":
        algo = SEAC(config, obs_dim, n_actions, n_agents)
    elif config.algorithm == "seac_pooled":
        algo = SEACPooled(config, obs_dim, n_actions, n_agents)
    elif config.algorithm == "seac_selective":
        algo = SEACSelective(config, obs_dim, n_actions, n_agents)
    else:
        raise NotImplementedError(f"Algorithm {config.algorithm} not implemented yet")

    run_name = f"{config.algorithm}_{config.env_name}_seed{config.seed}"
    log_path = os.path.join(config.log_dir, run_name)
    writer = SummaryWriter(log_path)
    checkpoint_path = os.path.join(config.checkpoint_dir, run_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    episode_rewards = []
    steps_per_update = config.num_envs * config.num_steps
    num_updates = config.total_steps // steps_per_update

    start_update = 0
    if args.resume:
        resume_path = args.resume if args.resume.endswith(".pt") else os.path.join(args.resume, "model.pt")
        resumed_steps = algo.load(resume_path)
        start_update = resumed_steps // steps_per_update
        print(f"Resumed from step {resumed_steps:,}, starting at update {start_update + 1}")

    if start_update == 0:
        start_time = time.time()
    else:
        start_time = time.time() - start_update * steps_per_update / 1000  # 近似

    print(f"\nTraining for {num_updates} updates ({config.total_steps} steps)...\n")

    for update_idx in range(start_update + 1, num_updates + 1):
        rollouts, completed = algo.collect_rollout(envs)
        episode_rewards.extend(completed)

        loss_stats = algo.update(rollouts)
        total_steps_done = update_idx * steps_per_update

        if update_idx % 100 == 0:
            writer.add_scalar("train/policy_loss", loss_stats["agent0/policy_loss"], total_steps_done)
            writer.add_scalar("train/value_loss", loss_stats["agent0/value_loss"], total_steps_done)
            writer.add_scalar("train/entropy", loss_stats["agent0/entropy"], total_steps_done)

            recent_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            writer.add_scalar("train/episode_reward", recent_r, total_steps_done)

            fps = total_steps_done / (time.time() - start_time)
            print(f"Update {update_idx:6d}/{num_updates} | "
                  f"steps {total_steps_done:9,d} | "
                  f"avg_r {recent_r:5.1f} | "
                  f"episodes {len(episode_rewards):4d} | "
                  f"loss {loss_stats['agent0/total_loss']:.4f} | "
                  f"fps {fps:.0f}")

        eval_step_interval = config.eval_interval // steps_per_update
        if eval_step_interval > 0 and update_idx % eval_step_interval == 0:
            eval_r = evaluate(algo, config.env_name,
                              n_episodes=config.eval_episodes, seed=config.seed)
            writer.add_scalar("eval/mean_reward", eval_r, total_steps_done)
            print(f"  >>> Eval at step {total_steps_done:9,d}: mean reward = {eval_r:.3f}")
            algo.save(checkpoint_path, total_steps_done)
            print(f"  >>> Checkpoint saved to {checkpoint_path}")


    algo.save(checkpoint_path, total_steps_done)
    print(f"\nFinal model saved to {checkpoint_path}")

    for e in envs:
        e.close()
    writer.close()

    print("\n=== Final Evaluation ===")
    final_r = evaluate(algo, config.env_name,
                       n_episodes=config.eval_episodes, seed=config.seed)
    print(f"Final mean reward over {config.eval_episodes} episodes: {final_r:.3f}")


if __name__ == "__main__":
    main()
