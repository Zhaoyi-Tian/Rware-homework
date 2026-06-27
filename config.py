"""所有超参数集中管理"""

from dataclasses import dataclass


@dataclass
class Config:
    # 环境
    env_name: str = "rware-tiny-2ag-v2"
    algorithm: str = "iac"
    seed: int = 42

    # 网络
    hidden_dim: int = 128
    adam_eps: float = 0.001

    # A2C 训练
    lr: float = 3e-4
    gamma: float = 0.99
    num_envs: int = 8            # 并行环境数
    num_steps: int = 20          # 每次 rollout 的步数
    total_steps: int = 60_000_000
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 日志 & 保存
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    eval_interval: int = 100_000
    eval_episodes: int = 10
