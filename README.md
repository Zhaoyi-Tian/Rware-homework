# RWARE 多智能体强化学习：SEAC 训练、分析与改进

多智能体强化学习课程作业 —— 在 RWARE 仓储环境中复现 SEAC 算法，发现并修正熵不对称问题，提出 SEAC-SIL 加速收敛。

**完整报告：** [report.pdf](report.pdf)

## 概述

从零实现 IAC、SNAC、SEAC 三个基线算法（PyTorch A2C），在 tiny-2ag（80M 步）和 small-4ag（150M 步）两个规模下完成训练与评估。主要工作：

1. **SEAC 熵不对称** — 发现 SEAC 跨智能体损失项不含熵正则化，导致策略坍缩为近确定性（H≈0.21 vs IAC 的 H≈0.76），训练 reward 虚高
2. **SEAC-Pooled** — 为跨智能体经验项补上重要性加权熵，消除不对称，实现公平比较
3. **温度扫描评估** — 将策略锐度（熵）与动作排序质量解耦，在不同熵水平下公平对比各算法
4. **SEAC-SIL** — 将跨智能体经验共享与自模仿学习结合，进一步加速收敛

## 算法

| 算法 | 说明 |
|------|------|
| `iac` | Independent Actor-Critic，每个智能体独立网络 |
| `snac` | Shared Network Actor-Critic，所有智能体共享网络 |
| `seac` | Shared Experience Actor-Critic，跨智能体重要性采样 |
| `seac_pooled` | SEAC + 跨智能体数据上的重要性加权熵 |
| `iac_sil` | IAC + Self-Imitation Learning |
| `seac_sil` | SEAC + Self-Imitation Learning |

实现位于 [`algorithms/`](algorithms/)。

## 安装

```bash
conda create -n Rware python=3.9 -y
conda activate Rware
pip install -r requirements.txt
```

## 训练

```bash
# 单个算法
python main.py --algorithm seac --env rware-tiny-2ag-v2 --total-steps 80000000 --seed 42

# 批量运行（全部基线，两个环境）
bash run_full.sh

# SIL 对比（small-4ag，50M 步）
bash run_sil.sh
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algorithm` | 必填 | `iac` / `snac` / `seac` / `seac_pooled` / `iac_sil` / `seac_sil` |
| `--env` | 必填 | `rware-tiny-2ag-v2` / `rware-small-4ag-v2` |
| `--total-steps` | 必填 | 总环境步数 |
| `--seed` | 42 | 随机种子 |
| `--lr` | 3e-4 | 学习率 |
| `--resume` | — | 续训 checkpoint 路径 |

Checkpoint 保存至 `checkpoints/<算法>_<环境>_seed<种子>/`，日志保存至 `logs/<算法>_<环境>_seed<种子>/`。

## 查看训练日志

```bash
tensorboard --logdir logs
```

浏览器打开 `http://localhost:6006`。每个算法+环境组合显示为独立 run。关键指标：

- `agent0/episode_reward` — episode 奖励
- `agent0/entropy` — 训练中的策略熵
- `agent0/policy_loss`、`agent0/value_loss`、`agent0/total_loss`

SIL 变体的日志在 `logs/sil/` 下：
```bash
tensorboard --logdir logs/sil
```

## 生成论文图片

```bash
# 图 1：基线训练曲线 + 熵曲线 + 温度扫描
python analysis/fig1.py

# 图 2：SIL 收敛对比
python analysis/fig2.py
```

输出 PDF 保存至 `analysis/figures/`。两个脚本直接读取 TensorBoard event 文件，无需重新训练。

## 超参数

见 [`config.py`](config.py)：

| 参数 | 值 |
|------|-----|
| 隐藏层维度 | 128 |
| 学习率 | 3×10⁻⁴ |
| Adam ε | 10⁻³ |
| 折扣因子 γ | 0.99 |
| 并行环境数 | 4 |
| Rollout 步数 | 5 |
| 熵系数 | 0.01 |
| 价值损失系数 | 0.5 |
| 最大梯度范数 | 0.5 |
| SIL 缓冲区容量 | 10000 |
| SIL 损失权重 | 1.0 |

## 预计训练时间

| 环境 | 步数 | 耗时（M1 Mac） |
|------|------|----------------|
| tiny-2ag（2 智能体） | 80M | ~7 小时 |
| small-4ag（4 智能体） | 150M | ~2 天 |
| small-4ag SIL（4 智能体） | 50M | ~8 小时 |
