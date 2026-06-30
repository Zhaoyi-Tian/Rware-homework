# RWARE Multi-Agent RL: SEAC Training, Analysis & Improvements

Course project on multi-agent reinforcement learning — reproducing and analyzing the SEAC algorithm in the RWARE robotic warehouse environment, with proposed improvements.

**Full report (Chinese, ICML 2022 format):** [report.pdf](report.pdf)

## Overview

Re-implemented IAC, SNAC, and SEAC baselines from scratch (PyTorch A2C), trained on tiny-2ag (80M steps) and small-4ag (150M steps). Key contributions:

1. **Entropy asymmetry in SEAC** — showed that SEAC's cross-agent loss lacks entropy regularization, causing policy to collapse to near-deterministic (H≈0.21 vs IAC's H≈0.76), inflating training reward
2. **SEAC-Pooled** — added importance-weighted entropy to cross-agent terms, eliminating the asymmetry for fair comparison
3. **Temperature-sweep evaluation** — decoupled policy sharpness (entropy) from action ranking quality, enabling fair cross-algorithm comparison
4. **SEAC-SIL** — combined cross-agent experience sharing with self-imitation learning for faster convergence

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `iac` | Independent Actor-Critic (per-agent networks) |
| `snac` | Shared Network Actor-Critic |
| `seac` | Shared Experience Actor-Critic (cross-agent importance sampling) |
| `seac_pooled` | SEAC + importance-weighted entropy on cross-agent data |
| `iac_sil` | IAC + Self-Imitation Learning |
| `seac_sil` | SEAC + Self-Imitation Learning |

All implementations in [`algorithms/`](algorithms/).

## Installation

```bash
# Create conda environment
conda create -n Rware python=3.9 -y
conda activate Rware

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Single algorithm
python main.py --algorithm seac --env rware-tiny-2ag-v2 --total-steps 80000000 --seed 42

# Batch run (all baselines, both environments)
bash run_full.sh

# SIL comparison (small-4ag, 50M steps)
bash run_sil.sh
```

**CLI arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | (required) | `iac` / `snac` / `seac` / `seac_pooled` / `iac_sil` / `seac_sil` |
| `--env` | (required) | `rware-tiny-2ag-v2` / `rware-small-4ag-v2` |
| `--total-steps` | (required) | Total environment steps |
| `--seed` | 42 | Random seed |
| `--lr` | 3e-4 | Learning rate |
| `--resume` | — | Path to checkpoint for resuming |

Checkpoints are saved to `checkpoints/<algo>_<env>_seed<seed>/`. Logs go to `logs/<algo>_<env>_seed<seed>/`.

## Viewing Training Logs

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006`. Each algorithm+environment combination appears as a separate run. Key metrics:

- `agent0/episode_reward` — episodic reward (smoothed via TB's built-in smoothing)
- `agent0/entropy` — policy entropy during training
- `agent0/policy_loss`, `agent0/value_loss`, `agent0/total_loss`

For SIL variants, logs are under `logs/sil/`:
```bash
tensorboard --logdir logs/sil
```

## Generating Figures

```bash
# Figure 1: baseline training curves + entropy + temperature sweep
python analysis/fig1.py

# Figure 2: SIL convergence comparison
python analysis/fig2.py
```

Output PDFs are saved to `analysis/figures/`. Both scripts read TensorBoard event files directly (no need to re-run training if logs are present).

## Configuration

All hyperparameters in [`config.py`](config.py):

| Parameter | Value |
|-----------|-------|
| Hidden dim | 128 |
| Learning rate | 3×10⁻⁴ |
| Adam ε | 10⁻³ |
| Discount γ | 0.99 |
| Parallel envs | 4 |
| Rollout steps | 5 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Max grad norm | 0.5 |
| SIL capacity | 10000 |
| SIL coefficient | 1.0 |

## Estimated Training Time

| Environment | Steps | Wall Time (M1 Mac) |
|-------------|-------|---------------------|
| tiny-2ag (2 agents) | 80M | ~7 hours |
| small-4ag (4 agents) | 150M | ~2 days |
| small-4ag SIL (4 agents) | 50M | ~8 hours |

## Citation

```bibtex
@misc{tian2025rware,
  author = {Zhaoyi Tian},
  title  = {RWARE Multi-Agent RL: SEAC Training, Analysis \& Improvements},
  year   = {2025},
  url    = {https://github.com/Zhaoyi-Tian/Rware-seac-analysis}
}
```
