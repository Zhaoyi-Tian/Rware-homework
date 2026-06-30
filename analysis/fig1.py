"""Figure 1: 综合对比 (2行×3列)。训练曲线 + 熵曲线 + 温度扫描。"""

import os, glob, sys, warnings
import numpy as np
import torch, gymnasium as gym
import rware  # noqa

# 确保从 rware-training 根目录导入
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from config import Config
from algorithms.iac import IAC
from algorithms.snac import SNAC
from algorithms.seac import SEAC
from algorithms.seac_pooled import SEACPooled

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
LOG_DIR = os.path.join(BASE, "..", "logs")
os.makedirs(FIG_DIR, exist_ok=True)

ENVS = [("tiny-2ag", "rware-tiny-2ag-v2"), ("small-4ag", "rware-small-4ag-v2")]
ALGOS = ["IAC", "SNAC", "SEAC", "SEAC-Pooled"]
ALGO_CLS = {"IAC": IAC, "SNAC": SNAC, "SEAC": SEAC, "SEAC-Pooled": SEACPooled}
COLORS = {"IAC": "#4C72B0", "SNAC": "#DD8452", "SEAC": "#55A868", "SEAC-Pooled": "#8172B3"}
TEMPERATURES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
EVAL_SEEDS = 10

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "font.weight": "medium",
    "axes.labelsize": 11, "axes.labelweight": "medium",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.5,
    "axes.facecolor": "#EAEAF2", "axes.edgecolor": "none", "axes.grid": True,
    "grid.color": "white", "grid.linestyle": "-", "grid.linewidth": 1.0,
    "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "none",
    "legend.framealpha": 0.8, "legend.fontsize": 9,
})


def load_algo(algo_name, env_tag, env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space[0].shape[0]
    n_actions = int(env.action_space[0].n)
    n_agents = env.unwrapped.n_agents
    env.close()
    cfg = Config(env_name=env_name, algorithm=algo_name.lower().replace("-", "_"))
    algo = ALGO_CLS[algo_name](cfg, obs_dim, n_actions, n_agents)
    ckpt = f"checkpoints/{algo_name.lower().replace('-','_')}_rware-{env_tag}-v2_seed42"
    if not os.path.exists(ckpt + "/model.pt"):
        ckpt = f"checkpoints/{algo_name.lower().replace('-','_')}_rware_{env_tag}_seed42"
    algo.load(ckpt + "/model.pt")
    return algo


def eval_temp(algo, env_name, T, n_episodes, rng):
    """返回 (rewards_array, mean_entropy)。"""
    env = gym.make(env_name)
    rewards = np.zeros(n_episodes)
    entropies = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done, truncated = False, False
        while not (done or truncated):
            ot = [torch.from_numpy(o).float().unsqueeze(0) for o in obs]
            with torch.inference_mode():
                if T == 0.0:
                    acts = [int(algo.networks[i].act(ot[i], deterministic=True)[1].item())
                            for i in range(algo.n_agents)]
                else:
                    acts = [int(algo.networks[i].act(ot[i], deterministic=False, temperature=T)[1].item())
                            for i in range(algo.n_agents)]
                # 记录当前策略的有效熵（T=0 时 argmax→H=0）
                if T == 0.0:
                    entropies.append(0.0)
                else:
                    for i in range(algo.n_agents):
                        _, logits = algo.networks[i].forward(ot[i])
                        logits = logits / T
                        lp = torch.log_softmax(logits, -1)
                        p = lp.exp()
                        H = -(p * lp).sum(-1).mean().item()
                        entropies.append(H)
            obs, r, done, truncated, info = env.step(acts)
            rewards[ep] += sum(r)
    env.close()
    return rewards, np.mean(entropies) if entropies else 0.0


def read_tb(log_dir, tag):
    if not os.path.isdir(log_dir): return np.array([]), np.array([])
    ea = EventAccumulator(log_dir); ea.Reload()
    if tag not in ea.Tags().get("scalars", {}): return np.array([]), np.array([])
    e = ea.Scalars(tag)
    return np.array([x.step for x in e]), np.array([x.value for x in e])


def smooth(y, w=50):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) > w else y

def smooth_light(y, w=3):
    if len(y) <= w: return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def collect_train(env_tag):
    result = {}
    for algo_name in ALGOS:
        key = algo_name.lower().replace("-", "_")
        ld = f"{LOG_DIR}/{key}_rware-{env_tag}-v2_seed42"
        if not os.path.isdir(ld):
            alt = glob.glob(f"{LOG_DIR}/{key}_{env_tag}*")
            ld = alt[0] if alt else None
        if not ld: continue
        d = {}
        for tag, name in [("train/episode_reward", "reward"), ("train/entropy", "entropy")]:
            s, v = read_tb(ld, tag)
            if len(s): d[name] = (s, v)
        if d: result[algo_name] = d
    return result


def collect_temp(env_tag, env_name, rng):
    result = {}
    for algo_name in ALGOS:
        try: algo = load_algo(algo_name, env_tag, env_name)
        except Exception as e: print(f"SKIP {algo_name} {env_tag}: {e}"); continue
        d = {}
        for T in TEMPERATURES:
            r, h = eval_temp(algo, env_name, T, EVAL_SEEDS, rng)
            d[T] = (np.mean(r), np.std(r), h)  # (reward_mean, reward_std, entropy)
            print(f"  {algo_name:12s} {env_tag} T={T:<5} H={h:.3f} reward={np.mean(r):5.1f}")
        result[algo_name] = d
    return result


def plot(train_data, temp_data):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    sub_labels = [
        ["(a) tiny-2ag: Train Reward", "(b) tiny-2ag: Train Entropy", "(c) tiny-2ag: Reward vs Entropy"],
        ["(d) small-4ag: Train Reward", "(e) small-4ag: Train Entropy", "(f) small-4ag: Reward vs Entropy"],
    ]

    for row, (env_tag, _) in enumerate(ENVS):
        td = train_data.get(env_tag, {})
        tmpd = temp_data.get(env_tag, {})

        ax = axes[row, 0]
        for a in ALGOS:
            if a in td and "reward" in td[a]:
                s, v = td[a]["reward"]; x = s / 1e6; y = smooth(v); x = x[:len(y)]
                ax.plot(x, y, color=COLORS[a], label=a, alpha=0.9)
        ax.set_xlabel("Environment Steps"); ax.set_ylabel("Returns"); ax.legend(loc="lower right")

        ax = axes[row, 1]
        for a in ALGOS:
            if a in td and "entropy" in td[a]:
                s, v = td[a]["entropy"]; x = s / 1e6; y = smooth(v); x = x[:len(y)]
                ax.plot(x, y, color=COLORS[a], label=a, alpha=0.9)
        ax.set_xlabel("Environment Steps"); ax.set_ylabel("Entropy"); ax.legend(loc="lower right")

        ax = axes[row, 2]
        for a in ALGOS:
            if a in tmpd:
                pts = sorted([(tmpd[a][t][2], tmpd[a][t][0], tmpd[a][t][1]) for t in sorted(tmpd[a].keys())])
                hs = np.array([p[0] for p in pts])
                ms = smooth_light(np.array([p[1] for p in pts]))
                ss = np.array([p[2] for p in pts])
                ax.plot(hs, ms, color=COLORS[a], label=a, alpha=0.9)
                ax.fill_between(hs, ms - ss, ms + ss, color=COLORS[a], alpha=0.15)
        ax.set_xlabel("Entropy H(π)"); ax.set_ylabel("Returns"); ax.legend(loc="upper right")

        for col in range(3):
            axes[row, col].set_title(sub_labels[row][col], y=-0.28, fontsize=11, fontweight="normal")

    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure1.pdf")
    plt.close(fig)
    print(f"Saved: {FIG_DIR}/figure1.pdf")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    train_data, temp_data = {}, {}
    for env_tag, env_name in ENVS:
        print(f"\n=== {env_tag} — training curves ===")
        train_data[env_tag] = collect_train(env_tag)
        print(f"=== {env_tag} — temperature sweep ===")
        temp_data[env_tag] = collect_temp(env_tag, env_name, rng)
    plot(train_data, temp_data)
