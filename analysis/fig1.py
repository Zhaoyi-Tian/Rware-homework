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
TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0]
EVAL_SEEDS = 5

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.3,
    "axes.facecolor": "#EAEAF2", "axes.edgecolor": "none", "axes.grid": True,
    "grid.color": "white", "grid.linestyle": "-", "grid.linewidth": 1.0,
    "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "none",
    "legend.framealpha": 0.8, "legend.fontsize": 8.5,
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
    env = gym.make(env_name)
    rewards = np.zeros(n_episodes)
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
            obs, r, done, truncated, info = env.step(acts)
            rewards[ep] += sum(r)
    env.close()
    return rewards


def read_tb(log_dir, tag):
    if not os.path.isdir(log_dir): return np.array([]), np.array([])
    ea = EventAccumulator(log_dir); ea.Reload()
    if tag not in ea.Tags().get("scalars", {}): return np.array([]), np.array([])
    e = ea.Scalars(tag)
    return np.array([x.step for x in e]), np.array([x.value for x in e])


def smooth(y, w=50):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) > w else y


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
            r = eval_temp(algo, env_name, T, EVAL_SEEDS, rng)
            d[T] = (np.mean(r), np.std(r))
            print(f"  {algo_name:12s} {env_tag} T={T:<5} mean={np.mean(r):5.1f}")
        result[algo_name] = d
    return result


def plot(train_data, temp_data):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    sub_labels = [
        ["(a) tiny-2ag: Train Reward", "(b) tiny-2ag: Train Entropy", "(c) tiny-2ag: Temperature Sweep"],
        ["(d) small-4ag: Train Reward", "(e) small-4ag: Train Entropy", "(f) small-4ag: Temperature Sweep"],
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
                ts = sorted(tmpd[a].keys())
                ms = np.array([tmpd[a][t][0] for t in ts])
                ss = np.array([tmpd[a][t][1] for t in ts])
                ax.plot(ts, ms, color=COLORS[a], label=a, alpha=0.9)
                ax.fill_between(ts, ms - ss, ms + ss, color=COLORS[a], alpha=0.15)
        ax.set_xlabel("Temperature"); ax.set_ylabel("Returns"); ax.legend(loc="lower right")

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
