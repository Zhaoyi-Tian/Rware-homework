"""Figure 2: SIL 收敛速率对比 — small-4ag 50M 训练曲线。"""

import os, glob, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
LOG_DIR = os.path.join(BASE, "..", "logs", "sil")
os.makedirs(FIG_DIR, exist_ok=True)

ALGOS = ["IAC", "IAC-SIL", "SEAC", "SEAC-SIL"]
ENV_TAG = "small-4ag"
COLORS = {"IAC": "#4C72B0", "IAC-SIL": "#55A868", "SEAC": "#DD8452", "SEAC-SIL": "#C44E52"}

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


def read_tb(log_dir, tag):
    if not os.path.isdir(log_dir): return np.array([]), np.array([])
    ea = EventAccumulator(log_dir); ea.Reload()
    if tag not in ea.Tags().get("scalars", {}): return np.array([]), np.array([])
    e = ea.Scalars(tag)
    return np.array([x.step for x in e]), np.array([x.value for x in e])


def smooth(y, w=50):
    return np.convolve(y, np.ones(w)/w, mode="valid") if len(y) > w else y


def collect_train():
    result = {}
    for algo_name in ALGOS:
        key = algo_name.lower().replace("-", "_")
        ld = f"{LOG_DIR}/{key}_rware-{ENV_TAG}-v2_seed42"
        if not os.path.isdir(ld):
            alt = glob.glob(f"{LOG_DIR}/{key}_*")
            ld = alt[0] if alt else None
        if not ld: continue
        d = {}
        for tag, name in [("train/episode_reward", "reward"), ("train/entropy", "entropy")]:
            s, v = read_tb(ld, tag)
            if len(s): d[name] = (s, v)
        if d: result[algo_name] = d
    return result


def plot(train_data):
    fig, ax = plt.subplots(figsize=(7, 4))

    for a in ALGOS:
        if a in train_data and "reward" in train_data[a]:
            s, v = train_data[a]["reward"]
            x = s / 1e6; y = smooth(v); x = x[:len(y)]
            ax.plot(x, y, color=COLORS[a], label=a, alpha=0.9)
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Episode Reward")
    ax.legend(loc="lower right")
    ax.set_title("Small-4ag: SIL Convergence Comparison", fontsize=11)

    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/figure2.pdf")
    plt.close(fig)
    print(f"Saved: {FIG_DIR}/figure2.pdf")


if __name__ == "__main__":
    print("Reading SIL training logs...")
    train_data = collect_train()
    plot(train_data)
