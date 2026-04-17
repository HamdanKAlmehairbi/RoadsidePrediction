"""Generate presentation figures with FedRL/MARL/SARL labels, save to Downloads."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
DOWNLOADS = os.path.expanduser("~/Downloads")

with open(os.path.join(RESULTS_DIR, "campaigns", "baseline", "results.json")) as f:
    baseline = json.load(f)["results"]
with open(os.path.join(RESULTS_DIR, "campaigns", "training-curves", "results.json")) as f:
    training = json.load(f)["results"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {"FedRL": "#1f77b4", "MARL": "#2ca02c", "SARL": "#ff7f0e"}
STYLES = {"FedRL": "-", "MARL": "--", "SARL": ":"}


def get_training_result(trainer, topology):
    for r in training:
        if (r["config"]["trainer_type"] == trainer
                and r["config"]["topology"] == topology
                and r["config"]["name"].endswith("_training")):
            return r
    return None


def get_eval_metrics(result):
    ev = result.get("evaluation")
    if not ev:
        return None
    if "campaign" in ev:
        return ev["campaign"][0].get("aggregated", {})
    return ev


# =============================================================
# FIG 1: Learning Curves
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
topos = ["grid-3x3", "grid-5x5"]
topo_labels = ["Grid-3x3", "Grid-5x5"]

for idx, (topo, label) in enumerate(zip(topos, topo_labels)):
    ax = axes[idx]
    for trainer in ["FedRL", "MARL", "SARL"]:
        r = get_training_result(trainer, topo)
        if not r:
            continue
        rewards = r.get("training_rewards", [])
        if not rewards:
            continue
        episodes = [ep["episode"] for ep in rewards]
        mean_rewards = [ep["mean_reward"] for ep in rewards]
        window = 3
        smoothed = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
        smooth_eps = episodes[window - 1:]
        ax.plot(episodes, mean_rewards, alpha=0.2, color=COLORS[trainer], linewidth=1)
        ax.plot(smooth_eps, smoothed, color=COLORS[trainer],
                linestyle=STYLES[trainer], linewidth=2, label=trainer)
    ax.set_xlabel("Training Episode")
    if idx == 0:
        ax.set_ylabel("Mean Episode Reward")
    ax.set_title("({}) {}".format(chr(97 + idx), label))
    ax.legend(loc="lower right")

fig.suptitle("Learning Curves with Each Training Approach", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS, "fig_learning_curves.png"), bbox_inches="tight")
plt.close()
print("1. Learning curves saved")

# =============================================================
# FIG 2: Evaluation — Travel Time + Waiting Time (2x2)
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

trainer_order = ["FedRL", "MARL", "SARL", "fixed-time"]
trainer_display = ["FedRL", "MARL", "SARL", "Fixed-Time"]
bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

metrics = [
    ("avg_travel_time", "Travel Time (s)"),
    ("avg_waiting_time", "Waiting Time (s)"),
]

for row, (metric_key, metric_label) in enumerate(metrics):
    for col, (topo, topo_label) in enumerate([
        ("grid-3x3", "Grid-3x3"),
        ("grid-5x5", "Grid-5x5"),
    ]):
        ax = axes[row][col]
        means = []
        stds = []
        labels = []
        bcolors = []
        for t_idx, t in enumerate(trainer_order):
            source = training if t in ("FedRL", "MARL", "SARL") else baseline
            for r in source:
                if not r.get("evaluation"):
                    continue
                cfg = r["config"]
                if cfg["trainer_type"] != t or cfg["topology"] != topo:
                    continue
                if t in ("FedRL", "MARL", "SARL") and not cfg["name"].endswith("_training"):
                    continue
                agg = get_eval_metrics(r)
                if not agg or metric_key not in agg:
                    continue
                m = agg[metric_key]
                means.append(m["mean"])
                stds.append(m["std"])
                labels.append(trainer_display[t_idx])
                bcolors.append(bar_colors[t_idx])
                break
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=bcolors,
                      edgecolor="white", linewidth=0.5, alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
        if col == 0:
            ax.set_ylabel(metric_label)
        if row == 0:
            ax.set_title("Tested On \"{}\"".format(topo_label))
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                    "{:.1f}".format(mean), ha="center", va="bottom", fontsize=8, fontweight="bold")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in bar_colors]
fig.legend(handles, trainer_display, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.05))
fig.suptitle("Evaluation of Trained Policy Networks on Each Road Network",
             fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS, "fig_evaluation_metrics.png"), bbox_inches="tight")
plt.close()
print("2. Evaluation metrics (travel + waiting) saved")

# =============================================================
# FIG 3: Waiting Time only (standalone)
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for col, (topo, topo_label) in enumerate([
    ("grid-3x3", "Grid-3x3"),
    ("grid-5x5", "Grid-5x5"),
]):
    ax = axes[col]
    means = []
    stds = []
    labels = []
    bcolors = []
    for t_idx, t in enumerate(trainer_order):
        source = training if t in ("FedRL", "MARL", "SARL") else baseline
        for r in source:
            if not r.get("evaluation"):
                continue
            cfg = r["config"]
            if cfg["trainer_type"] != t or cfg["topology"] != topo:
                continue
            if t in ("FedRL", "MARL", "SARL") and not cfg["name"].endswith("_training"):
                continue
            agg = get_eval_metrics(r)
            if not agg or "avg_waiting_time" not in agg:
                continue
            m = agg["avg_waiting_time"]
            means.append(m["mean"])
            stds.append(m["std"])
            labels.append(trainer_display[t_idx])
            bcolors.append(bar_colors[t_idx])
            break
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=bcolors,
                  edgecolor="white", linewidth=0.5, alpha=0.85, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Avg Waiting Time (s)" if col == 0 else "")
    ax.set_title(topo_label)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                "{:.1f}".format(mean), ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle("Average Waiting Time by Strategy", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS, "fig_waiting_time.png"), bbox_inches="tight")
plt.close()
print("3. Waiting time saved")

print("\nAll figures saved to " + DOWNLOADS)
