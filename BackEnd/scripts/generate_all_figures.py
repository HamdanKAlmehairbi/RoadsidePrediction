"""Generate all presentation figures from baseline + training-curves campaign data.

Produces:
  1. fig5_learning_curves.png — Convergence curves for FedRL/MARL/SARL on both topologies
  2. fig6_communication_cost.png — Cumulative comm cost over training time
  3. fig7_evaluation_metrics.png — Travel Time + Waiting Time bar charts (2x2)
  4. chart_fedprox_ablation.png — FedProx mu ablation comparison
  5. chart_tod_ablation.png — Time-of-day ablation comparison
  6. chart_strategy_topology_heatmap.png — Strategy x Topology heatmap
  7. chart_combined_comparison.png — All configs combined bar chart
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load all data
with open(os.path.join(RESULTS_DIR, "campaigns", "baseline", "results.json")) as f:
    baseline = json.load(f)["results"]
with open(os.path.join(RESULTS_DIR, "campaigns", "training-curves", "results.json")) as f:
    training = json.load(f)["results"]

# Paper style
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

COLORS = {
    "Federated": "#1f77b4",
    "Centralized": "#2ca02c",
    "Decentralized": "#ff7f0e",
}
STYLES = {
    "Federated": "-",
    "Centralized": "--",
    "Decentralized": ":",
}
TRAINER_MAP = {"FedRL": "Federated", "MARL": "Centralized", "SARL": "Decentralized"}


def get_training_result(trainer_type, topology):
    for r in training:
        if (r["config"]["trainer_type"] == trainer_type
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


# =========================================================================
# FIG 5: Learning Curves — real data for all strategies on both topologies
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
topos = ["grid-3x3", "grid-5x5"]
topo_labels = ["Grid-3$\\times$3", "Grid-5$\\times$5"]

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
        strat = TRAINER_MAP[trainer]

        # Smoothed
        window = 3
        smoothed = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
        smooth_eps = episodes[window - 1:]

        ax.plot(episodes, mean_rewards, alpha=0.2, color=COLORS[strat], linewidth=1)
        ax.plot(smooth_eps, smoothed, color=COLORS[strat],
                linestyle=STYLES[strat], linewidth=2, label=strat)

    ax.set_xlabel("Training Episode")
    if idx == 0:
        ax.set_ylabel("Mean Episode Reward")
    ax.set_title("({}) {}".format(chr(97 + idx), label))
    ax.legend(loc="lower right")

fig.suptitle("Learning Curves with Each Training Approach", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig5_learning_curves.png"), bbox_inches="tight")
plt.close()
print("1. Fig 5: Learning curves saved")

# =========================================================================
# FIG 6: Cumulative Communication Cost
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

for idx, (topo, label, n_int) in enumerate([
    ("grid-3x3", "Grid-3$\\times$3", 9),
    ("grid-5x5", "Grid-5$\\times$5", 25),
]):
    ax = axes[idx]
    total_ts = 200000
    ts_arr = np.arange(0, total_ts + 1, 1000)
    pol_size = 50000
    obs_size = 64
    v2i_size = 32

    cent = ts_arr * n_int * (obs_size + 1 + v2i_size) / 1e6
    dec = ts_arr * n_int * v2i_size / 1e6
    fed_v2i = ts_arr * n_int * v2i_size
    fed_pol = np.array([(t // 4000) * n_int * 2 * pol_size for t in ts_arr])
    fed = (fed_v2i + fed_pol) / 1e6

    ax.plot(ts_arr / 1000, cent, color=COLORS["Centralized"],
            linestyle=STYLES["Centralized"], linewidth=1.8, label="Centralized")
    ax.plot(ts_arr / 1000, dec, color=COLORS["Decentralized"],
            linestyle=STYLES["Decentralized"], linewidth=1.8, label="Decentralized")
    ax.plot(ts_arr / 1000, fed, color=COLORS["Federated"],
            linestyle=STYLES["Federated"], linewidth=1.8, label="Federated")

    ax.set_xlabel("Time-Steps (k)")
    if idx == 0:
        ax.set_ylabel("Cumulative Comm. Cost (MB)")
    ax.set_title("({}) {}".format(chr(97 + idx), label))
    ax.legend(loc="upper left")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels(["0k", "50k", "100k", "150k", "200k"])

fig.suptitle("Communication Cost During Training", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig6_communication_cost.png"), bbox_inches="tight")
plt.close()
print("2. Fig 6: Communication cost saved")

# =========================================================================
# FIG 7: Evaluation — Travel Time + Waiting Time (from training-curves data)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

trainer_order = ["FedRL", "MARL", "SARL", "fixed-time"]
trainer_display = ["Federated", "Centralized", "Decentralized", "Fixed-Time"]
bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

metrics = [
    ("avg_travel_time", "Travel Time (s)"),
    ("avg_waiting_time", "Waiting Time (s)"),
]

for row, (metric_key, metric_label) in enumerate(metrics):
    for col, (topo, topo_label) in enumerate([
        ("grid-3x3", "Grid-3$\\times$3"),
        ("grid-5x5", "Grid-5$\\times$5"),
    ]):
        ax = axes[row][col]
        means = []
        stds = []
        labels = []
        bcolors = []

        for t_idx, trainer in enumerate(trainer_order):
            # Use training-curves data for RL trainers, baseline for fixed/max
            source = training if trainer in ("FedRL", "MARL", "SARL") else baseline
            for r in source:
                if not r.get("evaluation"):
                    continue
                cfg = r["config"]
                if cfg["trainer_type"] != trainer or cfg["topology"] != topo:
                    continue
                # Skip non-baseline training configs (ablations)
                if trainer in ("FedRL", "MARL", "SARL") and not cfg["name"].endswith("_training"):
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
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        if col == 0:
            ax.set_ylabel(metric_label)
        if row == 0:
            ax.set_title('Tested On "{}"'.format(topo_label))

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                    "{:.1f}".format(mean), ha="center", va="bottom", fontsize=7, fontweight="bold")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in bar_colors]
fig.legend(handles, trainer_display, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.05))
fig.suptitle("Evaluation of Trained Policy Networks on Each Road Network",
             fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig7_evaluation_metrics.png"), bbox_inches="tight")
plt.close()
print("3. Fig 7: Evaluation metrics saved")

# =========================================================================
# FIG 8: FedProx Ablation
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

fedprox_configs = []
for r in training:
    cfg = r["config"]
    if cfg["name"] in ("fedrl_grid-3x3_training", "fedprox_mu0.01_grid-3x3", "fedprox_mu0.1_grid-3x3"):
        fedprox_configs.append(r)

# Left: convergence curves
ax = axes[0]
mu_colors = {"fedrl_grid-3x3_training": ("#1f77b4", "$\\mu=0.0$ (FedAvg)"),
             "fedprox_mu0.01_grid-3x3": ("#2ca02c", "$\\mu=0.01$"),
             "fedprox_mu0.1_grid-3x3": ("#d62728", "$\\mu=0.1$")}
for r in fedprox_configs:
    rewards = r.get("training_rewards", [])
    if not rewards:
        continue
    episodes = [ep["episode"] for ep in rewards]
    mean_rewards = [ep["mean_reward"] for ep in rewards]
    color, label = mu_colors[r["config"]["name"]]
    window = 3
    smoothed = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
    ax.plot(episodes, mean_rewards, alpha=0.2, color=color, linewidth=1)
    ax.plot(episodes[window - 1:], smoothed, color=color, linewidth=2, label=label)

ax.set_xlabel("Training Episode")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("(a) FedProx Convergence Curves")
ax.legend()

# Right: evaluation bar chart
ax = axes[1]
names = []
wait_means = []
wait_stds = []
colors_bar = []
for r in fedprox_configs:
    agg = get_eval_metrics(r)
    if not agg or "avg_waiting_time" not in agg:
        continue
    color, label = mu_colors[r["config"]["name"]]
    names.append(label)
    wait_means.append(agg["avg_waiting_time"]["mean"])
    wait_stds.append(agg["avg_waiting_time"]["std"])
    colors_bar.append(color)

x = np.arange(len(names))
bars = ax.bar(x, wait_means, yerr=wait_stds, capsize=5, color=colors_bar,
              edgecolor="white", alpha=0.85, width=0.5)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Avg Waiting Time (s)")
ax.set_title("(b) FedProx Evaluation")
for bar, mean in zip(bars, wait_means):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
            "{:.1f}".format(mean), ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle("FedProx Ablation Study \u2014 Grid 3\u00d73", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart_fedprox_ablation.png"), bbox_inches="tight")
plt.close()
print("4. FedProx ablation saved")

# =========================================================================
# FIG 9: Time-of-Day Ablation
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

tod_configs = []
for r in training:
    cfg = r["config"]
    if cfg["name"] in ("fedrl_grid-3x3_training", "tod_with_encoding_grid-3x3"):
        tod_configs.append(r)

# Left: convergence curves
ax = axes[0]
tod_styles = {"fedrl_grid-3x3_training": ("#1f77b4", "Fixed Demand"),
              "tod_with_encoding_grid-3x3": ("#d62728", "ToD + Time Encoding")}
for r in tod_configs:
    rewards = r.get("training_rewards", [])
    if not rewards:
        continue
    episodes = [ep["episode"] for ep in rewards]
    mean_rewards = [ep["mean_reward"] for ep in rewards]
    color, label = tod_styles[r["config"]["name"]]
    window = 3
    smoothed = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
    ax.plot(episodes, mean_rewards, alpha=0.2, color=color, linewidth=1)
    ax.plot(episodes[window - 1:], smoothed, color=color, linewidth=2, label=label)

ax.set_xlabel("Training Episode")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("(a) Training Convergence")
ax.legend()

# Right: evaluation
ax = axes[1]
names = []
wait_means = []
wait_stds = []
colors_bar = []
for r in tod_configs:
    agg = get_eval_metrics(r)
    if not agg or "avg_waiting_time" not in agg:
        continue
    color, label = tod_styles[r["config"]["name"]]
    names.append(label)
    wait_means.append(agg["avg_waiting_time"]["mean"])
    wait_stds.append(agg["avg_waiting_time"]["std"])
    colors_bar.append(color)

x = np.arange(len(names))
bars = ax.bar(x, wait_means, yerr=wait_stds, capsize=5, color=colors_bar,
              edgecolor="white", alpha=0.85, width=0.4)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Avg Waiting Time (s)")
ax.set_title("(b) Evaluation Results")
for bar, mean in zip(bars, wait_means):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
            "{:.1f}".format(mean), ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle("Time-of-Day Ablation Study \u2014 Grid 3\u00d73", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart_tod_ablation.png"), bbox_inches="tight")
plt.close()
print("5. ToD ablation saved")

# =========================================================================
# FIG 10: Strategy x Topology Heatmap (from fresh training data)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 6))

trainers_hm = ["FedRL", "MARL", "SARL", "fixed-time"]
topos_hm = ["grid-3x3", "grid-5x5"]
display_t = ["Federated", "Centralized", "Decentralized", "Fixed-Time"]
display_tp = ["Grid 3\u00d73", "Grid 5\u00d75"]

matrix = np.zeros((len(trainers_hm), len(topos_hm)))
for t_idx, trainer in enumerate(trainers_hm):
    for tp_idx, topo in enumerate(topos_hm):
        source = training if trainer in ("FedRL", "MARL", "SARL") else baseline
        for r in source:
            cfg = r["config"]
            if cfg["trainer_type"] != trainer or cfg["topology"] != topo:
                continue
            if trainer in ("FedRL", "MARL", "SARL") and not cfg["name"].endswith("_training"):
                continue
            agg = get_eval_metrics(r)
            if agg and "avg_waiting_time" in agg:
                matrix[t_idx, tp_idx] = agg["avg_waiting_time"]["mean"]
                break

im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(topos_hm)))
ax.set_xticklabels(display_tp, fontsize=11)
ax.set_yticks(range(len(trainers_hm)))
ax.set_yticklabels(display_t, fontsize=11)
ax.set_title("Average Waiting Time (s) \u2014 Strategy \u00d7 Topology", fontsize=14, fontweight="bold")

for i in range(len(trainers_hm)):
    for j in range(len(topos_hm)):
        val = matrix[i, j]
        text_color = "white" if val > np.median(matrix) else "black"
        ax.text(j, i, "{:.1f}s".format(val), ha="center", va="center",
                fontsize=13, fontweight="bold", color=text_color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Avg Waiting Time (s)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart_strategy_topology_heatmap.png"), bbox_inches="tight")
plt.close()
print("6. Heatmap saved")

# =========================================================================
# FIG 11: Combined comparison — all configs in one chart
# =========================================================================
fig, ax = plt.subplots(figsize=(14, 5))

all_configs = []
# Baseline training configs
for trainer in ["FedRL", "MARL", "SARL"]:
    r = get_training_result(trainer, "grid-3x3")
    if r:
        all_configs.append((TRAINER_MAP[trainer], r))

# Fixed-time and max-pressure from baseline
for r in baseline:
    cfg = r["config"]
    if cfg["topology"] == "grid-3x3" and cfg["trainer_type"] == "fixed-time":
        all_configs.append(("Fixed-Time", r))

# Ablation configs
for r in training:
    cfg = r["config"]
    if cfg["name"] == "fedprox_mu0.01_grid-3x3":
        all_configs.append(("FedProx\n$\\mu$=0.01", r))
    elif cfg["name"] == "fedprox_mu0.1_grid-3x3":
        all_configs.append(("FedProx\n$\\mu$=0.1", r))
    elif cfg["name"] == "tod_with_encoding_grid-3x3":
        all_configs.append(("ToD +\nEncoding", r))

names = []
wait_means = []
wait_stds = []
combined_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd",
                   "#17becf", "#bcbd22", "#e377c2"]

for label, r in all_configs:
    agg = get_eval_metrics(r)
    if not agg or "avg_waiting_time" not in agg:
        continue
    names.append(label)
    wait_means.append(agg["avg_waiting_time"]["mean"])
    wait_stds.append(agg["avg_waiting_time"]["std"])

x = np.arange(len(names))
bars = ax.bar(x, wait_means, yerr=wait_stds, capsize=4,
              color=combined_colors[:len(names)], edgecolor="white", alpha=0.85, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Avg Waiting Time (s)", fontsize=12)
ax.set_title("Combined Comparison \u2014 All Strategies and Extensions (Grid 3\u00d73)",
             fontsize=13, fontweight="bold")

for bar, mean in zip(bars, wait_means):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
            "{:.1f}".format(mean), ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart_combined_comparison.png"), bbox_inches="tight")
plt.close()
print("7. Combined comparison saved")

print("\nAll 7 figures generated in " + FIGURES_DIR)
