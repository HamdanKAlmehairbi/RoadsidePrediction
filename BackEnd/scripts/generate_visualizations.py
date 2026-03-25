"""Generate 4 presentation-ready visualizations from existing campaign data."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
with open(os.path.join(RESULTS_DIR, "campaigns", "baseline", "results.json")) as f:
    baseline = json.load(f)["results"]
with open(os.path.join(RESULTS_DIR, "campaigns", "fedprox-ablation", "results.json")) as f:
    ablation = json.load(f)["results"]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# =========================================================================
# CHART 1: Convergence Curve
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5))

for r in ablation:
    rewards = r.get("training_rewards")
    if rewards and len(rewards) > 0:
        episodes = [ep["episode"] for ep in rewards]
        mean_rewards = [ep["mean_reward"] for ep in rewards]

        window = 5
        smoothed = np.convolve(mean_rewards, np.ones(window) / window, mode="valid")
        smooth_eps = episodes[window - 1 :]

        ax.plot(episodes, mean_rewards, alpha=0.3, color="#2196F3", linewidth=1)
        ax.plot(
            smooth_eps,
            smoothed,
            color="#1565C0",
            linewidth=2.5,
            label="FedRL (grid-5x5) \u2014 smoothed",
        )

ax.set_xlabel("Training Episode", fontsize=12)
ax.set_ylabel("Mean Reward", fontsize=12)
ax.set_title("FedRL Training Convergence \u2014 Grid 5\u00d75", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.annotate(
    f"Start: {mean_rewards[0]:.1f}",
    xy=(1, mean_rewards[0]),
    fontsize=9,
    xytext=(5, mean_rewards[0] + 1),
    arrowprops=dict(arrowstyle="->", color="gray"),
)
ax.annotate(
    f"End: {mean_rewards[-1]:.1f}",
    xy=(50, mean_rewards[-1]),
    fontsize=9,
    xytext=(42, mean_rewards[-1] + 1.5),
    arrowprops=dict(arrowstyle="->", color="gray"),
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart_convergence_fedrl.png"), bbox_inches="tight")
plt.close()
print("1. Convergence curve saved")

# =========================================================================
# CHART 2: Per-Topology Comparison (side by side)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

trainer_labels = {
    "FedRL": "FedRL",
    "MARL": "MARL",
    "SARL": "SARL",
    "fixed-time": "Fixed-Time",
    "max-pressure": "Max-Pressure",
}
colors = {
    "FedRL": "#1565C0",
    "MARL": "#2E7D32",
    "SARL": "#E65100",
    "fixed-time": "#757575",
    "max-pressure": "#6A1B9A",
}

for idx, topo in enumerate(["grid-3x3", "grid-5x5"]):
    ax = axes[idx]
    topo_results = [
        r
        for r in baseline
        if r["config"]["topology"] == topo and r.get("evaluation")
    ]

    names = []
    means = []
    stds = []
    bar_colors = []

    for r in topo_results:
        trainer = r["config"]["trainer_type"]
        agg = r["evaluation"]["campaign"][0]["aggregated"]
        wt = agg["avg_waiting_time"]
        names.append(trainer_labels.get(trainer, trainer))
        means.append(wt["mean"])
        stds.append(wt["std"])
        bar_colors.append(colors.get(trainer, "#999"))

    x = np.arange(len(names))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Avg Waiting Time (s)" if idx == 0 else "", fontsize=11)
    grid_label = topo.replace("grid-", "Grid ").replace("x", "\u00d7")
    ax.set_title(f"{grid_label}", fontsize=13, fontweight="bold")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

fig.suptitle(
    "Average Waiting Time by Training Strategy and Topology",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "chart_per_topology_comparison.png"), bbox_inches="tight"
)
plt.close()
print("2. Per-topology comparison saved")

# =========================================================================
# CHART 3: Communication Cost (Theoretical Model from SEAL paper)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 5))

timesteps = 200000
n_intersections = 9
policy_size = 50000  # ~50KB

# Centralized: obs + action every timestep + V2I
centralized_cost = timesteps * n_intersections * (64 + 1 + 32)

# Decentralized: only V2I
decentralized_cost = timesteps * n_intersections * 32

# Federated: V2I + periodic policy exchange (every 4000 steps)
n_fed_rounds = timesteps // 4000
federated_cost = timesteps * n_intersections * 32 + n_fed_rounds * n_intersections * 2 * policy_size

strategies = ["Centralized\n(MARL)", "Decentralized\n(SARL)", "Federated\n(FedRL)"]
costs = [centralized_cost / 1e6, decentralized_cost / 1e6, federated_cost / 1e6]
bar_colors_comm = ["#2E7D32", "#E65100", "#1565C0"]

bars = ax.bar(
    strategies, costs, color=bar_colors_comm, edgecolor="white",
    linewidth=0.5, alpha=0.85, width=0.5,
)
ax.set_ylabel("Communication Cost (MB)", fontsize=12)
ax.set_title(
    "Estimated Communication Cost During Training \u2014 Grid 3\u00d73",
    fontsize=14,
    fontweight="bold",
)

for bar, cost in zip(bars, costs):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 2,
        f"{cost:.1f} MB",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

reduction = (1 - federated_cost / centralized_cost) * 100
ax.annotate(
    f"{reduction:.1f}% reduction\nvs Centralized",
    xy=(2, costs[2]),
    fontsize=10,
    color="#1565C0",
    fontweight="bold",
    xytext=(2.4, costs[0] * 0.6),
    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart_communication_cost.png"), bbox_inches="tight")
plt.close()
print("3. Communication cost chart saved")

# =========================================================================
# CHART 4: Strategy x Topology Heatmap
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 6))

trainers_order = ["FedRL", "MARL", "SARL", "fixed-time", "max-pressure"]
topos = ["grid-3x3", "grid-5x5"]
trainer_display = ["FedRL", "MARL", "SARL", "Fixed-Time", "Max-Pressure"]
topo_display = ["Grid 3\u00d73", "Grid 5\u00d75"]

matrix = np.zeros((len(trainers_order), len(topos)))

for r in baseline:
    if not r.get("evaluation"):
        continue
    trainer = r["config"]["trainer_type"]
    topo = r["config"]["topology"]
    if trainer in trainers_order and topo in topos:
        row = trainers_order.index(trainer)
        col = topos.index(topo)
        agg = r["evaluation"]["campaign"][0]["aggregated"]
        matrix[row, col] = agg["avg_waiting_time"]["mean"]

im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(topos)))
ax.set_xticklabels(topo_display, fontsize=11)
ax.set_yticks(range(len(trainers_order)))
ax.set_yticklabels(trainer_display, fontsize=11)
ax.set_title(
    "Average Waiting Time (s) \u2014 Strategy \u00d7 Topology",
    fontsize=14,
    fontweight="bold",
)

for i in range(len(trainers_order)):
    for j in range(len(topos)):
        val = matrix[i, j]
        text_color = "white" if val > np.median(matrix) else "black"
        ax.text(
            j, i, f"{val:.1f}s", ha="center", va="center",
            fontsize=13, fontweight="bold", color=text_color,
        )

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Avg Waiting Time (s)", fontsize=11)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "chart_strategy_topology_heatmap.png"), bbox_inches="tight"
)
plt.close()
print("4. Strategy x Topology heatmap saved")

print("\nAll 4 charts generated successfully.")
print(f"Output directory: {FIGURES_DIR}")
