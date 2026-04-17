"""Generate SEAL paper-style figures from campaign data.

Replicates the style of Hudson 2022 SEAL paper figures:
  Fig 5: Learning curves per strategy per topology
  Fig 6: Cumulative communication cost over training time
  Fig 7: Evaluation bar charts — Travel Time + Waiting Time per topology
"""
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

# Paper-matching style
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

STRATEGY_COLORS = {
    "Federated": "#1f77b4",
    "Centralized": "#2ca02c",
    "Decentralized": "#ff7f0e",
}
STRATEGY_STYLES = {
    "Federated": "-",
    "Centralized": "--",
    "Decentralized": ":",
}

# =========================================================================
# FIG 5: Learning Curves per topology
# Paper shows 3 subplots (3x3, 5x5, 7x7), each with 3 lines
# We have 1 real convergence curve (FedRL grid-5x5 from ablation).
# For the other strategies, we use the final evaluation reward as a
# horizontal reference line to show where they converge.
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

topos = ["grid-3x3", "grid-5x5"]
topo_labels = ["Grid-3$\\times$3", "Grid-5$\\times$5"]

for idx, (topo, label) in enumerate(zip(topos, topo_labels)):
    ax = axes[idx]

    # Get final eval rewards for each RL strategy as reference lines
    for r in baseline:
        if not r.get("evaluation"):
            continue
        trainer = r["config"]["trainer_type"]
        rtopo = r["config"]["topology"]
        if rtopo != topo or trainer not in ("FedRL", "MARL", "SARL"):
            continue

        agg = r["evaluation"]["campaign"][0]["aggregated"]
        reward = agg["mean_reward"]["mean"]

        strategy_map = {"FedRL": "Federated", "MARL": "Centralized", "SARL": "Decentralized"}
        strat = strategy_map[trainer]

        # Check if we have real training data for this config
        has_curve = False
        for ar in ablation:
            rewards = ar.get("training_rewards")
            if (rewards and len(rewards) > 0
                    and ar["config"]["topology"] == topo
                    and ar["config"]["trainer_type"] == trainer):
                episodes = [ep["episode"] for ep in rewards]
                mean_rewards = [ep["mean_reward"] for ep in rewards]
                ax.plot(episodes, mean_rewards,
                        color=STRATEGY_COLORS[strat],
                        linestyle=STRATEGY_STYLES[strat],
                        linewidth=1.8, label=strat)
                has_curve = True
                break

        if not has_curve:
            # Draw horizontal dashed line at eval reward level
            ax.axhline(y=reward, color=STRATEGY_COLORS[strat],
                       linestyle=STRATEGY_STYLES[strat], linewidth=1.5,
                       alpha=0.7, label=f"{strat} (eval)")

    ax.set_xlabel("Training Episode" if idx == 0 else "Training Episode")
    if idx == 0:
        ax.set_ylabel("Mean Episode Reward")
    ax.set_title(f"({chr(97+idx)}) {label}")
    ax.legend(loc="lower right")

fig.suptitle("Learning Curves with Each Training Approach", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig5_learning_curves.png"), bbox_inches="tight")
plt.close()
print("Fig 5: Learning curves saved")

# =========================================================================
# FIG 6: Cumulative Communication Cost Over Training Time
# Paper shows bytes transmitted cumulatively, 3 subplots
# We compute theoretical cost based on SEAL paper Section III-D
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)

for idx, (topo, label, n_intersections) in enumerate([
    ("grid-3x3", "Grid-3$\\times$3", 9),
    ("grid-5x5", "Grid-5$\\times$5", 25),
]):
    ax = axes[idx]

    total_timesteps = 200000
    timestep_arr = np.arange(0, total_timesteps + 1, 1000)
    policy_size = 50000  # ~50KB per policy network
    obs_size = 64        # 16 floats x 4 bytes
    action_size = 1      # 1 byte per action
    v2i_size = 32        # V2I communication per vehicle per step
    fed_frame = 4000     # aggregation every 4000 steps

    # Centralized: obs + action + V2I every timestep
    cent_per_step = n_intersections * (obs_size + action_size + v2i_size)
    cent_cumulative = timestep_arr * cent_per_step / 1000  # in KB

    # Decentralized: only V2I
    dec_per_step = n_intersections * v2i_size
    dec_cumulative = timestep_arr * dec_per_step / 1000

    # Federated: V2I + periodic policy exchange
    fed_v2i = timestep_arr * n_intersections * v2i_size
    fed_policy = np.array([
        (t // fed_frame) * n_intersections * 2 * policy_size
        for t in timestep_arr
    ])
    fed_cumulative = (fed_v2i + fed_policy) / 1000

    ax.plot(timestep_arr / 1000, cent_cumulative / 1000,
            color=STRATEGY_COLORS["Centralized"],
            linestyle=STRATEGY_STYLES["Centralized"],
            linewidth=1.8, label="Centralized")
    ax.plot(timestep_arr / 1000, dec_cumulative / 1000,
            color=STRATEGY_COLORS["Decentralized"],
            linestyle=STRATEGY_STYLES["Decentralized"],
            linewidth=1.8, label="Decentralized")
    ax.plot(timestep_arr / 1000, fed_cumulative / 1000,
            color=STRATEGY_COLORS["Federated"],
            linestyle=STRATEGY_STYLES["Federated"],
            linewidth=1.8, label="Federated")

    ax.set_xlabel("Time-Steps (k)")
    if idx == 0:
        ax.set_ylabel("Cumulative Comm. Cost (MB)")
    ax.set_title(f"({chr(97+idx)}) {label}")
    ax.legend(loc="upper left")

    # Format x-axis like paper: 0k, 100k, 200k
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels(["0k", "50k", "100k", "150k", "200k"])

fig.suptitle("Communication Cost During Training", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig6_communication_cost.png"), bbox_inches="tight")
plt.close()
print("Fig 6: Communication cost saved")

# =========================================================================
# FIG 7: Evaluation — Travel Time and Waiting Time bar charts
# Paper shows 2 rows (Travel Time, Waiting Time) x 3 cols (topologies)
# Each subplot has grouped bars: one per strategy
# We do 2 rows x 2 cols (we have 2 topologies)
# =========================================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

trainer_order = ["FedRL", "MARL", "SARL", "fixed-time", "max-pressure"]
trainer_display = ["Federated", "Centralized", "Decentralized", "Fixed-Time", "Max-Pressure"]
bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

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
            for r in baseline:
                if not r.get("evaluation"):
                    continue
                if (r["config"]["trainer_type"] == trainer
                        and r["config"]["topology"] == topo):
                    agg = r["evaluation"]["campaign"][0]["aggregated"]
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
            ax.set_title(f"Tested On \"{topo_label}\"")

        # Value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.5,
                    f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

# Legend at the top
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in bar_colors]
fig.legend(handles, trainer_display, loc="upper center",
           ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.05))

fig.suptitle("Evaluation of Trained Policy Networks on Each Road Network",
             fontweight="bold", y=1.08)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig7_evaluation_metrics.png"), bbox_inches="tight")
plt.close()
print("Fig 7: Evaluation metrics saved")

print(f"\nAll SEAL-style figures generated in {FIGURES_DIR}")
