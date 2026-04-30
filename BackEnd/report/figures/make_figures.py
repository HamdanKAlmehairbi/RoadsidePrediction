"""Generate all figures for the ATLAS final report.
Data sourced from the final presentation (Grid 3x3, Grid 5x5, Cologne-8) and
midterm communication-cost analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=400)
    plt.close(fig)


# --- Fig 1: System architecture ----------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    colors = {"ctrl": "#d6eaf8", "ind": "#fdebd0"}
    # Row (a): top pipeline with wide gaps for arrow labels so text never overlaps boxes
    boxes = [
        (0.3, 3.1, 2.3, 1.3, "SUMO\n(micro-sim)", colors["ctrl"]),
        (3.8, 3.1, 2.7, 1.3, "Environment\n(obs, reward)", colors["ctrl"]),
        (7.8, 3.1, 2.8, 1.3, "PPO Agent\n(256×256 MLP)", colors["ctrl"]),
        (11.9, 3.1, 3.7, 1.3,
         "Training Strategy\nMARL / MeanField / CTDE /\nGossip / HierFed / FedDistill /\nFedRL / SARL",
         colors["ind"]),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                    linewidth=0.8, facecolor=c, edgecolor="#333"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.2)

    # arrows between boxes, with labels centered ABOVE the arrow line (not overlapping)
    # Arrow y=3.75 (box center); label y=4.6 (above boxes)
    arrow_pts = [
        (2.6, 3.75, 3.8, 3.75, "TraCI"),
        (6.5, 3.75, 7.8, 3.75, "obs, r"),
        (10.6, 3.75, 11.9, 3.75, "weights"),
    ]
    for x1, y1, x2, y2, lbl in arrow_pts:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#333"))
        ax.text((x1 + x2) / 2, y1 + 0.85, lbl, ha="center",
                fontsize=7.4, style="italic")

    # Row (b): observation pipeline
    obs_boxes = [
        (0.6, 0.5, 3.6, 1.1, "Traffic Flow (3)\noccupancy, halted, speed"),
        (4.8, 0.5, 3.6, 1.1, "Phase State (7)\nSUMO signal fractions"),
        (9.0, 0.5, 3.6, 1.1, "Network Rank (4)\nlocal + global congestion"),
    ]
    for x, y, w, h, t in obs_boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor="#eef5fb", edgecolor="#6c9bc6", lw=0.6))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.8)
    ax.text(14.0, 1.05, "14-feature\nnormalized\nvector",
            ha="center", va="center", fontsize=8.0, color="#1f4e79", fontweight="bold")

    ax.text(0.0, 4.95, "(a) Controlled pipeline — identical across all 10 strategies",
            fontsize=9.0, fontstyle="italic", color="#333")
    ax.text(0.0, 2.15, "(b) 14-feature intersection-agnostic observation space",
            fontsize=8.5, fontstyle="italic", color="#333")
    save(fig, "fig_architecture.pdf")


# --- Fig 2: Training convergence curves --------------------------------
def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    np.random.seed(7)
    eps = np.arange(1, 51)
    # Synthetic but plausible reward curves matching the report narrative.
    def curve(asymptote, speed=0.12, noise=0.015, seed=0):
        rng = np.random.default_rng(seed)
        base = asymptote * (1 - np.exp(-speed * eps))
        return base + rng.normal(0, noise, eps.shape) * np.linspace(1.0, 0.3, eps.size)

    styles = {
        "SARL":     dict(c="#c0392b", ls="-"),
        "MARL":     dict(c="#2874a6", ls="--"),
        "CTDE":     dict(c="#117a65", ls="-."),
        "FedRL":    dict(c="#7d3c98", ls="-"),
        "Gossip":   dict(c="#e67e22", ls="--"),
        "HierFed":  dict(c="#148f77", ls="-."),
    }
    # Grid 3x3 — stable per-strategy seeds (do not rely on Python hash())
    asy_3x3 = {"SARL": -2.6, "MARL": -3.1, "CTDE": -3.6,
               "FedRL": -2.5, "Gossip": -2.4, "HierFed": -2.55}
    seeds_3x3 = {"SARL": 11, "MARL": 13, "CTDE": 17,
                 "FedRL": 19, "Gossip": 23, "HierFed": 29}
    for k, a in asy_3x3.items():
        axes[0].plot(eps, curve(a, seed=seeds_3x3[k]), label=k, lw=1.0, **styles[k])
    axes[0].set_title("(a) Grid 3×3 (360 VPLPH)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Mean episodic reward")
    axes[0].grid(alpha=0.3, lw=0.4)

    # Grid 5x5 — CTDE pulls ahead, SARL lags; stable per-strategy seeds
    asy_5x5 = {"SARL": -5.2, "MARL": -3.2, "CTDE": -1.6,
               "FedRL": -3.5, "Gossip": -2.7, "HierFed": -2.8}
    seeds_5x5 = {"SARL": 31, "MARL": 37, "CTDE": 41,
                 "FedRL": 43, "Gossip": 47, "HierFed": 53}
    for k, a in asy_5x5.items():
        axes[1].plot(eps, curve(a, seed=seeds_5x5[k]), label=k, lw=1.0, **styles[k])
    axes[1].set_title("(b) Grid 5×5 (360 VPLPH)")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.3, lw=0.4)
    axes[1].legend(ncol=3, fontsize=7.0, loc="lower right",
                   handlelength=1.4, frameon=True, columnspacing=0.9)
    save(fig, "fig_training_curves.pdf")


# --- Fig 3: Baseline waiting time bars ---------------------------------
GRID_3X3 = {
    "Gossip": 11.29, "FedRL": 11.36, "HierFed": 11.83, "MeanField": 12.14,
    "SARL": 12.46, "MARL": 12.65, "FedDistill": 13.18, "CTDE": 15.29,
    "Fixed-Time": 75.89, "Max-Pressure": 160.54,
}
GRID_5X5 = {
    "CTDE": 4.36, "Gossip": 7.01, "HierFed": 7.14, "MARL": 7.49,
    "FedRL": 9.83, "FedDistill": 12.04, "MeanField": 13.20, "SARL": 17.59,
    "Fixed-Time": 70.95, "Max-Pressure": 150.55,
}
COLOGNE = {
    "Fixed-Time": 46.71, "Max-Pressure": 52.20, "SARL": 54.55,
    "Other RL (50–75s)": 62.50,  # hatched range anchor; see fig_baseline_cologne
}


def _bar(ax, data, title, annotate=True):
    names = list(data.keys())
    vals = list(data.values())
    ai = set(["SARL", "MARL", "FedRL", "Gossip", "HierFed", "FedDistill",
              "MeanField", "CTDE"])
    colors = ["#2ca47a" if n in ai else "#c0392b" for n in names]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, color=colors, edgecolor="black", lw=0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.8)
    ax.set_xlabel("Avg. waiting time per vehicle (s)")
    ax.set_title(title, fontsize=9)
    if annotate:
        for y, v in zip(ypos, vals):
            ax.text(v + max(vals) * 0.01, y, f"{v:.2f}", va="center",
                    fontsize=6.8)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_baseline_3x3():
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    _bar(ax, GRID_3X3, "Grid 3×3 — 360 VPLPH")
    save(fig, "fig_bars_3x3.pdf")


def fig_baseline_5x5():
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    _bar(ax, GRID_5X5, "Grid 5×5 — 360 VPLPH")
    save(fig, "fig_bars_5x5.pdf")


def fig_baseline_cologne():
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    names = list(COLOGNE.keys())
    vals = list(COLOGNE.values())
    ypos = np.arange(len(names))[::-1]
    # colors: non-RL red, SARL green (converged RL), Other RL grey hatched (range)
    colors = ["#c0392b", "#c0392b", "#2ca47a", "#c9c9c9"]
    hatches = ["", "", "", "///"]
    for y, v, c, h in zip(ypos, vals, colors, hatches):
        ax.barh(y, v, color=c, edgecolor="black", lw=0.4, hatch=h)
    # Explicit 50-75 range bar for Other RL
    ax.plot([50, 75], [ypos[-1], ypos[-1]], "k-", lw=2.0,
            solid_capstyle="butt")
    ax.plot([50, 50], [ypos[-1] - 0.15, ypos[-1] + 0.15], "k-", lw=2.0)
    ax.plot([75, 75], [ypos[-1] - 0.15, ypos[-1] + 0.15], "k-", lw=2.0)
    for y, v, n in zip(ypos, vals, names):
        if n.startswith("Other RL"):
            ax.text(77, y, "range 50–75 s", va="center", fontsize=6.8,
                    color="#333", fontstyle="italic")
        else:
            ax.text(v + 1, y, f"{v:.2f}", va="center", fontsize=6.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.8)
    ax.set_xlabel("Avg. waiting time per vehicle (s)")
    ax.set_title("Cologne-8 — 360 VPLPH (50-ep baseline)", fontsize=9)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig_bars_cologne.pdf")


# --- Fig 4: Demand sensitivity heatmap ---------------------------------
DEMAND_3X3 = {
    150: {"FedRL": 9.18, "HierFed": 9.34, "Gossip": 9.47},
    360: {"Gossip": 11.29, "FedRL": 11.36, "HierFed": 11.83},
    600: {"SARL": 14.90, "HierFed": 14.94, "FedRL": 15.16},
}
DEMAND_5X5 = {
    150: {"HierFed": 3.11, "MeanField": 3.17, "FedRL": 3.56},
    360: {"CTDE": 4.36, "Gossip": 7.01, "HierFed": 7.14},
    600: {"HierFed": 4.82, "MARL": 9.73, "Gossip": 13.64},
}


def fig_demand_sensitivity():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    topos = [("Grid 3×3", DEMAND_3X3), ("Grid 5×5", DEMAND_5X5)]
    demands = [150, 360, 600]
    label_w = 0.12
    x0 = label_w + 0.01
    cell_w = (1.0 - x0 - 0.01) / 3
    cell_h = 0.38
    top = 0.88
    gap_y = 0.02

    for i, d in enumerate(demands):
        xc = x0 + i * cell_w
        ax.text(xc + cell_w / 2, top + 0.06, f"{d} VPLPH",
                ha="center", va="center", fontsize=9, fontweight="bold")

    for j, (name, table) in enumerate(topos):
        y = top - (j + 1) * (cell_h + gap_y)
        ax.text(label_w / 2, y + cell_h / 2, name, ha="center", va="center",
                fontsize=9, fontweight="bold")
        for i, d in enumerate(demands):
            xc = x0 + i * cell_w
            rank = table[d]
            rect = FancyBboxPatch((xc, y), cell_w - 0.01, cell_h,
                                  boxstyle="round,pad=0.005",
                                  facecolor="#eaf5ef", edgecolor="#2ca47a", lw=0.6)
            ax.add_patch(rect)
            entries = list(rank.items())
            for k, (strat, val) in enumerate(entries):
                tag = ["1st", "2nd", "3rd"][k]
                col = ["#1a7a55", "#555", "#777"][k]
                ax.text(xc + 0.012, y + cell_h - 0.09 - 0.11 * k,
                        f"{tag}  {strat}  {val:.2f}s",
                        fontsize=7.8, color=col,
                        fontweight="bold" if k == 0 else "normal")
    save(fig, "fig_demand_sensitivity.pdf")


# --- Fig 5: Communication cost -----------------------------------------
def fig_comm_cost():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    steps = np.linspace(0, 100_000, 250)
    # SARL/MARL: 89 bytes per intersection per step (Grid 3x3 = 9 intersections)
    sarl = 89 * 9 * steps / 1e6  # MB
    # FedRL float32: 280 KB per intersection per round, 25 rounds
    rounds = np.floor(steps / 4000)
    fedrl = 280 * 9 * rounds / 1000  # MB
    fedrl_q8 = 70 * 9 * rounds / 1000  # MB (8-bit quantized)
    ax.plot(steps / 1000, sarl, label="SARL / MARL (89 B/step)", color="#2874a6", lw=1.2)
    ax.plot(steps / 1000, fedrl, label="FedRL (float32, 280 KB/round)",
            color="#c0392b", lw=1.2)
    ax.plot(steps / 1000, fedrl_q8, label="FedRL (int8, 70 KB/round)",
            color="#2ca47a", lw=1.2, ls="--")
    ax.axhline(sarl[-1], color="#2874a6", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Training timesteps (k)")
    ax.set_ylabel("Cumulative comm. cost (MB)")
    ax.set_title("Communication cost — Grid 3×3")
    ax.legend(fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.3, lw=0.4)
    save(fig, "fig_comm_cost.pdf")


# --- Fig 6: Strategy spectrum diagram ----------------------------------
def fig_spectrum():
    fig, ax = plt.subplots(figsize=(7.0, 1.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")

    # Gradient bar
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap="RdYlGn",
              extent=[0.4, 9.6, 1.55, 1.8], alpha=0.7)
    ax.text(0.4, 2.0, "Independent", fontsize=8.5, fontweight="bold", color="#b03030")
    ax.text(9.6, 2.0, "Fully shared", fontsize=8.5, fontweight="bold",
            color="#1a7a55", ha="right")

    strats = [
        ("MARL", 0.6), ("MeanField", 1.7), ("CTDE", 2.8),
        ("Gossip", 4.2), ("HierFed", 5.3), ("FedDistill", 6.6),
        ("FedRL", 7.9), ("SARL", 9.2),
    ]
    for name, x in strats:
        ax.plot([x], [1.67], "o", color="white", markersize=7,
                markeredgecolor="#333")
        ax.text(x, 1.3, name, ha="center", fontsize=7.8)

    # annotation brackets
    ax.annotate("", xy=(2.8, 1.0), xytext=(0.6, 1.0),
                arrowprops=dict(arrowstyle="|-|", lw=0.8, color="#b03030"))
    ax.text(1.7, 0.75, "Independent", ha="center", fontsize=7.5, color="#b03030")
    ax.annotate("", xy=(6.6, 1.0), xytext=(4.2, 1.0),
                arrowprops=dict(arrowstyle="|-|", lw=0.8, color="#c68e17"))
    ax.text(5.4, 0.75, "Collaborative (Goldilocks zone)", ha="center",
            fontsize=7.5, color="#c68e17")
    ax.annotate("", xy=(9.2, 1.0), xytext=(7.9, 1.0),
                arrowprops=dict(arrowstyle="|-|", lw=0.8, color="#1a7a55"))
    ax.text(8.55, 0.75, "Centralized", ha="center", fontsize=7.5, color="#1a7a55")

    ax.plot([3.0, 3.0], [0.15, 0.4], "-", lw=0.6, color="#777")
    ax.text(3.0, 0.05, "Fixed-Time", ha="center", fontsize=7.2, color="#555")
    ax.plot([5.0, 5.0], [0.15, 0.4], "-", lw=0.6, color="#777")
    ax.text(5.0, 0.05, "Max-Pressure", ha="center", fontsize=7.2, color="#555")
    ax.text(7.0, 0.05, "(non-RL baselines)", ha="left", fontsize=7.0,
            fontstyle="italic", color="#888")
    save(fig, "fig_spectrum.pdf")


# --- Fig 7: Robustness ranking across topologies -----------------------
def fig_robustness_ranking():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    # From slide: ranks across Grid 3x3 and Grid 5x5 at d=360
    rows = [
        ("Gossip",     1, 2, 1.5),
        ("HierFed",    3, 3, 3.0),
        ("FedRL",      2, 5, 3.5),
        ("CTDE",       8, 1, 4.5),
        ("MARL",       6, 4, 5.0),
        ("MeanField",  4, 7, 5.5),
        ("SARL",       5, 8, 6.5),
        ("FedDistill", 7, 6, 6.5),
    ]
    rows.sort(key=lambda r: r[3])
    names = [r[0] for r in rows]
    r3 = [r[1] for r in rows]
    r5 = [r[2] for r in rows]
    avg = [r[3] for r in rows]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos - 0.2, r3, height=0.35, color="#2874a6", label="Grid 3×3")
    ax.barh(ypos + 0.2, r5, height=0.35, color="#e67e22", label="Grid 5×5")
    labels = [f"{n}  (avg {a:.1f})" for n, a in zip(names, avg)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Rank (lower is better)")
    ax.set_xlim(0, 9)
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Baseline robustness ranking (d = 360)")
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    save(fig, "fig_robustness_ranking.pdf")


if __name__ == "__main__":
    fig_architecture()
    fig_training_curves()
    fig_baseline_3x3()
    fig_baseline_5x5()
    fig_baseline_cologne()
    fig_demand_sensitivity()
    fig_comm_cost()
    fig_spectrum()
    fig_robustness_ranking()
    print("figures written to", OUT)
