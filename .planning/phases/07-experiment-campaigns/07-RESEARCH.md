# Phase 7: Experiment Campaigns - Research

**Researched:** 2026-03-23
**Domain:** RL experiment automation, statistical analysis, FedRL traffic signal control
**Confidence:** HIGH

---

## Summary

Phase 7 is the capstone research phase — it runs actual training and evaluation campaigns to produce publishable results. The key insight from reading the codebase is that most of the infrastructure is already in place: the Monte Carlo orchestrator (`run_full_campaign`), the metric aggregation pipeline, and the JSON result store all exist from Phase 3. Phase 4 added FedProx, cooperative reward, and time-of-day extensions.

What Phase 7 needs is a **campaign automation layer** that sits above the existing infrastructure: Python scripts (not just REST calls) that run sequences of training + evaluation campaigns with specific configs, collect results into a canonical results directory, and generate comparison tables and charts. The paper's experiment structure is documented in the SUMO-FedRL-main notebooks — it compares Federated / Centralized / Decentralized trainers across grid-3x3/5x5/7x7 with 10 Monte Carlo seeds, using waiting time, travel time, and reward as primary metrics.

A critical finding: the example_weights directory has paper-original weights for FedRL and SARL on grid-3x3 and grid-5x5 (no grid-7x7 weights exist). The weight filenames use a `v3_naive-aggr_ranked.pkl` naming convention that does NOT match the current runner's expected `ranked.pkl` naming. This means **the example weights cannot be loaded directly by the current evaluation runner** without a path override. All paper-reproduction experiments must either retrain from scratch or explicitly pass the example weight path into `run_trial()`.

**Primary recommendation:** Build a standalone experiment runner script (`BackEnd/scripts/run_campaign.py`) that wraps the existing evaluation API, adds training-then-evaluating automation for extension configs, and exports results to CSV/JSON for Matplotlib/pandas chart generation. Keep the planner-focused on 3 plans: (1) campaign runner script, (2) extension comparison campaigns, (3) chart and table generation.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAMP-01 | Reproduce baseline paper results: FedRL vs MARL vs SARL vs fixed-time on grid-3x3, grid-5x5 | Example weights exist; resolve naming mismatch; use existing run_full_campaign() |
| CAMP-02 | Extension ablation: FedProx (mu in {0.0, 0.01, 0.1}) vs FedAvg baseline | Needs fresh training runs per mu value; runner already supports fedprox_mu param |
| CAMP-03 | Extension ablation: cooperative reward (alpha in {1.0, 0.5, 0.0}) vs selfish baseline | Training + evaluation; runner already supports alpha param |
| CAMP-04 | Extension ablation: time-of-day curriculum vs fixed demand | Training + evaluation; runner supports time_of_day + use_time_encoding params |
| CAMP-05 | Statistical rigor: 10 MC seeds per config, 95% CI, Wilcoxon significance test | Already implemented in monte_carlo.py for eval; need seed-controlled training too |
| CAMP-06 | Publishable output: comparison tables (LaTeX-compatible), bar charts with error bars | Matplotlib + pandas; paper notebooks use seaborn barplot with CI |
| CAMP-07 | Results persistence and reproducibility: all configs and seeds logged alongside results | Extend existing JSON store format; add config snapshot to each experiment record |
</phase_requirements>

---

## What Already Exists (Critical Context)

The planner MUST know these before designing tasks — they determine what to build vs reuse:

### Evaluation Infrastructure (Phase 3 — Complete)
- `BackEnd/api/evaluation/runner.py` — `run_trial(trainer, topology, seed, ranked, weights_path, horizon)` runs one episode
- `BackEnd/api/evaluation/monte_carlo.py` — `run_full_campaign(trainers, topologies, n_runs)` runs 5×3×10
- `BackEnd/api/evaluation/metrics.py` — `compute_trial_metrics()` extracts tripinfo + reward + comm cost
- `BackEnd/api/evaluation/transfer.py` — `build_transfer_matrix()` for cross-topology transfer
- `BackEnd/api/evaluation/store.py` — `save_evaluation()` / `load_evaluation()` for JSON persistence
- Results land in `BackEnd/results/evaluations/eval_*.json` — working (4 eval JSONs exist)

### Extension Params (Phase 4 — Complete)
- `fedprox_mu: float` — 0.0 = pure FedAvg; > 0 activates FedProx proximal term
- `alpha: float` — 1.0 = fully selfish; < 1.0 = cooperative reward sharing
- `time_of_day: bool` — AM/midday/PM rush demand curriculum
- `use_time_encoding: bool` — sine/cosine time features appended to observations
- All params flow: TrainRequest → create_trainer() → env_config_fn() → SumoEnv

### Training Infrastructure
- `BackEnd/api/training_runner.py` — `create_trainer()` + `run_training_loop()` with per-episode streaming
- Trainers: FedPolicyTrainer (FedRL), MultiPolicyTrainer (MARL), SinglePolicyTrainer (SARL)
- Weights saved to `trained_weights/weights/{TrainerType}/{topology}/ranked.pkl`

### Existing Weights
| Trainer | Topology | Source | Naming Issue |
|---------|----------|--------|--------------|
| FedRL | grid-3x3 | example_weights/ICCPS/Final/ | `v3_naive-aggr_ranked.pkl` — NOT `ranked.pkl` |
| FedRL | grid-5x5 | example_weights/ICCPS/Final/ | Same naming mismatch |
| MARL | grid-3x3, grid-5x5 | example_weights/ICCPS/Final/ | `v3_ranked.pkl` — NOT `ranked.pkl` |
| SARL | grid-3x3, grid-5x5 | example_weights/ICCPS/Final/ | `v3_ranked.pkl` — NOT `ranked.pkl` |
| FedRL | grid-3x3 | trained_weights/ | `ranked.pkl` — matches runner expectation |
| SARL | grid-3x3 | trained_weights/ | `ranked.pkl` — matches runner expectation |
| ANY | grid-7x7 | MISSING | Must train from scratch |

---

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib statistics | 3.x | Mean, std, CI in monte_carlo.py | Already used, no new dep |
| matplotlib | 3.x | Charts with error bars | Paper notebooks use it; project already has it |
| pandas | 2.x | CSV export, DataFrame manipulation | Paper notebooks use it; already a dep (used in BaseTrainer) |
| seaborn | 0.13.x | Bar charts, ECDF plots | Paper notebooks use seaborn; installed in SUMO-FedRL notebooks |
| scipy.stats | 1.x | Wilcoxon signed-rank test | Needed for statistical significance; widely available |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.x / 2.x | Array ops for chart data | Already used in fed_agent.py |
| json (stdlib) | - | Result serialization | Already used everywhere |
| argparse (stdlib) | - | CLI for campaign runner | Clean entry point for scripts |

**Installation (if scipy not present):**
```bash
pip install scipy seaborn
```

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy Wilcoxon | Mann-Whitney U / t-test | Wilcoxon is correct for paired non-parametric RL data |
| seaborn barplot | Matplotlib manually | Seaborn is faster, already in paper notebooks |
| standalone script | REST API only | Script bypasses server overhead, runs campaigns faster |

---

## Architecture Patterns

### Recommended Project Structure
```
BackEnd/
├── scripts/
│   ├── run_campaign.py          # Campaign automation script (new)
│   ├── run_extension_ablation.py # Extension comparison script (new)
│   └── generate_tables.py       # LaTeX table + chart generation (new)
├── api/evaluation/
│   ├── runner.py               # Existing — run_trial()
│   ├── monte_carlo.py          # Existing — run_full_campaign()
│   ├── metrics.py              # Existing — compute_trial_metrics()
│   └── campaign_config.py      # New — typed config for experiment campaigns
├── results/
│   ├── evaluations/            # Existing — per-job JSON
│   ├── campaigns/              # New — named campaign results (paper-baseline, fedprox-ablation, etc.)
│   └── figures/                # New — generated charts
```

### Pattern 1: Training-then-Evaluating Pipeline

For extension ablations, you must train with a specific config first, then evaluate the resulting weights. This is a two-phase sequence the existing API does not automate.

**What:** Encapsulate (train → save weights → evaluate → collect metrics) as a reusable function.
**When to use:** All extension ablation campaigns (FedProx, cooperative reward, time-of-day).
**Example:**
```python
# BackEnd/scripts/run_campaign.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.training_runner import create_trainer, run_training_loop
from api.evaluation.monte_carlo import run_monte_carlo, MCConfig

def train_and_evaluate(
    trainer_type: str,
    topology: str,
    n_episodes: int,
    n_eval_runs: int,
    seed: int = 42,
    fedprox_mu: float = 0.0,
    alpha: float = 1.0,
    time_of_day: bool = False,
    use_time_encoding: bool = False,
) -> dict:
    """Train with config, then evaluate with 10 MC seeds. Returns aggregated metrics."""
    # Step 1: Train
    trainer = create_trainer(
        trainer_type=trainer_type,
        topology=topology,
        ranked=True,
        n_episodes=n_episodes,
        fedprox_mu=fedprox_mu,
        alpha=alpha,
        time_of_day=time_of_day,
        use_time_encoding=use_time_encoding,
    )
    result = run_training_loop(trainer, n_episodes=n_episodes)
    weights_path = result.get("weights_path")  # ranked.pkl path

    # Step 2: Evaluate
    config = MCConfig(
        trainer=trainer_type,
        topology=topology,
        n_runs=n_eval_runs,
        base_seed=seed,
        ranked=True,
        weights_path=weights_path,
    )
    mc_result = run_monte_carlo(config)
    return mc_result
```

### Pattern 2: Example Weights Path Override

The example weights have non-matching filenames (`v3_naive-aggr_ranked.pkl` vs `ranked.pkl`). Use `weights_path` override in `run_trial()` / `MCConfig` to load them explicitly.

```python
# Load paper-original FedRL weights for grid-3x3
EXAMPLE_WEIGHTS = {
    ("FedRL", "grid-3x3"): "BackEnd/example_weights/ICCPS/Final/FedRL/grid-3x3/v3_naive-aggr_ranked.pkl",
    ("FedRL", "grid-5x5"): "BackEnd/example_weights/ICCPS/Final/FedRL/grid-5x5/v3_naive-aggr_ranked.pkl",
    ("MARL", "grid-3x3"): "BackEnd/example_weights/ICCPS/Final/MARL/grid-3x3/v3_ranked.pkl",
    ("SARL", "grid-3x3"): "BackEnd/example_weights/ICCPS/Final/SARL/grid-3x3/v3_ranked.pkl",
    # grid-7x7: no example weights — must train
}

config = MCConfig(
    trainer="FedRL",
    topology="grid-3x3",
    n_runs=10,
    weights_path=EXAMPLE_WEIGHTS[("FedRL", "grid-3x3")],  # explicit override
)
```

### Pattern 3: Statistical Significance with Wilcoxon

For comparing two conditions (e.g., FedProx mu=0.1 vs mu=0.0), the Wilcoxon signed-rank test is appropriate for non-parametric paired comparison of MC runs.

```python
from scipy import stats

def wilcoxon_test(values_a: list, values_b: list) -> dict:
    """Compare two sets of MC waiting times. Returns statistic, p-value, significant."""
    stat, p = stats.wilcoxon(values_a, values_b)
    return {"statistic": stat, "p_value": p, "significant": p < 0.05}

# Usage: compare FedAvg vs FedProx on avg_waiting_time across 10 seeds
fedavg_waits = [r["tripinfo"]["avg_waiting_time"] for r in fedavg_result.individual_results]
fedprox_waits = [r["tripinfo"]["avg_waiting_time"] for r in fedprox_result.individual_results]
test = wilcoxon_test(fedavg_waits, fedprox_waits)
```

### Pattern 4: Comparison Table Generation

The paper uses a LaTeX table (visible in the notebook's `make_table()` function). The table structure:
- Rows: topology pairs (trained on / tested on)
- Columns: trainer × metric (travel time, waiting time)
- Values: mean ± std

```python
def results_to_dataframe(campaign_results: list) -> pd.DataFrame:
    """Flatten campaign JSON into a pandas DataFrame for table/chart generation."""
    rows = []
    for entry in campaign_results:
        agg = entry["aggregated"]
        rows.append({
            "trainer": entry["trainer"],
            "topology": entry["topology"],
            "avg_waiting_time_mean": agg["avg_waiting_time"]["mean"],
            "avg_waiting_time_std": agg["avg_waiting_time"]["std"],
            "avg_travel_time_mean": agg["avg_travel_time"]["mean"],
            "mean_reward_mean": agg["mean_reward"]["mean"],
            "total_comm_cost_mean": agg["total_comm_cost"]["mean"],
        })
    return pd.DataFrame(rows)

def export_latex_table(df: pd.DataFrame, output_path: str) -> None:
    """Write a LaTeX booktabs table from the results DataFrame."""
    # Use df.to_latex() with formatters, or hand-build for full control
    df.style.format("{:.1f}").to_latex(output_path, hrules=True)
```

### Pattern 5: Error Bars Chart

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_bar(df: pd.DataFrame, metric: str, output_path: str) -> None:
    """Grouped bar chart with 95% CI error bars, one group per topology."""
    trainers = df["trainer"].unique()
    topologies = df["topology"].unique()
    x = np.arange(len(topologies))
    width = 0.8 / len(trainers)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, trainer in enumerate(trainers):
        means = [df[(df.trainer==trainer) & (df.topology==t)][f"{metric}_mean"].values[0]
                 for t in topologies]
        stds = [df[(df.trainer==trainer) & (df.topology==t)][f"{metric}_std"].values[0]
                for t in topologies]
        # 95% CI from stored std (n=10): 1.96 * std / sqrt(10)
        ci = [1.96 * s / np.sqrt(10) for s in stds]
        ax.bar(x + i*width, means, width, yerr=ci, label=trainer, capsize=4)

    ax.set_xticks(x + width * (len(trainers)-1)/2)
    ax.set_xticklabels(topologies)
    ax.set_ylabel(metric.replace("_", " ").title() + " (s)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
```

### Anti-Patterns to Avoid

- **Don't call REST endpoints from campaign scripts:** The scripts should import Python modules directly (`from api.evaluation.monte_carlo import run_full_campaign`), not use `requests.post()`. REST adds latency and server-restart risk for multi-hour campaigns.
- **Don't use pandas `.to_latex()` blindly:** It doesn't add booktabs formatting. Use `buf.write()` with manual `\toprule / \midrule / \bottomrule` or the `styler.to_latex(hrules=True)` argument (pandas ≥ 1.3).
- **Don't share a single Ray instance across training and evaluation in the same process:** Training builds a full Algorithm with workers; evaluation uses PPOTorchPolicy standalone. The current singleton pattern handles this correctly — do not add `ray.shutdown()` calls between runs.
- **Don't assume grid-7x7 evaluations are fast:** grid-7x7 episodes with 450-step horizon take significantly longer than grid-3x3. Plan for campaign runtimes of hours, not minutes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Monte Carlo aggregation | Custom loop with dicts | `run_full_campaign()` in monte_carlo.py | Already tested; 95% CI, error handling, progress callbacks |
| Statistical significance | Manual rank sums | `scipy.stats.wilcoxon` | Handles tied ranks, exact/approx modes, edge cases |
| Result serialization | Custom JSON encoder | `json.dump(default=str)` + existing `save_evaluation()` | Already handles datetime, numpy types |
| Metric extraction from tripinfo | XML parser from scratch | `compute_tripinfo_metrics()` in metrics.py | Already handles missing files, malformed XML |
| Weight loading | Re-implement pickle load | `resolve_weights_path()` + `load_weights()` | Already handles fallback paths |

**Key insight:** Phase 7's value is in **running** the experiments and **interpreting** results, not in rebuilding infrastructure. The planner should allocate implementation tasks only for the campaign automation script, config management, and output generation.

---

## Common Pitfalls

### Pitfall 1: Example Weights Filename Mismatch
**What goes wrong:** `resolve_weights_path("FedRL", "grid-3x3", ranked=True)` looks for `ranked.pkl` but the example weights are `v3_naive-aggr_ranked.pkl`. The runner raises `FileNotFoundError` even though example weights exist.
**Why it happens:** The paper trained with multiple aggregation strategies; filenames distinguish them. Our runner expects a single canonical `ranked.pkl`.
**How to avoid:** Build an explicit `EXAMPLE_WEIGHTS_MAP` dict in the campaign script and always pass `weights_path=` override for paper-reproduction runs. Never rely on `resolve_weights_path()` for example weights.
**Warning signs:** `FileNotFoundError: No weights found for trainer='FedRL', topology='grid-3x3'` during baseline campaigns.

### Pitfall 2: grid-7x7 Missing Weights
**What goes wrong:** No example weights exist for grid-7x7 (confirmed by directory listing). Paper-reproduction campaigns on grid-7x7 will fail immediately.
**Why it happens:** The example_weights directory only has grid-3x3 and grid-5x5 subtopologies.
**How to avoid:** Either (a) train fresh weights on grid-7x7 as a prerequisite task (resource-intensive), or (b) limit paper-reproduction to grid-3x3 and grid-5x5, and only run extension ablations on grid-3x3.
**Warning signs:** `FileNotFoundError` for grid-7x7 in any RL trainer campaign.

### Pitfall 3: Sequential Training Blocking Campaign
**What goes wrong:** Training 50 episodes on grid-3x3 may take 10-30 minutes per run. Running 4 extension configs × 3 topologies = 12 training runs sequentially could take 6-12 hours.
**Why it happens:** SUMO runs in real-time simulation with Ray's single-worker config (num_workers=0).
**How to avoid:** (a) Design the campaign script to accept a `--skip-training` flag that uses existing trained_weights. (b) Run only grid-3x3 for ablation studies, not all topologies. (c) Use smaller n_episodes (e.g., 30) for ablation training since we care about relative differences, not absolute convergence.
**Warning signs:** Script appears frozen — check SUMO process CPU usage.

### Pitfall 4: Throughput Metric Incompleteness
**What goes wrong:** `compute_tripinfo_metrics()` currently sets `throughput = 1.0` unconditionally (hardcoded, see line 128 of metrics.py) because it doesn't have access to the total spawned vehicle count. All throughput values in results will be 1.0.
**Why it happens:** The tripinfo XML only records *completed* trips; spawned count comes from the route file.
**How to avoid:** For throughput metric: either fix `compute_tripinfo_metrics()` to parse the route file alongside tripinfo (note route file path), or report throughput as `completed_trips` (raw count) rather than ratio. This must be flagged in any paper that reports throughput.
**Warning signs:** All runs show `"throughput": {"mean": 1.0, "std": 0.0}` in results.

### Pitfall 5: Wilcoxon Test with n < 10
**What goes wrong:** If some MC runs fail (network errors, SUMO crash), n_completed may be < 10. Wilcoxon with n < 5 is unreliable; scipy will raise a warning.
**Why it happens:** MC runs can fail silently — the orchestrator logs errors but continues.
**How to avoid:** Check `n_completed` before running significance tests. Require `n_completed >= 8` to report results as statistically valid. The campaign script should log a warning when below threshold.
**Warning signs:** `scipy.stats.wilcoxon` raises `UserWarning: Sample size too small for normal approximation`.

### Pitfall 6: Cooperative Reward with alpha=0 Creates Degenerate Agents
**What goes wrong:** Setting `alpha=0.0` means agents get zero selfish reward and only neighbor penalty. In early training, before policies learn anything useful, neighbor_mean may always be 0, collapsing the reward signal entirely.
**Why it happens:** The cooperative reward formula is `r = alpha * own_reward + (1 - alpha) * neighbor_mean`. At alpha=0 with untrained neighbors, all rewards are 0.
**How to avoid:** Use alpha values of {1.0, 0.5, 0.1} rather than going all the way to 0.0. Verify training curves show learning progression before running MC evaluation.
**Warning signs:** Episode rewards stay flat at 0.0 throughout training.

---

## Code Examples

### Campaign Config Dataclass
```python
# BackEnd/api/evaluation/campaign_config.py
from dataclasses import dataclass, field, asdict
from typing import Optional, List

@dataclass
class ExtensionConfig:
    """Config for a single training + evaluation experiment."""
    name: str                          # Human-readable label, e.g. "fedprox_mu0.1"
    trainer_type: str                  # "FedRL", "MARL", "SARL"
    topology: str                      # "grid-3x3", etc.
    n_episodes: int = 50               # Training episodes
    n_eval_runs: int = 10              # MC evaluation runs
    base_seed: int = 42
    fedprox_mu: float = 0.0
    alpha: float = 1.0
    time_of_day: bool = False
    use_time_encoding: bool = False
    weights_path: Optional[str] = None  # If set, skip training
```

### Comparison Table Export (LaTeX-ready)
```python
def to_latex_table(results_df: pd.DataFrame, metric: str) -> str:
    """Generate booktabs-formatted LaTeX table."""
    pivot = results_df.pivot_table(
        index="topology", columns="trainer",
        values=[f"{metric}_mean", f"{metric}_std"]
    )
    # Format as "mean ± std"
    lines = ["\\begin{tabular}{l" + "c" * len(pivot.columns.get_level_values(1).unique()) + "}"]
    lines.append("\\toprule")
    # ... header row ...
    lines.append("\\midrule")
    for topo, row in pivot.iterrows():
        # ... data rows ...
        pass
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
```

### Minimal Paper-Reproduction Campaign
```python
# Run only the paper baseline (FedRL/MARL/SARL/fixed-time on grid-3x3 and grid-5x5)
# using the existing example weights — fastest path to publishable comparison
from api.evaluation.monte_carlo import run_full_campaign

results = run_full_campaign(
    trainers=["FedRL", "MARL", "SARL", "fixed-time", "max-pressure"],
    topologies=["grid-3x3", "grid-5x5"],
    n_runs=10,
    ranked=True,
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Jupyter notebooks (SUMO-FedRL-main) | Standalone Python scripts in BackEnd/scripts/ | Phase 7 | Reproducible, version-controlled, scriptable |
| REST API for evaluation | Direct Python import from scripts | Phase 7 | No server dependency for long-running campaigns |
| Fixed seed evaluation | 10 MC seeds with 95% CI | Phase 3 | Publishable statistical rigor |
| No significance test | Wilcoxon signed-rank test | Phase 7 | Claims about extensions are verifiable |

---

## Experiment Campaign Design

### Campaign 1: Baseline Paper Reproduction
**Goal:** Confirm our implementation matches the paper's results (ICCPS 2022).
**Configs:** FedRL(naive-aggr), MARL, SARL, fixed-time, max-pressure × grid-3x3, grid-5x5
**Weights:** Example weights (with path override mapping)
**Metrics:** avg_waiting_time, avg_travel_time, mean_reward, total_comm_cost
**Expected runtime:** ~2-4 hours (no training, evaluation only)
**Note:** grid-7x7 skipped — no example weights.

### Campaign 2: FedProx Ablation
**Goal:** Measure effect of proximal term mu on convergence and evaluation performance.
**Configs:** FedRL × {mu=0.0, mu=0.01, mu=0.1} × grid-3x3
**Required:** 3 separate training runs + 3 × 10 MC evaluations
**Metrics:** Convergence speed (episodes to plateau), avg_waiting_time, mean_reward
**Expected runtime:** ~3-6 hours (3 training runs × ~1h each + evaluation)

### Campaign 3: Cooperative Reward Ablation
**Goal:** Measure effect of alpha on traffic coordination.
**Configs:** FedRL × {alpha=1.0, alpha=0.5, alpha=0.1} × grid-3x3
**Required:** 3 training runs + 3 × 10 MC evaluations
**Metrics:** avg_waiting_time (primary — cooperation should reduce it), mean_reward
**Expected runtime:** ~3-6 hours

### Campaign 4: Time-of-Day Ablation
**Goal:** Measure adaptation to demand variation.
**Configs:** FedRL × {time_of_day=False, time_of_day=True+time_encoding} × grid-3x3
**Required:** 2 training runs + 2 × 10 MC evaluations
**Metrics:** avg_waiting_time by time-of-day period, mean_reward
**Expected runtime:** ~2-4 hours

### Minimum Viable Campaign (if time is limited)
Campaign 1 only (paper baseline evaluation using example weights) can be done in 2-4 hours with no new training required. It produces a directly publishable comparison table. Extensions can be added incrementally.

---

## Seeds and Statistical Rigor

| Parameter | Recommendation | Justification |
|-----------|----------------|---------------|
| MC seeds per config | 10 | Matches SEAL paper (seeds 42..51); already default in MCConfig |
| Confidence interval | 95% (1.96 σ / √n) | Already implemented in _aggregate_metrics() |
| Significance test | Wilcoxon signed-rank (paired) | Non-parametric, correct for comparing two conditions with same 10 seeds |
| Significance threshold | p < 0.05 | Standard in RL/systems literature |
| Minimum valid n | 8 of 10 completed | Flag results with n_completed < 8 as potentially unreliable |
| Training seed | RAY_TRAINER_SEED = 54321 | Fixed in BaseTrainer — provides deterministic weight initialization |

The existing CI formula (`1.96 * std / sqrt(n)`) is the normal approximation. With n=10, this is a rough estimate — a t-distribution would be more accurate (t_9,0.025 = 2.262 vs 1.96). For publication, prefer reporting 95% CI via `scipy.stats.t.interval(0.95, df=9, loc=mean, scale=sem)` instead of the approximation. This is a LOW-effort improvement worth noting in the plan.

---

## Open Questions

1. **Training run duration on this machine**
   - What we know: Grid-3x3 with 50 episodes is the standard. State-of-the-art results: A single grid-3x3 training run on the original SUMO-FedRL hardware took ~1-2 hours based on paper context.
   - What's unclear: Current machine (Windows 11, no GPU) training time — could be 1-4 hours per run.
   - Recommendation: Add a `--dry-run 2` flag to scripts that runs only 2 episodes for smoke-testing before committing to full runs.

2. **Throughput metric fix**
   - What we know: `metrics.py` hardcodes `throughput = 1.0` because it lacks the spawned vehicle count.
   - What's unclear: How to get spawned count — it's available from the route file, but the route file path isn't passed to `compute_tripinfo_metrics()`.
   - Recommendation: For Phase 7, report `completed_trips` as the raw throughput proxy. Fix the formula as a separate task or note the limitation in paper text. Do NOT block the campaign on this.

3. **Whether trained_weights/weights/FedRL/grid-3x3/ranked.pkl is compatible with current env**
   - What we know: It exists and was trained during Phase 2. The policy architecture (PPOTorchPolicy, obs space 14-feature) should match.
   - What's unclear: Whether Phase 4 obs space changes (use_time_encoding adds 2 features) break compatibility when loading these weights.
   - Recommendation: Evaluate with `use_time_encoding=False` (default) when using existing weights. Document that extension-trained weights require extension-eval configs.

---

## Validation Architecture

The config.json does not set `nyquist_validation: false` — it only has `_auto_chain_active: false`. Treating validation as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (used in Phase 3 tests) |
| Config file | None detected — pytest auto-discovers |
| Quick run command | `pytest BackEnd/tests/ -x -q --timeout=30` |
| Full suite command | `pytest BackEnd/tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CAMP-01 | Baseline campaign runs without error on example weights | integration | `pytest BackEnd/tests/test_campaign_runner.py -x` | No — Wave 0 |
| CAMP-02 | FedProx ablation produces lower waiting time than FedAvg | smoke | Manual / script execution | No |
| CAMP-03 | Cooperative reward ablation produces valid results | smoke | Manual / script execution | No |
| CAMP-05 | Wilcoxon test returns expected structure | unit | `pytest BackEnd/tests/test_stats.py -x` | No — Wave 0 |
| CAMP-06 | Chart generation produces valid PNG/PDF files | unit | `pytest BackEnd/tests/test_charts.py -x` | No — Wave 0 |
| CAMP-07 | Campaign config saved alongside results in JSON | unit | `pytest BackEnd/tests/test_campaign_store.py -x` | No — Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest BackEnd/tests/ -x -q --timeout=30`
- **Per wave merge:** Full suite
- **Phase gate:** Full suite green before marking phase complete

### Wave 0 Gaps
- [ ] `BackEnd/tests/test_campaign_runner.py` — smoke test that campaign script runs end-to-end on 1 seed
- [ ] `BackEnd/tests/test_stats.py` — unit test for wilcoxon_test() helper
- [ ] `BackEnd/tests/test_charts.py` — unit test that chart functions produce non-empty output files
- [ ] `BackEnd/api/evaluation/campaign_config.py` — new module, no file yet

---

## Sources

### Primary (HIGH confidence)
- `BackEnd/api/evaluation/runner.py` — confirmed: `run_trial()` signature, weights path resolution logic, TrialResult structure
- `BackEnd/api/evaluation/monte_carlo.py` — confirmed: `run_full_campaign()` default configs (5 trainers × 3 topologies × 10 runs), seed scheme (base_seed + i)
- `BackEnd/api/evaluation/metrics.py` — confirmed: throughput hardcoded to 1.0, TripinfoMetrics fields
- `BackEnd/example_weights/` directory listing — confirmed: no grid-7x7 weights, naming mismatch (`v3_naive-aggr_ranked.pkl`)
- `BackEnd/trained_weights/weights/` directory listing — confirmed: only FedRL and SARL on grid-3x3 exist
- `SUMO-FedRL-main/notebooks/Experiments Analysis - v2.ipynb` — confirmed: paper uses 10 MC runs, Federated/Centralized/Decentralized labels, seaborn barplot, LaTeX table structure with travel_time + waiting_time metrics
- `BackEnd/results/evaluations/eval_ebf0b19c.json` — confirmed: fixed-time on grid-3x3 shows avg_waiting_time=92.7s, framework working

### Secondary (MEDIUM confidence)
- `BackEnd/seal/trainer/fed_agent.py` — fedprox_mu flow: confirmed param passthrough to FedProxPPOTorchPolicy
- `BackEnd/seal/trainer/base.py` — RAY_TRAINER_SEED=54321 confirmed as fixed training seed

### Tertiary (LOW confidence)
- Training runtime estimates (1-4 hours per run) — extrapolated from paper context and hardware; not measured on this machine

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — direct code inspection of all evaluation modules
- Architecture: HIGH — paper notebook analysis + full codebase read
- Pitfalls: HIGH — found real bugs (throughput hardcoded, weight naming mismatch, missing grid-7x7 weights) from code inspection
- Statistical methods: MEDIUM — Wilcoxon recommendation from RL literature best practices, not verified against published SEAL paper methods section

**Research date:** 2026-03-23
**Valid until:** 2026-06-23 (stable — evaluation infrastructure unlikely to change)
