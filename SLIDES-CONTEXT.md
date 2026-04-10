# Slides Context Document — Federated RL Traffic Signal Control Benchmarking

> **Purpose:** Complete context for creating presentation slides. Compiled from all project documentation, archive files, planning artifacts, and experimental results.
> **Last updated:** 2026-04-10

---

## 1. PROJECT IDENTITY

### Title Options
- **Working title:** "SEAL Dashboard — Federated RL Traffic Signal Control"
- **Paper-ready title:** "When Does Federation Help? A Controlled Comparison of Training Strategies for Multi-Intersection Traffic Signal Control"
- **One-liner:** A benchmarking framework that evaluates RL training strategies for traffic signal control under standardized conditions

### Core Framing
We are NOT proposing a new algorithm. We built the first standardized evaluation framework that compares training strategies for RL-based traffic signal control under identical conditions. The contribution is the framework, the fair comparison, and the findings it produces.

### Core Value
Real FedRL experiments working end-to-end with publishable results — training, evaluation, and cross-topology transfer on SUMO networks.

### The Analogy
"Nobody asks a civil engineer 'but you didn't invent steel.' The contribution is the bridge."

---

## 2. THE PROBLEM

### Why This Matters
- Traffic congestion costs the US economy **$87 billion annually** (INRIX 2019)
- Most traffic lights run on fixed timers: green for N seconds, red for M seconds, regardless of conditions
- RL can train traffic lights to adapt to real-time demand
- Multiple ways to organize training across a network of intersections, each with different trade-offs
- Existing literature (fewer than 10 papers on FedRL for traffic signals worldwide) compares strategies using **different setups, different simulators, different metrics** — no fair comparison exists
- **Our gap:** No standardized head-to-head comparison of training strategies under identical conditions

---

## 3. THE 10 TRAINING STRATEGIES

### Strategy Spectrum

```
Independent <---------------------------------------------> Shared

MARL -> MeanField -> CTDE -> Gossip -> HierFed -> FedDistill -> FedRL -> SARL
 (0)     (obs)     (critic)  (mesh)    (tree)     (logits)     (star)  (full)
```

### Strategy Details

| # | Strategy | What It Does | Unique Mechanism |
|---|----------|-------------|------------------|
| 1 | **SARL** | One shared policy for all intersections | Maximum sharing, highest communication |
| 2 | **MARL** | Independent per-intersection policies, no sharing | Pure specialization, zero coordination |
| 3 | **FedRL** | Central server FedAvg (star topology) | Periodic reward-weighted weight averaging |
| 4 | **Gossip** | Peer-to-peer neighbor weight averaging | Mesh topology, no central server |
| 5 | **HierFed** | Two-tier cluster-then-global averaging | Tree topology, reward-based clustering |
| 6 | **FedDistill** | Share action logits not weights | KL distillation, lightweight communication |
| 7 | **MeanField** | Obs augmented with mean neighbor action | +1 obs dimension, implicit coordination |
| 8 | **CTDE** | Centralized critic, decentralized actors | Global state during training only |
| 9 | **Fixed-Time** | Fixed phase cycling at timed intervals | Non-RL floor baseline |
| 10 | **Max-Pressure** | Switch to highest queue pressure phase | Classical traffic engineering baseline |

### Original 3 Strategies (Midterm)

| Property | SARL | MARL | FedRL |
|----------|------|------|-------|
| Policies | 1 shared | 1 per intersection | 1 per intersection |
| Learning | Learns from ALL intersections | Each from own only | Each from own, periodic sharing |
| Communication | All obs to central model | All obs to central trainer | Only weights, periodically |
| Comm cost (3x3, 200k steps) | 174.6 MB | 174.6 MB | 102.6 MB |
| Privacy | No (all data centralized) | No (all data centralized) | Yes (raw data stays local) |
| Scalability | Degrades with network size | Moderate | Best |

### Federated Strategy Variants (4 of 10)

| Strategy | Topology | Shared Payload | Key Difference |
|----------|----------|---------------|----------------|
| FedRL | Star (central) | Full weights | Standard FedAvg |
| Gossip | Mesh (neighbors only) | Full weights | No central server, peer-to-peer |
| HierFed | Tree (cluster -> global) | Full weights | Two-tier aggregation hierarchy |
| FedDistill | Star (central) | Action logits only | KL divergence loss, not weight averaging |

---

## 4. TECHNICAL ARCHITECTURE

### System Architecture

```
LovableOutput/ (read-only)  ->  FrontEnd/  <-WebSocket/REST->  BackEnd/  <-TraCI->  SUMO
SUMO-FedRL-main/ (read-only) ->  BackEnd/
```

```
FrontEnd/          React + Vite + shadcn/ui + Tailwind + Canvas API
    |  REST + WebSocket
BackEnd/           FastAPI + Ray RLlib + PyTorch (SEAL framework)
    |  TraCI (Python API)
SUMO               Microscopic traffic simulator
```

### The 8-Layer Controlled Comparison

The framework guarantees fair comparison through 8 controlled layers. When we report performance differences, they can ONLY come from how policies are organized — everything else is locked.

| Layer | What's Controlled | How Enforced |
|-------|------------------|--------------|
| 1. Same simulator | SUMO physics | Same binary |
| 2. Same network | Road topology | Same .net.xml file |
| 3. Same demand | Traffic volume | Same VPLPH, same randomTrips, same seeds |
| 4. Same observations | What agents see | 14-feature intersection-agnostic vector |
| 5. Same reward | How agents are graded | r = -(o + h)^2 |
| 6. Same algorithm | How agents learn | PPO, identical hyperparameters |
| 7. Same training budget | How long agents train | Same episode count per config |
| 8. Same evaluation | How we measure | Monte Carlo with bootstrap CI, Wilcoxon tests |

**The ONLY thing that changes:** How policies are organized across intersections.

---

## 5. OBSERVATION SPACE (14-16 Features)

All normalized to [0, 1]. Intersection-agnostic — works for any intersection type.

### Traffic Flow Features (indices 0-2)

| Index | Feature | What It Captures |
|-------|---------|-----------------|
| 0 | Lane Occupancy | vehicle length / lane length — how full the road is |
| 1 | Halted Lane Occupancy | halted vehicle length / lane length (speed < 0.1 m/s) |
| 2 | Speed Ratio | avg vehicle speed / speed limit — traffic flow quality |

### Phase State Ratios (indices 3-9)

| Index | Feature | What It Captures |
|-------|---------|-----------------|
| 3-9 | Phase State r/y/g/G/u/o/O | Fraction of lights in each state |

This is the intersection-agnostic encoding: instead of "GGrr" (topology-specific), we encode "50% green, 50% red" (works for any intersection type).

### Network Ranking Features (indices 10-13)

| Index | Feature | What It Captures |
|-------|---------|-----------------|
| 10 | Local Rank | Congestion relative to immediate neighbors |
| 11 | Global Rank | Congestion relative to entire network |
| 12 | Local Halt Rank | Halted vehicle rank among neighbors |
| 13 | Global Halt Rank | Halted vehicle rank across network |

These enable cross-topology transfer: "I am the most congested in my neighborhood" means the same on a 3x3 grid and a 5x5 grid.

### Optional Time Encoding (indices 14-15)

| Index | Feature | Range |
|-------|---------|-------|
| 14 | sin(2pi * t/H) | [-1, 1] |
| 15 | cos(2pi * t/H) | [-1, 1] |

---

## 6. REWARD FUNCTION

Per intersection k at timestep t:

```
r_k = -(o_k + h_k)^2
```

- `o_k` = lane occupancy (fraction)
- `h_k` = halted lane occupancy (fraction)
- Quadratic: small queue = small penalty, large queue = massive penalty
- Total network reward: r = sum of r_k across all intersections

### Why Quadratic?
Linear treats 10%->20% occupancy the same as 80%->90%. Quadratic makes high congestion disproportionately expensive: -(0.2+0.1)^2 = -0.09 vs -(0.8+0.7)^2 = -2.25.

---

## 7. ACTION SPACE

- **Binary:** Discrete(2) — keep current phase (0) or advance to next phase (1)
- **Timing constraints:** min 4 seconds between changes, max 120 seconds before forced change
- Binary is intersection-agnostic — same action space for any intersection type, essential for federated weight sharing

---

## 8. FEDERATED AGGREGATION

### Standard FedAvg (McMahan et al. 2017)
```
w_global = (1/K) * sum(w_k)
```

### Reward-Weighted FedAvg (Our Design)
```
w_global = sum(c_k * w_k)
```
where c_k proportional to intersection k's normalized reward. Better-performing intersections contribute more.

### Aggregation Weight Functions

| Function | Description |
|----------|-------------|
| naive | Equal weights 1/K (baseline FedAvg) |
| pos_reward | Weight by shift-normalized positive reward (default) |
| traffic | Weight by vehicles served |

### FedProx Extension (Li et al. 2020)
```
L = L_ppo + (mu/2) * ||w_k - w_global||^2
```
Penalizes local models from drifting too far from consensus.

---

## 9. EXPERIMENT DESIGN

### Current Design: 2-Tier Approach

#### Tier 1: Core Strategy Comparison (HPC batch)
- **10 strategies x 3 topologies x 3 demands x 3 training seeds = 270 runs**
- Topologies: grid-3x3, grid-5x5, cologne-8
- Demand levels: 150, 360, 600 VPLPH
- Training seeds: 42, 123, 456
- 50 episodes per config, 10 MC eval runs

#### Tier 2: Targeted Ablations (after Tier 1 results)
Designed based on findings. Candidates: aggregation rule, FedProx mu, cooperative alpha, time-of-day, gossip radius, CTDE critic scope.

### Topologies

| Network | Intersections | Controlled Lanes | Purpose |
|---------|--------------|------------------|---------|
| Grid 3x3 | 9 | 24 | Small-scale baseline |
| Grid 5x5 | 25 | Heterogeneous | Scalability test |
| Cologne-8 | 8 | Real-world | Real topology validation |

### Traffic Demand Settings

| Setting | VPLPH | Purpose |
|---------|-------|---------|
| Low | 150 | Does RL even matter when traffic is light? |
| Medium | 360 | Standard conditions — literature baseline |
| High | 600 | Stress test — how do strategies degrade? |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| RL Algorithm | PPO (Proximal Policy Optimization) |
| Learning rate | 0.001 |
| SGD minibatch size | 128 |
| PPO clip parameter | 0.3 |
| Train batch size | 4000 timesteps |
| Discount factor (gamma) | 0.95 |
| Network architecture | 2 hidden layers x 256 neurons, ReLU |

### Statistical Methods

- Bootstrap 95% CI (10,000 resamples, deterministic seed)
- Wilcoxon signed-rank test with rank-biserial effect size
- Bonferroni correction for multiple comparisons
- 3 training seeds per config to separate training variance from eval variance

---

## 10. MIDTERM RESULTS (3 Strategies, 360 VPLPH)

### Training Convergence (25 episodes)

| Strategy | Grid 3x3 Start -> End | Improvement | Grid 5x5 Start -> End | Improvement |
|----------|----------------------|-------------|----------------------|-------------|
| Federated | -11.0 -> -7.4 | +33% | -22.5 -> -16.5 | +27% |
| Centralized (MARL) | -11.0 -> -7.7 | +30% | -22.5 -> -17.1 | +24% |
| Decentralized (SARL) | -10.1 -> -7.1 | +30% | -22.5 -> -17.4 | +23% |

### Evaluation Performance — Waiting Time (seconds, lower = better)

| Strategy | Grid 3x3 | Grid 5x5 |
|----------|--------:|--------:|
| FedRL | 11.5 | 16.8 |
| MARL | 13.9 | 23.6 |
| SARL | 10.4 | 17.4 |
| Fixed-Time | 76.9 | 70.6 |

### Evaluation Performance — Travel Time (seconds, lower = better)

| Strategy | Grid 3x3 | Grid 5x5 |
|----------|--------:|--------:|
| FedRL | 56.4 | 83.5 |
| MARL | 62.0 | 91.5 |
| SARL | 54.8 | 83.7 |
| Fixed-Time | 117.3 | 123.6 |

### Communication Cost (MB, theoretical, 200k timesteps)

| Strategy | Grid 3x3 | Grid 5x5 | Reduction vs Centralized |
|----------|---------|---------|--------------------------|
| SARL (Centralized) | 174.6 | 485.0 | -- |
| MARL | 162.3 | 451.2 | ~7% |
| FedRL | 102.6 | 285.0 | 41% |

### Key Findings

1. **All RL strategies reduce waiting time by 75-85%** vs fixed-time control
2. **FedRL uses 41% less communication** than centralized approaches
3. **FedRL's advantage grows with network size** — trails on 3x3 (11.5s vs 10.4s), leads on 5x5 (16.8s vs 17.4s)
4. **MARL performs worst among RL strategies** — local specialization without knowledge sharing hurts
5. **FedProx shows modest 4% improvement** (11.0s vs 11.5s at mu=0.01) — expected to grow with heterogeneity

### FedProx Ablation (Grid 3x3)

| Config | Avg Wait (s) | Change vs Baseline |
|--------|------------:|--------------------|
| FedAvg (mu=0.0) | 11.5 | -- |
| FedProx (mu=0.01) | 11.0 | -4.3% |
| FedProx (mu=0.1) | 11.2 | -2.6% |

### Time-of-Day Ablation (Grid 3x3)

| Config | Avg Wait (s) |
|--------|------------:|
| Fixed Demand (360 VPLPH) | 11.5 |
| ToD + Time Encoding (200-700 VPLPH) | 12.4 |

ToD-trained model handles variable demand with only 0.9s additional waiting time despite training on a harder, more variable task.

---

## 11. EXTENSIONS IMPLEMENTED

### FedProx (mu parameter)
- Adds proximal term to PPO loss: `L = L_ppo + (mu/2) * ||w - w_global||^2`
- Prevents local models drifting from consensus
- Designed for heterogeneous intersection types
- Implementation: `seal/trainer/fedprox_policy.py`

### Cooperative Reward Shaping (alpha parameter)
```
r = alpha * r_local + (1 - alpha) * mean(r_neighbors)
```
- alpha = 1.0: fully selfish (default)
- alpha = 0.5: equal weight local and neighbors
- alpha = 0.1: near-fully cooperative
- Implementation: `seal/sumo/env.py`

### Time-of-Day Demand Curriculum
| Period | VPLPH Range |
|--------|-------------|
| AM Rush | 500-700 |
| Midday | 200-300 |
| PM Rush | 400-600 |

Combined with sin/cos time encoding at obs indices 14-15.

---

## 12. LITERATURE CONTEXT

### Field Size
Fewer than 10 papers worldwide on federated RL for traffic signal control.

### Key Papers

| Paper | Year | Venue | Unique Contribution |
|-------|------|-------|---------------------|
| FedLight (Ye et al.) | 2021 | DAC | First FedRL for traffic signals |
| SEAL (Hudson et al.) | 2022 | SMARTCOMP | Intersection-agnostic representation |
| Bao et al. | 2023 | Scientific Reports | Partial model aggregation |
| Fed-PPO (Li et al.) | 2025 | Scientific Reports | Soft-update aggregation |
| HFRL (Fu et al.) | 2025 | arXiv | Hierarchical clustering of intersections |
| FitLight (Ye et al.) | 2025 | arXiv | Imitation learning bootstrap |

### Foundational References
- McMahan et al. (2017) — FedAvg. AISTATS.
- Li et al. (2020) — FedProx. MLSys.
- Ault & Sharon (2021) — RESCO benchmark. NeurIPS.
- Schulman et al. (2017) — PPO. arXiv.
- DRLE (Zhou et al.) — Decentralized RL at the Edge for traffic light control.

### Our Positioning
- RESCO benchmarks single-agent and multi-agent RL
- LibSignal benchmarks across simulators
- **Nobody benchmarks federated approaches under controlled conditions** — that's our gap
- We are not proposing a new algorithm; we provide the standardized framework + fair comparison

---

## 13. TEAM & CONTRIBUTIONS

| Person | Responsibility |
|--------|---------------|
| Majid | System architecture, evaluation framework, RL agent design |
| Hamdan | Federated learning loop, FedAvg and FedProx aggregation |
| Abdallah | SUMO simulation environment, observation space |
| Mohammad | Baseline controllers, evaluation protocol, Monte Carlo pipeline |
| Mohamed | Literature review, benchmark scenario design, results analysis, report |

---

## 14. PRESENTATION HISTORY

### Midterm Presentation Structure (16 slides, ~9.5 minutes, 5 speakers)

| Slide | Title | Speaker |
|-------|-------|---------|
| 1 | Title | Al Rayassi |
| 2 | Research Context | Al Rayassi |
| 3 | Objectives | Al Rayassi |
| 4 | Related Work | Al Rayassi |
| 5 | Benchmarking Framework | Al Rayassi |
| 6 | Simulation Layer | Majid |
| 7 | The RL Agent | Majid |
| 8 | Dataset | Majid |
| 9 | Three Strategies | Majid |
| 10 | Demand Settings | Majid |
| 11 | Learning Curves | Abdallah |
| 12 | Waiting Time | Abdallah |
| 13 | Communication Cost | Abdallah |
| 14 | Trade-off Analysis | Blooshi |
| 15 | Next Steps | Blooshi |
| 16 | Contributions + Thank You | Blooshi |

---

## 15. WHAT'S CHANGED SINCE MIDTERM

### Major Additions
1. **7 new training strategies** — expanded from 3 (SARL/MARL/FedRL) to 10 (+ Gossip, HierFed, FedDistill, MeanField, CTDE, fixed-time, max-pressure)
2. **Experiment design overhaul** — from ad-hoc single-demand to Tier 1 (270 runs) + Tier 2 (targeted ablations)
3. **Multiple training seeds** — 3 seeds per config to separate training vs eval variance
4. **Multiple demand levels** — low (150), medium (360), high (600) VPLPH
5. **Cologne-8 real-world topology** — first non-synthetic network
6. **Statistical upgrades** — Bonferroni correction, bootstrap CI, rank-biserial effect size
7. **Codex audit** — independent review found and fixed critical issues (MARL eval, alpha computation, results consistency)
8. **Per-process SUMO route isolation** — enables safe parallel experiments on HPC

### Bug Fixes from Audit
- MARL evaluation now preserves per-agent specialization (multi-policy save format)
- Alpha (aggregation weight) computation uses shift-normalize for negative rewards
- `save_campaign_results()` appends instead of overwriting
- FedRL episode_data reset after aggregation (prevents cumulative weighting bias)
- training_data initialized in BaseTrainer.__init__() (prevents subclass write failures)

---

## 16. PAPER TARGET TIERS

### Tier 1: ITSC 2026
- Learning curves with confidence bands (3 seeds)
- Explicit limitations section
- Threats to validity
- Main results table: 10 strategies x 3 topologies x 3 demands with bootstrap CI
- Align all claims with actual code

### Tier 2: IEEE T-ITS
- Convergence analysis (episodes-to-threshold)
- Cross-topology transfer experiment
- Mechanistic explanation (weight divergence, policy entropy)
- Hierarchical statistical analysis
- Wall-clock and sample efficiency
- Per-intersection fairness (Gini coefficient)
- Demand heterogeneity narrative

### Tier 3: NeurIPS D&B
- Multi-simulator (CityFlow)
- Multi-algorithm (DQN, A2C)
- Scale to 16+ intersections
- Open-source benchmark release
- Privacy analysis
- Real-world network validation
- Robustness experiments

---

## 17. TRADE-OFF ANALYSIS FRAMEWORK

### The Deliverable Table

| Strategy | Best When | Worst When | Communication | Scalability |
|----------|-----------|------------|---------------|-------------|
| SARL | Small networks, rich data | Large networks, bandwidth-limited | Highest | Poor |
| MARL | Per-intersection specialization needed | Communication-constrained | High | Moderate |
| FedRL | Large networks, privacy needed | Very small networks | Low | Good |
| Gossip | No central server available | Need global consensus fast | Medium | Good |
| HierFed | Clustered intersection types | Uniform networks | Medium | Good |
| FedDistill | Extreme bandwidth constraints | Need full weight fidelity | Very Low | Good |
| MeanField | Implicit coordination sufficient | Strong coordination needed | Low | Good |
| CTDE | Training compute available | Distributed-only deployment | High (training) | Moderate |
| Fixed-Time | Very low demand, no compute | Any non-trivial demand | None | N/A |
| Max-Pressure | No training data available | Complex signal coordination | None | Good |

---

## 18. KEY DESIGN DECISIONS & RATIONALE

### Why RL for Traffic Signals?
Traffic is dynamic and stochastic. RL learns directly from interaction without needing a closed-form model. Can discover strategies that outperform any fixed or rule-based policy.

### Why PPO?
Most common in FedRL traffic literature. Stable for discrete action spaces. Held constant across all strategies — it's a controlled variable, not a contribution.

### Why Binary Action (Not Phase Selection)?
Phase selection = Discrete(N) where N varies by intersection type. Can't share weights between different output dimensions. Binary = Discrete(2) everywhere, essential for federated aggregation.

### Why 14 Features?
3 traffic flow (physical state) + 7 phase ratios (intersection-agnostic light state) + 4 ranking (relative congestion context). Each group has a purpose.

### Why Ratios Instead of Raw Counts?
A 2-lane intersection with 10 cars vs a 4-lane with 10 cars are different situations. Ratios normalize by capacity, enabling comparison across intersection types.

### Why Ranking Features?
Without them: "I have 60% occupancy." With them: "I'm the most congested in my neighborhood." Enables cross-topology transfer.

### Why Reward-Weighted Aggregation?
Center intersections serve 3-4x more traffic. Equal weighting dilutes their hard-won knowledge. Reward weighting propagates useful patterns faster.

### Why Synthetic Grids?
1. Controlled heterogeneity — design exactly how intersections differ
2. Reproducibility — same seed = same traffic
3. Standard practice — FedLight, SEAL, HFRL all use SUMO grids
4. Now also testing Cologne-8 real-world topology

---

## 19. KNOWN LIMITATIONS (For Honest Slides)

1. Synthetic grids + one real-world topology (Cologne-8) — limited real-world validation
2. No turning vehicles — isolates signal control variable but reduces realism
3. Communication cost is theoretical (computed from data structure sizes), not measured
4. PPO only — algorithm independence not yet tested
5. No formal privacy guarantees (no differential privacy)
6. No convergence proofs
7. Fixed-time baseline is "always stay in current phase" — not industry-standard actuated control
8. Training seed independence not fully validated
9. FedRL weight reset is an optimization confound vs SARL/MARL (acknowledged, fundamental to FL)

---

## 20. CURRENT PROJECT STATUS

- **Phase 9 of 10:** Pre-Experiment Hardening (ready for HPC)
- **All 10 strategies implemented and tested**
- **Codex audit findings addressed**
- **Next:** Run Tier 1 experiments on HPC (270 runs)
- **After HPC:** Phase 10 post-experiment analysis (Pareto curves, fairness, mechanistic explanation)

### Implementation Progress

| Component | Status |
|-----------|--------|
| SUMO + TraCI + Ray 2.x environment | Complete |
| SEAL engine modernized (Ray 2.x) | Complete |
| API endpoints (train, simulate, evaluate) | Complete |
| Evaluation framework (runner, MC, metrics) | Complete |
| Extensions (FedProx, cooperative, ToD) | Complete |
| 10 training strategies | Complete |
| Experiment pipeline (campaign runner) | Complete |
| Statistical analysis (Wilcoxon, bootstrap CI) | Complete |
| HPC experiment execution | Pending |
| Post-experiment analysis | Pending |
| Paper writing | Pending |

---

## 21. GENERATED VISUALIZATIONS

| Figure | File | Description |
|--------|------|-------------|
| Learning Curves | fig5_learning_curves.png | Convergence for all strategies on both topologies |
| Communication Cost | fig6_communication_cost.png | Cumulative comm cost over training time |
| Evaluation Metrics | fig7_evaluation_metrics.png | Travel time + waiting time bar charts |
| FedProx Ablation | chart_fedprox_ablation.png | Convergence + evaluation for mu sweep |
| ToD Ablation | chart_tod_ablation.png | Fixed vs variable demand comparison |
| Strategy x Topology | chart_strategy_topology_heatmap.png | Heatmap of waiting times |
| Combined Comparison | chart_combined_comparison.png | All configs side by side |
| Training Strategies | chart_training_strategies.png | Architecture diagrams for SARL/MARL/FedRL |

All figures in `BackEnd/results/figures/`.

---

## 22. KEY FILES REFERENCE

| Purpose | File |
|---------|------|
| FedRL trainer | seal/trainer/fed_agent.py |
| MARL trainer | seal/trainer/multi_agent.py |
| SARL trainer | seal/trainer/single_agent.py |
| Gossip trainer | seal/trainer/gossip_agent.py |
| HierFed trainer | seal/trainer/hierfed_agent.py |
| FedDistill trainer | seal/trainer/feddistill_agent.py |
| MeanField trainer | seal/trainer/mean_field_agent.py |
| CTDE trainer | seal/trainer/ctde_agent.py |
| FedProx policy | seal/trainer/fedprox_policy.py |
| FedDistill policy | seal/trainer/feddistill_policy.py |
| Aggregation weights | seal/trainer/weight_aggr.py |
| Observation computation | seal/sumo/kernel/trafficlight/light.py |
| Reward + environment | seal/sumo/env.py |
| Feature indices | seal/sumo/config.py |
| Route generation + ToD | seal/sumo/abstract_env.py |
| MeanField env wrapper | seal/sumo/mean_field_env.py |
| CTDE env wrapper | seal/sumo/ctde_env.py |
| Trainer factory | api/training_runner.py |
| Evaluation runner | api/evaluation/runner.py |
| Monte Carlo pipeline | api/evaluation/monte_carlo.py |
| Campaign runner | scripts/run_full_experiments.py |
| Ablation builder | scripts/run_extension_ablation.py |
| Table generation | scripts/generate_tables.py |
| Report generation | scripts/generate_report.py |

---

## 23. Q&A PREPARATION — ANTICIPATED QUESTIONS

### "What exactly is YOUR contribution?"
The standardized benchmarking framework: 8 controlled layers, intersection-agnostic observation space, 10-strategy fair comparison. No existing work compares strategies under identical conditions.

### "Show me the model / architecture"
Two hidden layers x 256 neurons, ReLU, input=14, output=2. Deliberately default — architecture is a controlled variable. Same across all strategies.

### "Why this model / method? What alternatives?"
PPO is the standard in FedRL traffic literature. Algorithm is a controlled variable, not a contribution.

### "Training is noisy / too few epochs"
Curves plateau by episode 20. Separate 50-episode run confirmed same plateau. Relative ordering is stable.

### "Dataset is too small"
Acknowledged — second half includes low/medium/high demand (270 configurations), plus Cologne-8 real topology.

### "Where is the deployment?"
Simulation-based study, standard for this stage. All 7 papers in FedRL traffic evaluate in simulation. Policy inference is lightweight enough for edge hardware.

### "SARL beats FedRL on 3x3 — doesn't that undermine federation?"
That's exactly what a benchmark should show. On small networks, one shared policy suffices. FedRL's advantage emerges at scale (leads on 5x5). The benchmark reveals WHEN each strategy wins.

### "Why does MARL perform worst?"
MARL agents optimize locally with no awareness of neighbors. Can push congestion downstream. FedRL gets local specialization + periodic knowledge sharing. SARL inherently coordinates via shared policy.

### "How is this an IoT project?"
Traffic lights are distributed edge devices on wireless networks. Training a network of IoT devices without centralizing data is an IoT systems question. Federated learning was designed for this.

### "What about privacy?"
Raw observation data never leaves the intersection — only model weights shared. Standard FL privacy notion. We do NOT claim differential privacy — that requires formal (epsilon, delta) guarantees.

### "How would you deploy this?"
Each light gets edge compute (RPi/Jetson). Inference is microseconds for a 256x256 MLP on 14 inputs. Sensors provide traffic flow observations. Training happens offline. Periodic federated updates over V2X wireless infrastructure.

---

## 24. RELATED WORK COMPARISON

| Paper | Year | Venue | What They Did | How We Differ |
|-------|------|-------|---------------|---------------|
| FedLight (Ye et al.) | 2021 | DAC | First FedRL for traffic, A2C | We use PPO, 10 strategies, 3 topologies |
| SEAL (Hudson et al.) | 2022 | SMARTCOMP | Intersection-agnostic obs | We train from scratch, add rankings, standardized comparison |
| Bao et al. | 2023 | Sci Reports | Partial aggregation, DQN | Different algo/obs — incomparable without our framework |
| Fed-PPO (Li et al.) | 2025 | Sci Reports | Soft-update aggregation | One variant; we compare multiple |
| HFRL (Fu et al.) | 2025 | arXiv | Hierarchical clustering | Real data; we focus on controlled comparison |
| RESCO (Ault & Sharon) | 2021 | NeurIPS | RL traffic benchmark | Benchmarks single/multi-agent; we add federated |

---

## 25. NUMBERS QUICK REFERENCE

- **$87B** — annual US congestion cost (INRIX 2019)
- **75-85%** — waiting time reduction, RL vs fixed-time
- **41%** — communication reduction, FedRL vs centralized
- **10** — total training strategies compared
- **270** — Tier 1 experiment runs planned
- **14** — observation features per intersection (16 with time encoding)
- **3** — topologies (grid-3x3, grid-5x5, cologne-8)
- **3** — demand levels (150, 360, 600 VPLPH)
- **3** — training seeds per config
- **10** — Monte Carlo eval runs per config
- **< 10** — total papers worldwide on FedRL for traffic signals
- **8** — controlled layers in benchmarking framework
- **50KB** — approximate weight payload per FedRL aggregation round
- **256x256** — MLP hidden layer dimensions
- **Discrete(2)** — action space (keep/switch)
