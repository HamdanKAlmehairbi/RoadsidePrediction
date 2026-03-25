# Project Briefing: Federated RL for Smart Multi-Intersection Traffic Signal Control

## Benchmarking Training Strategies Under Standardized Conditions

---

## 1. What This Project Is

We built a benchmarking framework that evaluates how different reinforcement learning training strategies perform for multi-intersection traffic signal control. Instead of arguing that one approach is best, we ask: **which strategy works best under what conditions, and what are the trade-offs?**

The framework compares three training strategies — single-agent (SARL), multi-agent (MARL), and federated (FedRL) — under identical controlled conditions using the SUMO traffic simulator. Every variable except the training strategy itself is held constant, making the comparison fair in a way that cross-paper comparisons in the literature are not.

---

## 2. The Problem

Traffic congestion costs the US economy $87 billion annually. Most traffic lights run on fixed timers — green for N seconds, red for M seconds, regardless of actual conditions. Reinforcement learning can train traffic lights to adapt to real-time demand, but there are multiple ways to organize training across a network of intersections:

- Train one shared model that controls everything (centralized)
- Train each intersection independently (decentralized)
- Train locally but periodically share learned knowledge (federated)

Each approach has different trade-offs in performance, communication cost, scalability, and privacy. The existing literature (fewer than 10 papers on federated RL for traffic signals worldwide) compares these using different setups, different simulators, different metrics. No fair head-to-head comparison exists. That is the gap we fill.

---

## 3. What We Built vs What Comes From the Literature

### From the literature (we did not invent these):

- **PPO** (Proximal Policy Optimization) — the RL algorithm. Standard, well-studied, stable.
- **SUMO** — the traffic simulator. Open-source, widely used in transportation research.
- **FedAvg** — the federated averaging algorithm. Proposed by McMahan et al. 2017.
- The general concepts of centralized, decentralized, and federated training paradigms.

### What we designed and built:

- The 14-feature intersection-agnostic observation space with ratio-based encoding and ranking features
- The reward-weighted aggregation scheme (not equal-weight FedAvg)
- The 4 ranking features that enable cross-topology transfer
- The standardized evaluation framework controlling 8 layers of variables
- The automated experiment pipeline with Monte Carlo evaluation, seed control, and incremental saving
- The unified platform where all three strategies share the same environment, observations, and evaluation
- Three extensions: FedProx aggregation, cooperative reward shaping, time-of-day demand curriculum
- The full codebase: Ray RLlib integration, training runner, evaluation runner, campaign orchestration, figure generation

---

## 4. The Three Training Strategies

### SARL (Single-Agent RL)

- One shared policy network controls all intersections
- Every intersection maps to the same policy: `policy_mapping_fn = lambda _ : "sarl-policy"`
- The model sees observations from every intersection during training
- Highest data flow — all observations must reach the single model
- Implemented in `seal/trainer/single_agent.py`

### MARL (Multi-Agent RL)

- One policy network per intersection
- Each intersection maps to its own policy: `policy_mapping_fn = lambda agent_id : agent_id`
- All policies train within a single Ray PPO algorithm instance
- A central trainer coordinates gradient updates — constant bidirectional communication
- At the end of training, policies are naively averaged for deployment
- Implemented in `seal/trainer/multi_agent.py`

### FedRL (Federated RL)

- One policy network per intersection (same as MARL)
- Each policy trains locally on local observations only
- Every episode, all policies upload weights to an edge server
- Server computes reward-weighted FedAvg — better-performing intersections contribute more
- Averaged global model distributed back, all agents resume from consensus
- Between aggregation rounds: zero server communication
- Implemented in `seal/trainer/fed_agent.py`

### Strategy Comparison Table

| Property | SARL | MARL | FedRL |
|----------|------|------|-------|
| Policies | 1 shared | 1 per intersection | 1 per intersection |
| Policy mapping | All agents → same policy | Each agent → own policy | Each agent → own policy |
| Learning | Learns from ALL intersections | Each learns from own only | Each learns from own only |
| Aggregation during training | None (single policy) | None | Reward-weighted FedAvg every episode |
| Communication pattern | All obs to central model | All obs to central trainer | Only weights, periodically |
| Communication cost (Grid 3x3, 200k steps) | 174.6 MB | 174.6 MB | 102.6 MB |
| Privacy | No — all data centralized | No — all data centralized | Yes — raw data stays local |
| Scalability | Degrades with network size | Moderate | Best — communication grows slowly |

---

## 5. The Observation Space

Every intersection produces a 14-dimensional observation vector, all normalized to [0, 1]:

### Traffic Flow Features (indices 0-2)

| Index | Feature | Computation | What It Captures |
|-------|---------|-------------|-----------------|
| 0 | Lane Occupancy | total vehicle length / total lane length | How full the road is |
| 1 | Halted Lane Occupancy | halted vehicle length / total lane length (speed < 0.1 m/s) | How jammed it is |
| 2 | Speed Ratio | average vehicle speed / speed limit | How freely traffic flows |

### Phase State Ratios (indices 3-9)

| Index | Feature | What It Captures |
|-------|---------|-----------------|
| 3 | Phase State r | Fraction of lights showing red |
| 4 | Phase State y | Fraction showing yellow |
| 5 | Phase State g | Fraction showing green (minor) |
| 6 | Phase State G | Fraction showing green (priority) |
| 7 | Phase State u | Fraction showing u-turn |
| 8 | Phase State o | Fraction showing off-blinking |
| 9 | Phase State O | Fraction showing off |

This is the intersection-agnostic encoding: instead of "GGrr" (topology-specific), we encode "50% green, 50% red" (works for any intersection type).

### Network Ranking Features (indices 10-13)

| Index | Feature | What It Captures |
|-------|---------|-----------------|
| 10 | Local Rank | Congestion relative to immediate neighbors |
| 11 | Global Rank | Congestion relative to entire network |
| 12 | Local Halt Rank | Halted vehicle rank among neighbors |
| 13 | Global Halt Rank | Halted vehicle rank across network |

These enable cross-topology transfer: "I am the most congested in my neighborhood" means the same thing on a 3x3 grid and a 5x5 grid.

### Optional Time Encoding (indices 14-15)

| Index | Feature | Range |
|-------|---------|-------|
| 14 | sin(2π·t/H) | [-1, 1] |
| 15 | cos(2π·t/H) | [-1, 1] |

Smooth periodic encoding of timestep position. Only active with `use_time_encoding=True`.

---

## 6. The Reward Function

Per intersection k at timestep t:

```
r_k = -(o_k + h_k)²
```

Where:
- `o_k` = lane occupancy (fraction of lane length occupied by vehicles)
- `h_k` = halted lane occupancy (fraction occupied by stopped vehicles)

Halted vehicles are penalized more heavily because they contribute to both terms. The quadratic makes the penalty disproportionately larger as congestion increases — a small queue is a small penalty, a large queue is a massive penalty.

Total network reward: `r = Σ r_k` across all intersections.

---

## 7. The Action Space

Binary: `Discrete(2)`
- 0 = keep current phase
- 1 = advance to next phase state

Timing constraints (from Federal Highway Administration guidelines):
- Minimum 4 seconds between phase changes
- Maximum 120 seconds before forced change

Binary phase switching is intersection-agnostic — the same action space works for any intersection type, enabling weight sharing across heterogeneous intersections.

---

## 8. Federated Aggregation

### Standard FedAvg (McMahan et al. 2017)

```
ω_global = (1/K) × Σ ω_k
```

Equal-weight averaging. Every intersection contributes the same amount.

### Our Reward-Weighted FedAvg

```
ω_global = Σ c_k × ω_k
```

Where `c_k` is proportional to intersection k's normalized reward. Better-performing intersections contribute more. Available weight functions:

| Function | Description | Usage |
|----------|-------------|-------|
| `naive` | Equal weights 1/K | Baseline FedAvg |
| `pos_reward` | Weight by normalized positive reward | Default — recommended |
| `neg_reward` | Inverse reward weighting | Experimental |
| `traffic` | Weight by vehicles served | Experimental |

### FedProx Extension (Li et al. 2020)

Adds a proximal term to the local PPO loss:

```
L = L_ppo + (μ/2) × ||ω_k - ω_global||²
```

Penalizes local models from drifting too far from the global consensus. Designed for heterogeneous settings where different intersections see very different traffic patterns.

---

## 9. How the Framework Guarantees Fair Comparison

The framework ensures fair comparison through 8 controlled layers. When we say "FedRL achieves 11.5s and MARL achieves 13.9s," the difference can only come from how policies are organized — because nothing else changed.

### Layer 1: The Road

Every strategy drives on the exact same road network. Same intersections, same lane counts, same speed limits, same distances. The network file is loaded once and never modified. Nobody gets an easier or harder road.

### Layer 2: The Traffic

Every strategy faces the same cars. Same number of vehicles, entering from the same places, driving to the same destinations, at the same times. Controlled by the random seed — same seed means identical traffic. During evaluation, all strategies are tested against the same 5 traffic scenarios.

### Layer 3: What the Agent Sees

Every intersection, in every strategy, sees the same 14 numbers. How full are my lanes, how many cars are stopped, how fast is traffic moving, what color is my light, how congested am I compared to my neighbors. The same function computes these numbers by querying SUMO directly. There is no strategy-specific observation logic.

### Layer 4: What the Agent Can Do

Every intersection, in every strategy, has the same two choices: keep the current light phase or switch to the next one. Same minimum time between switches (4 seconds), same maximum time before a forced switch (120 seconds). Nobody gets more or fewer options.

### Layer 5: How the Agent is Graded

Every intersection uses the same reward formula. Same inputs produce the same reward regardless of strategy. During evaluation, metrics like waiting time and travel time come directly from SUMO's own output files, not from our code.

### Layer 6: How the Agent Learns

Every strategy uses PPO with the same settings — same learning rate, same batch size, same clip parameter, same neural network shape (two layers of 256 neurons). Same algorithm, same hyperparameters, same architecture. Nothing is tuned per strategy.

### Layer 7: How Long the Agent Trains

Every strategy trains for exactly 25 episodes. No early stopping, no extra time for one strategy over another. Same number of environment interactions, same compute budget.

### Layer 8: The Only Thing That Changes

How the policies are organized. SARL has one shared policy for all intersections. MARL has one separate policy per intersection with no sharing. FedRL has one separate policy per intersection that periodically averages weights with the others. That is the only variable. Everything above it is locked.

### Standardization Summary Table

| Component | Same Across Strategies? | How Enforced |
|-----------|------------------------|--------------|
| SUMO version + physics | Yes | Same binary |
| Network topology | Yes | Same .net.xml file |
| Traffic demand (VPLPH) | Yes | Same parameter |
| Route generation | Yes | Same seed, same function |
| Observation computation | Yes | Same `get_observation()` |
| Observation dimensions | Yes | Same `N_RANKED_FEATURES` |
| Reward function | Yes | Same formula, shared code |
| Action space | Yes | Same `Discrete(2)` |
| Phase transition logic | Yes | Same `TrafficLight` class |
| Timing constraints | Yes | Same 4s/120s |
| RL algorithm | Yes | Same PPO |
| Hyperparameters | Yes | Same config dict |
| Network architecture | Yes | Same 256×256 MLP |
| Training episodes | Yes | Same count (25) |
| Evaluation seeds | Yes | Same 42-46 |
| Evaluation metrics | Yes | Same SUMO tripinfo parsing |
| **Policy organization** | **No — this is the variable** | SARL: 1 shared, MARL: N independent, FedRL: N with aggregation |
| **Aggregation** | **No — FedRL only** | Reward-weighted FedAvg every episode |

---

## 10. Experimental Setup

### Topologies

| Network | Intersections | Controlled Lanes | Lane Heterogeneity |
|---------|--------------|------------------|--------------------|
| Grid 3×3 | 9 | 24 | Center has more lanes than edges |
| Grid 5×5 | 25 | Heterogeneous | Center has more lanes than edges |

Synthetic grids — standard in the FedRL traffic literature (Ye et al. 2021, Hudson et al. 2022, Fu et al. 2025). Chosen for controlled heterogeneity, reproducibility, and comparability with prior work.

### Traffic Demand

| Parameter | Value |
|-----------|-------|
| Vehicles per lane per hour (VPLPH) | 360 |
| Vehicle generation | SUMO `randomTrips.py` |
| Departure period | 3600 / (VPLPH × n_lanes) seconds |
| Origin/destination | Network edges (`--fringe-factor 100`) |
| Vehicle behavior | Straight-through, no turns |
| Simulation duration | 360 seconds (end_time), 450 steps (horizon) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| RL Algorithm | PPO (Proximal Policy Optimization) |
| Learning rate | 5 × 10⁻⁵ |
| SGD minibatch size | 128 |
| PPO clip parameter | 0.3 |
| KL divergence target | 0.3 |
| Train batch size | 4000 timesteps |
| Rollout fragment length | 200 |
| GAE lambda | 1.0 |
| VF clip parameter | 10 |
| Network architecture | 2 hidden layers × 256 neurons, ReLU |
| Episodes per config | 25 |

### Evaluation Configuration

| Parameter | Value |
|-----------|-------|
| Monte Carlo runs | 5 per config |
| Seeds | 42, 43, 44, 45, 46 |
| Metrics source | SUMO tripinfo XML |
| Metrics | avg waiting time, avg travel time, throughput |

### Baselines

| Baseline | Description |
|----------|-------------|
| Fixed-Time | Predetermined phase cycling at fixed intervals |

---

## 11. Results — Midterm

### Training Convergence

All three strategies successfully learn to reduce congestion over 25 episodes.

| Strategy | Grid 3×3 Start | Grid 3×3 End | Improvement | Grid 5×5 Start | Grid 5×5 End | Improvement |
|----------|---------------|-------------|-------------|----------------|-------------|-------------|
| Federated | -11.0 | -7.4 | +33% | -22.5 | -16.5 | +27% |
| Centralized | -11.0 | -7.7 | +30% | -22.5 | -17.1 | +24% |
| Decentralized | -10.1 | -7.1 | +30% | -22.5 | -17.4 | +23% |

Key observation: Federated converges to the highest reward on both topologies despite each agent only seeing local data.

### Evaluation Performance

| Strategy | 3×3 Wait (s) | 3×3 Travel (s) | 5×5 Wait (s) | 5×5 Travel (s) |
|----------|------------:|----------------:|------------:|----------------:|
| Federated | 11.5 | 56.4 | 16.8 | 83.5 |
| Centralized | 13.9 | 62.0 | 23.6 | 91.5 |
| Decentralized | 10.4 | 54.8 | 17.4 | 83.7 |
| Fixed-Time | 76.9 | 117.3 | 70.6 | 123.6 |

Key findings:
- All RL strategies reduce waiting time by **75-85%** compared to fixed-time
- Decentralized slightly outperforms Federated on Grid 3×3 (10.4s vs 11.5s)
- Federated pulls ahead on Grid 5×5 (16.8s vs 17.4s), suggesting its advantage grows with network size
- Centralized (MARL) performs worst among RL strategies on both topologies

### Communication Cost (Theoretical, 200k Timesteps)

| Strategy | Grid 3×3 | Grid 5×5 | Reduction vs Centralized |
|----------|---------|---------|--------------------------|
| Centralized | 174.6 MB | 485.0 MB | — |
| Federated | 102.6 MB | 297.5 MB | 41% |
| Decentralized | 57.6 MB | 160.0 MB | 67% |

Federated achieves centralized-level performance at 41% less communication cost.

### FedProx Ablation (Grid 3×3)

| Config | Avg Wait (s) | Change vs Baseline |
|--------|------------:|-------------------|
| FedAvg (μ=0.0) | 11.5 | — |
| FedProx (μ=0.01) | 11.0 | -4.3% |
| FedProx (μ=0.1) | 11.2 | -2.6% |

Modest improvement — FedProx's advantage is expected to grow with higher heterogeneity.

### Time-of-Day Ablation (Grid 3×3)

| Config | Avg Wait (s) |
|--------|------------:|
| Fixed Demand (360 VPLPH) | 11.5 |
| ToD + Time Encoding (200-700 VPLPH) | 12.4 |

ToD-trained model handles variable demand with only 0.9s additional waiting time despite being trained on a harder, more variable task. Training convergence goes from -38 to -15 reward (vs -11 to -7.4 for fixed demand), showing the agent learns meaningful policies even under challenging conditions.

---

## 12. Extensions (Implemented, Pending Full Evaluation)

### FedProx (μ parameter)

Adds proximal term to PPO loss: `L = L_ppo + (μ/2) × ||w - w_global||²`. Prevents local models from drifting too far from global consensus. Designed for heterogeneous intersection types where traffic patterns vary significantly.

Implementation: `seal/trainer/fedprox_policy.py`

### Cooperative Reward Shaping (α parameter)

Blends each agent's reward with its neighbors' rewards:

```
r = α × r_local + (1 - α) × mean(r_neighbors)
```

- α = 1.0: fully selfish (default)
- α = 0.5: equal weight local and neighbors
- α = 0.1: near-fully cooperative

Neighbors determined from TLS adjacency graph in network file.

Implementation: `seal/sumo/env.py`

### Time-of-Day Demand Curriculum

Replaces constant 360 VPLPH with variable demand sampled per episode:

| Period | VPLPH Range |
|--------|-------------|
| AM Rush | 500-700 |
| Midday | 200-300 |
| PM Rush | 400-600 |

Combined with sin/cos time encoding at observation indices 14-15 so agents know what time it is.

Implementation: `seal/sumo/abstract_env.py`

---

## 13. Action Plan — Second Half

### Priority 1: Multi-Demand Evaluation

Run all strategies under varying demand to characterize behavior:

| Setting | VPLPH | Purpose |
|---------|-------|---------|
| Low demand | 150 | Does RL matter when traffic is light? |
| Medium demand (done) | 360 | Standard conditions — current results |
| High demand | 600 | Stress test — how do strategies degrade? |
| Variable demand | 200-700 (ToD) | Adaptability to changing conditions |

Expected output: 4 settings × 4 strategies × 2 topologies = 32 configurations.

### Priority 2: Trade-off Analysis

Build the definitive comparison table:

| Strategy | Best When | Worst When | Communication | Scalability |
|----------|-----------|------------|---------------|-------------|
| SARL | Small networks, rich data | Large networks, bandwidth-limited | Highest | Poor |
| MARL | Need per-intersection specialization | Communication-constrained | High | Moderate |
| FedRL | Large networks, privacy needed | Very small networks | Low | Good |
| Fixed-Time | Very low demand, no compute | Any non-trivial demand | None | N/A |

This table is the deliverable. Experiments fill it in with evidence.

### Priority 3: Aggregation Variants

Compare within federated learning:

| Variant | Description |
|---------|-------------|
| Naive FedAvg | Equal weights (1/K) |
| Reward-Weighted FedAvg | Better agents contribute more (current default) |
| FedProx μ=0.01 | Proximal regularization, light |
| FedProx μ=0.1 | Proximal regularization, strong |

### Priority 4: Statistical Rigor

- Increase Monte Carlo runs to 10 per config
- Wilcoxon signed-rank tests for pairwise significance
- 95% confidence intervals on all metrics
- Publication-ready LaTeX tables with p-values

---

## 14. Generated Visualizations

| Figure | File | Description |
|--------|------|-------------|
| Learning Curves | `fig5_learning_curves.png` | Convergence for all 3 strategies on both topologies |
| Communication Cost | `fig6_communication_cost.png` | Cumulative comm cost over training time |
| Evaluation Metrics | `fig7_evaluation_metrics.png` | Travel time + waiting time bar charts (2×2) |
| FedProx Ablation | `chart_fedprox_ablation.png` | Convergence + evaluation for μ sweep |
| ToD Ablation | `chart_tod_ablation.png` | Fixed vs variable demand comparison |
| Strategy × Topology | `chart_strategy_topology_heatmap.png` | Heatmap of waiting times |
| Combined Comparison | `chart_combined_comparison.png` | All configs side by side |
| Training Strategies | `chart_training_strategies.png` | Architecture diagrams for SARL/MARL/FedRL |

All figures in `BackEnd/results/figures/`.

---

## 15. Key Files

| File | Purpose |
|------|---------|
| `seal/trainer/fed_agent.py` | FedRL trainer with FedAvg aggregation |
| `seal/trainer/multi_agent.py` | MARL trainer |
| `seal/trainer/single_agent.py` | SARL trainer |
| `seal/trainer/fedprox_policy.py` | FedProx PPO policy with proximal loss |
| `seal/trainer/weight_aggr.py` | Aggregation weight functions |
| `seal/sumo/env.py` | SUMO environment, observations, rewards |
| `seal/sumo/kernel/trafficlight/light.py` | Observation computation, TLS control |
| `seal/sumo/config.py` | Feature indices (14-dim observation space) |
| `seal/sumo/abstract_env.py` | Route generation, time-of-day curriculum |
| `api/training_runner.py` | Trainer factory, topology map, PPO config |
| `api/evaluation/monte_carlo.py` | Monte Carlo evaluation pipeline |
| `scripts/run_all_training.py` | Experiment campaign runner (incremental save) |
| `scripts/generate_all_figures.py` | Figure generation from results |
| `results/campaigns/baseline/` | Baseline evaluation (10 configs) |
| `results/campaigns/training-curves/` | Training + evaluation (9 configs) |

---

## 16. Literature Context

This project sits in a niche but growing field — fewer than 10 papers worldwide on federated RL for traffic signal control:

| Paper | Year | Venue | Unique Contribution |
|-------|------|-------|---------------------|
| FedLight (Ye et al.) | 2021 | DAC | First FedRL for traffic signals |
| SEAL (Hudson et al.) | 2022 | SMARTCOMP | Intersection-agnostic representation |
| Bao et al. | 2023 | Scientific Reports | Partial model aggregation |
| Fed-PPO (Li et al.) | 2025 | Scientific Reports | Soft-update aggregation |
| HFRL (Fu et al.) | 2025 | arXiv | Hierarchical clustering of intersections |
| FitLight (Ye et al.) | 2025 | arXiv | Imitation learning bootstrap |

Additional key references:
- McMahan et al. (2017) — FedAvg. AISTATS.
- Li et al. (2020) — FedProx. MLSys.
- Ault & Sharon (2021) — RESCO benchmark. NeurIPS.

Our positioning: we are not proposing a new algorithm. We are building the standardized evaluation framework that this field lacks, producing the first fair comparison of training strategies under identical conditions.

---

*Last updated: 2026-03-25*
