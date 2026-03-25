# Q&A Assistant Context — Federated RL Traffic Signal Control Benchmarking

You are helping a team answer questions after a presentation about their project. This document contains everything you need to answer any question — technical, conceptual, or design-related. Answer confidently, concisely, and honestly. If something is a limitation, say so. If something comes from the literature, say so. If something is the team's design, say so.

---

## PROJECT IDENTITY

**Title:** Federated Reinforcement Learning for Smart Multi-Intersection Traffic Signal Control

**What it is:** A standardized benchmarking framework that evaluates three RL training strategies (single-agent, multi-agent, federated) for traffic signal control under identical conditions using the SUMO simulator.

**What it is NOT:** A paper proposing a new algorithm. The contribution is the evaluation framework and the fair comparison it enables.

**Core value:** We designed an intersection-agnostic observation space that lets different intersection types share learned knowledge through federated aggregation, and we built a unified platform that produces the first fair comparison of SARL, MARL, and FedRL under identical conditions.

**Stage:** Midterm — baseline comparison complete. Multi-demand scenarios, aggregation variants, and full trade-off analysis are second half.

**Team:** 5 members — Majid (architecture/RL design), Hamdan (federated loop/aggregation), Abdallah (SUMO environment), Mohammad (baselines/evaluation), Mohamed (literature/analysis).

---

## THE PROBLEM

Traffic congestion costs the US $87B annually (INRIX 2019). Most traffic lights use fixed timers with no awareness of actual conditions. RL can train them to adapt, but with multiple intersections, the question becomes how to organize training across the network.

Three paradigms exist: centralized (one model sees everything), decentralized (each learns alone), and federated (each learns locally, periodically shares model weights). The literature has fewer than 10 papers on federated RL for traffic signals. Each uses different setups, making comparison impossible. We built the framework that makes it possible.

---

## THE THREE STRATEGIES

### SARL (Single-Agent RL) — "One brain for everyone"
- One shared PPO policy network
- All intersections map to the same policy: `lambda _ : "sarl-policy"`
- The model trains on observations from every intersection simultaneously
- Highest information sharing, highest communication cost
- In the SEAL paper's terminology: this corresponds to "Decentralized" (confusingly named — decentralized means one shared policy trained at each site)
- Code: `seal/trainer/single_agent.py`

### MARL (Multi-Agent RL) — "Separate brains, central coordinator"
- One policy per intersection: `lambda agent_id : agent_id`
- All policies train inside a single Ray PPO algorithm instance
- Central trainer manages gradient updates for all agents
- Each agent specializes for its intersection but requires constant data exchange
- At end of training: policies are naively averaged (equal-weight) for deployment
- In the SEAL paper's terminology: this corresponds to "Centralized"
- Code: `seal/trainer/multi_agent.py`

### FedRL (Federated RL) — "Separate brains, periodic sharing"
- One policy per intersection (same mapping as MARL)
- Each trains locally on its own observations
- Every episode (every 4000 timesteps): all policies upload weights to edge server
- Server computes reward-weighted average — better agents contribute more
- Averaged model distributed back, all resume from consensus
- Between rounds: zero server communication, only local V2I data
- Code: `seal/trainer/fed_agent.py`

### The ONLY code difference between MARL and FedRL
About 15 lines in `on_data_recording_step()` — the FedAvg aggregation block:
```python
if aggregate_this_round:
    policy_dict = {id: trainer.get_policy(id) for id in policies}
    new_params = self.fedavg(policy_dict)
    for id in policies:
        trainer.get_policy(id).set_weights(new_params)
```
Everything else is shared code.

---

## THE OBSERVATION SPACE — 14 FEATURES

All normalized to [0, 1]. Same function, same code, same features for every strategy.

**Traffic Flow (indices 0-2):**
- Index 0: Lane Occupancy — `sum(vehicle_lengths) / sum(lane_lengths)` across controlled lanes. How full the road is.
- Index 1: Halted Lane Occupancy — same but only vehicles with speed < 0.1 m/s. How jammed it is.
- Index 2: Speed Ratio — `sum(min(vehicle_speed, speed_limit)) / sum(speed_limit)`. How freely traffic flows. Returns 1.0 if no vehicles present.

**Phase State Ratios (indices 3-9):**
- Index 3: fraction red (r)
- Index 4: fraction yellow (y)
- Index 5: fraction minor green (g)
- Index 6: fraction priority green (G)
- Index 7: fraction u-turn (u)
- Index 8: fraction off-blinking (o)
- Index 9: fraction off (O)

This is the intersection-agnostic trick. Instead of encoding "GGrr" (specific to 4-lane intersections), we encode "50% green, 50% red" (works for ANY intersection type). A 2-lane and 8-lane intersection produce observations in the same format.

**Network Ranking (indices 10-13):**
- Index 10: Local Rank — congestion relative to immediate neighbors
- Index 11: Global Rank — congestion relative to entire network
- Index 12: Local Halt Rank — same for halted vehicles
- Index 13: Global Halt Rank — same for halted vehicles globally

Rankings make observations contextual: "I'm the most congested in my neighborhood" transfers across topologies. Computed from the TLS adjacency graph.

**Optional Time Encoding (indices 14-15):**
- sin(2π·t/H) and cos(2π·t/H) — smooth periodic encoding of timestep position
- Only active when `use_time_encoding=True`
- Expands observation to 16 features

### Why this design matters
Without intersection-agnostic observations, you cannot do federated averaging across different intersection types. A policy trained on 4-lane intersections would have incompatible weights with a policy trained on 2-lane intersections. Our ratio-based encoding makes all intersections produce the same format, enabling weight sharing.

### Who designed this
We designed the specific 14-feature combination. The concept of normalized observations exists in the literature. Phase ratios were explored by Hudson et al. (2022). Ranking features for cross-topology transfer are our design. The integration into a format specifically enabling federated aggregation across heterogeneous intersections is our contribution.

---

## THE REWARD FUNCTION

```
r_k = -(o_k + h_k)²
```

- o_k = lane occupancy at intersection k
- h_k = halted lane occupancy at intersection k
- Halted vehicles counted TWICE (once in o, once in h) — being stopped is penalized more than just being present
- Quadratic: small queue → small penalty, large queue → massive penalty. Encourages preventing queues, not tolerating them.
- Total network reward: r = Σ r_k across all intersections
- What does -7.4 mean? Less negative = less congestion. Going from -11 to -7.4 means the agent learned to keep queues significantly shorter.

### Why quadratic not linear
Linear treats going from 10% to 20% occupancy the same as 80% to 90%. Quadratic makes high congestion disproportionately expensive. The agent learns to prevent queues from building up rather than tolerating moderate congestion.

### Why not use waiting time directly as reward
The reward needs to be computed per-timestep during training. SUMO's waiting time metric is only available per-trip after vehicles arrive. Occupancy ratios are available at every timestep via TraCI and correlate strongly with waiting time.

---

## THE ACTION SPACE

Binary: Discrete(2)
- 0 = keep current phase
- 1 = advance to next phase in the cycle

Phase cycle is fixed per intersection (e.g., GGrr → yyrr → rrGG → rryy). Agent decides WHEN to advance, not WHERE to jump.

Timing constraints from US Federal Highway Administration:
- Minimum 4 seconds between changes (safety)
- Maximum 120 seconds before forced change (fairness)

### Why binary
A "choose which phase" action would be Discrete(4) for a 4-lane intersection and Discrete(8) for an 8-lane one. You can't share weights between them. Binary is the same Discrete(2) for every intersection type — essential for federated aggregation.

---

## FEDERATED AGGREGATION DETAILS

### Standard FedAvg (McMahan et al. 2017)
```
ω_global = (1/K) × Σ ω_k
```
Equal weights. Every intersection contributes the same.

### Our Reward-Weighted FedAvg
```
ω_global = Σ c_k × ω_k   where c_k ∝ normalized_reward_k
```
Better-performing intersections contribute more. Busy central intersection B1 with heavy traffic has more influence than quiet corner A0.

Available weight functions in `weight_aggr.py`:
- `naive`: equal 1/K (baseline FedAvg)
- `pos_reward`: weight by normalized positive reward (default, recommended)
- `neg_reward`: inverse reward (experimental)
- `traffic`: weight by vehicles served (experimental)

### FedProx Extension (Li et al. 2020)
```
L = L_ppo + (μ/2) × ||ω_k - ω_global||²
```
Proximal term prevents any local model from drifting too far from global consensus. μ controls strength:
- μ=0.0: standard FedAvg
- μ=0.01: light regularization (our best result: 11.0s vs 11.5s baseline)
- μ=0.1: strong regularization (11.2s)

Implemented in `fedprox_policy.py`. Key detail: `_fedprox_mu` must be set BEFORE `super().__init__()` because Ray's TorchPolicyV2 calls `loss()` during init.

### Aggregation frequency
Every episode (every 4000 timesteps). Over 200k timesteps = 50 aggregation rounds. Each round: 9 intersections × 2 directions × ~50KB = ~900KB. Total: ~45MB of weight exchange.

---

## THE 8 STANDARDIZATION LAYERS

This is how we guarantee fair comparison. When we say "FedRL achieves 11.5s and MARL achieves 13.9s," the difference can ONLY come from how policies are organized.

1. **Same road** — identical .net.xml file, never modified
2. **Same traffic** — identical VPLPH (360), same randomTrips parameters, same seeds
3. **Same observations** — same `get_observation()` function, same 14 features
4. **Same actions** — same Discrete(2), same timing constraints
5. **Same reward** — same `r = -(o+h)²`, same code path
6. **Same algorithm** — PPO, same hyperparameters (lr=5e-5, batch=4000, clip=0.3, 256×256 MLP)
7. **Same training budget** — 25 episodes, no early stopping
8. **Same evaluation** — 5 MC runs, seeds 42-46, metrics from SUMO tripinfo XML

### What could introduce bias and how we prevent it
- Different training length → prevented: all train exactly 25 episodes
- Different network architectures → prevented: all use same 256×256 MLP
- Different observation dimensions → prevented: all see 14 features
- Different reward scales → prevented: same formula, same alpha=1.0
- Different evaluation traffic → prevented: same MC seeds
- Weight initialization → controlled: RLlib default init with num_workers=0

---

## EXPERIMENTAL SETUP

### Topologies
- Grid 3×3: 9 signalized intersections, 24 controlled lanes
- Grid 5×5: 25 signalized intersections, heterogeneous lane counts
- Heterogeneous: center intersections have more lanes than edges
- Synthetic: standard in FedRL traffic literature (Ye et al. 2021, Hudson et al. 2022, Fu et al. 2025)

### Why synthetic grids
1. Controlled heterogeneity — we know exactly how intersections differ
2. Reproducibility — same network + same seed = same results anywhere
3. Standard practice — enables comparison with prior work
4. Real-world network integration attempted (Manhattan from OSM, Cologne from RESCO) but the observation layer needs adaptation for arbitrary intersection geometries. This is planned future work.

### Traffic demand
- 360 vehicles per lane per hour (VPLPH)
- Generated by SUMO `randomTrips.py`
- Period between departures: `3600 / (360 × n_lanes)` seconds
- `--fringe-factor 100`: vehicles enter/exit from network borders
- Vehicles drive straight through (no turns — SUMO limitation, simplifies evaluation)
- Same demand for all strategies

### Why 360 VPLPH
Standard in the literature. Produces moderate, non-trivial congestion. Lower = too easy (fixed-time works fine). Higher = gridlock. 360 is the sweet spot where adaptive control matters.

### Why no turns
Turning vehicles can get stuck in SUMO and introduce confounding factors. Restricting to straight-through traffic isolates signal timing effects from routing complications. Follows established methodology in the literature.

### PPO Hyperparameters
- Learning rate: 5 × 10⁻⁵
- SGD minibatch: 128
- Clip parameter: 0.3
- KL target: 0.3
- Train batch: 4000 timesteps
- Rollout fragment: 200
- GAE lambda: 1.0
- VF clip: 10
- Network: 2 × 256 hidden layers, ReLU
- Framework: Ray RLlib 2.x, old API stack mode

### Why PPO
Stable on-policy algorithm for discrete action spaces. Well-supported by Ray RLlib for multi-agent training. Used across the FedRL traffic literature. Not our contribution — a pragmatic choice.

### Why these specific hyperparameters
Drawn from the SEAL framework literature. We did not perform extensive hyperparameter tuning — all strategies use the same values so tuning would not affect the comparison. Tuning per-strategy would violate our standardization principle.

### Evaluation
- 5 Monte Carlo runs per config, seeds 42-46
- Metrics from SUMO's tripinfo XML (not computed by our code):
  - avg_waiting_time: time vehicles spend at speed 0
  - avg_travel_time: total time from departure to arrival
  - throughput: completed trips / total trips
- 25 episodes of training per config
- Communication cost: theoretical model based on actual data sizes

---

## RESULTS — MIDTERM

### Training Convergence (25 episodes)

| Strategy | Grid 3×3 Start→End | Improvement | Grid 5×5 Start→End | Improvement |
|----------|-------------------|-------------|-------------------|-------------|
| Federated | -11.0 → -7.4 | +33% | -22.5 → -16.5 | +27% |
| Centralized | -11.0 → -7.7 | +30% | -22.5 → -17.1 | +24% |
| Decentralized | -10.1 → -7.1 | +30% | -22.5 → -17.4 | +23% |

Federated converges to highest reward on both topologies.

### Evaluation (avg waiting time)

| Strategy | Grid 3×3 | Grid 5×5 |
|----------|--------:|--------:|
| Federated | 11.5s | 16.8s |
| Centralized (MARL) | 13.9s | 23.6s |
| Decentralized (SARL) | 10.4s | 17.4s |
| Fixed-Time | 76.9s | 70.6s |

### Evaluation (avg travel time)

| Strategy | Grid 3×3 | Grid 5×5 |
|----------|--------:|--------:|
| Federated | 56.4s | 83.5s |
| Centralized | 62.0s | 91.5s |
| Decentralized | 54.8s | 83.7s |
| Fixed-Time | 117.3s | 123.6s |

### Communication Cost (theoretical, Grid 3×3, 200k timesteps)

| Strategy | Total | Reduction vs Centralized |
|----------|------:|------------------------:|
| Centralized | 174.6 MB | — |
| Federated | 102.6 MB | 41% |
| Decentralized | 57.6 MB | 67% |

### FedProx Ablation (Grid 3×3)

| μ value | Avg Wait | Change |
|---------|--------:|-------:|
| 0.0 (FedAvg) | 11.5s | — |
| 0.01 | 11.0s | -4.3% |
| 0.1 | 11.2s | -2.6% |

### Time-of-Day Ablation (Grid 3×3)

| Config | Avg Wait | Training Reward Start→End |
|--------|--------:|------------------------:|
| Fixed Demand (360 VPLPH) | 11.5s | -11.0 → -7.4 |
| ToD + Encoding (200-700 VPLPH) | 12.4s | -38.4 → -15.1 |

---

## COMMON QUESTIONS AND ANSWERS

### "What's novel / what's yours?"
The components are standard (PPO, SUMO, FedAvg). What we built is: (1) the 14-feature intersection-agnostic observation space with ranking features enabling cross-topology federated aggregation, (2) reward-weighted averaging where better agents contribute more, (3) the unified evaluation framework controlling 8 layers of variables, (4) the automated experiment pipeline with Monte Carlo evaluation. The integration of these into a working benchmarking system is the contribution.

### "Why not just use MARL? It seems simpler."
MARL requires constant communication between all agents and a central coordinator. For 25 intersections streaming 64-byte observations every timestep over 200k steps, that's ~485 MB. FedRL sends 50KB weights periodically — 41% less total. In a real deployment with traffic lights on wireless networks, that bandwidth difference matters.

### "Decentralized beats Federated on Grid 3×3. Doesn't that undermine your argument?"
No — it shows exactly what a benchmark should show. On a small 9-intersection grid, one shared policy can capture the full dynamics. Federated's advantage emerges at scale: on Grid 5×5, Federated leads (16.8s vs 17.4s). The benchmark reveals WHEN each strategy wins, not that one is universally best.

### "Is 25 episodes enough?"
The reward curves plateau around episode 15-20, indicating convergence. Our earlier 50-episode run on Grid 5×5 showed the same plateau. For the final evaluation we plan longer training and more MC runs with significance testing.

### "5 Monte Carlo runs — is that statistically sufficient?"
For the midterm, 5 runs gives variance estimates and error bars. For the final: 10 runs with Wilcoxon signed-rank tests. 5 is sufficient to show consistent trends.

### "Why synthetic grids? Why not real-world networks?"
Three reasons: (1) controlled heterogeneity — we design exactly how intersections differ, (2) reproducibility — anyone can replicate, (3) standard practice in the literature. We attempted Manhattan (OSM) and Cologne (RESCO benchmark) integration — the network loads but the observation layer needs adaptation for arbitrary intersection geometries. This is planned for the final evaluation.

### "Why fixed-time baseline and not something stronger?"
Fixed-time is the real-world baseline — it's what most traffic lights actually use today. We also implemented max-pressure (an adaptive heuristic) but it performed worse than fixed-time in our grids, likely because it's a greedy local optimizer that doesn't account for downstream effects. We removed it from the presentation to keep the comparison clean.

### "Your vehicles don't turn. Isn't that unrealistic?"
Yes, it's a simplification that isolates signal timing effects from routing complexity. It follows the methodology in the SEAL and FedLight papers. Supporting turns requires careful route generation to prevent SUMO deadlocks and is planned future work.

### "How would you deploy this in a real city?"
Each traffic light gets edge compute (Raspberry Pi / Jetson Nano) running the trained policy. Inference is lightweight — 256×256 MLP on 14 inputs. Sensors (loop detectors or cameras) provide the three traffic flow observations. Training happens offline in SUMO. Periodic federated updates could happen overnight over existing V2X infrastructure.

### "What privacy guarantee does federated provide?"
Raw observation data never leaves the intersection. Only model weights (~50KB of neural network parameters) are shared. This is the standard FL privacy notion from McMahan et al. 2017. We do NOT claim differential privacy — that would require noise injection and formal (ε, δ) analysis, which is out of scope.

### "Could a malicious intersection poison the model?"
Yes, byzantine attacks are a known FL vulnerability. Our reward-weighted averaging provides natural defense — a malicious agent with poor performance gets low weight. Formal robustness against adversarial agents would require median-based aggregation or anomaly detection.

### "Why PPO and not DQN/A2C/SAC?"
PPO is stable for discrete actions and well-supported by Ray RLlib for multi-agent setups. Bao et al. (2023) use DQN, FedLight uses A2C — both work. We chose PPO for pragmatic reasons and held it constant across all strategies. The benchmark evaluates training strategies, not RL algorithms.

### "What are the PPO hyperparameters? Did you tune them?"
lr=5e-5, minibatch=128, clip=0.3, KL=0.3, batch=4000, rollout=200, GAE=1.0, VF clip=10. Drawn from the SEAL literature. We did NOT tune per-strategy — that would violate standardization. All strategies use identical settings.

### "The reward function seems arbitrary. Why that specific formula?"
It comes from the traffic RL literature. Occupancy + halted occupancy captures both presence and congestion. The quadratic makes large queues disproportionately expensive. Alternative formulations (delay, queue length, throughput) exist but we needed a per-timestep signal computable from TraCI. The evaluation uses SUMO's own travel/waiting time metrics as the ground truth.

### "What does reward-weighted averaging actually do differently?"
Standard FedAvg: every intersection contributes 1/K. Ours: a central intersection handling 500 vehicles with reward -5 contributes more than a corner intersection handling 50 vehicles with reward -2. The busy intersection's complex traffic patterns have more influence on the shared model.

### "FedProx only improves 4%. Is that worth it?"
On a small grid with moderate heterogeneity, 4% is modest. FedProx is designed for HIGHLY heterogeneous settings. On larger networks or real-world intersections with very different traffic patterns, the improvement should be larger. The preliminary result shows the mechanism works; the impact depends on the degree of heterogeneity in the deployment scenario.

### "How does time-of-day training work?"
Each episode randomly samples a demand period: AM rush (500-700 VPLPH), midday (200-300), or PM rush (400-600). Sin/cos time features are appended to observations so the agent knows where in the episode it is. The agent must generalize across conditions rather than memorize one demand level. Training starts much harder (reward -38 vs -11) but converges to similar evaluation performance (12.4s vs 11.5s).

### "What's the communication cost model based on?"
Actual data sizes: observation vectors = 64 bytes (16 floats × 4 bytes), actions = 1 byte, V2I messages = 32 bytes, policy weights = ~50KB. Timing follows the actual training protocol. Centralized streams obs+actions every timestep. Decentralized has V2I only. Federated has V2I + weight exchange every 4000 steps. The 41% reduction ratio is robust even if absolute bytes differ with compression.

### "What would you do differently with unlimited compute?"
Train 200+ episodes on Grid 3×3, 5×5, and 7×7. Run 50 MC seeds. Test on real-world RESCO networks. Sweep PPO hyperparameters. Test cooperative reward with alpha from 0.0 to 1.0. Compare FedAvg vs FedProx vs FedFomo vs FedCluster aggregation. Run low/medium/high demand scenarios. Full Wilcoxon significance testing.

### "What makes this publishable?"
The full second-half ablation studies with significance tests, results on multiple demand scenarios, the completed trade-off matrix, and ideally one real-world network. The midterm demonstrates the platform works. The final results need to show statistically significant differences across conditions and provide actionable guidance on strategy selection.

### "How does this relate to IoT?"
Traffic lights are distributed edge devices with limited bandwidth on wireless networks. The core challenge is training a network of IoT devices efficiently without centralizing all data. Federated learning was designed for exactly this deployment model. The communication cost analysis directly addresses IoT constraints.

### "What existing benchmarks are there for this?"
RESCO (Ault & Sharon, NeurIPS 2021) benchmarks single-agent and multi-agent RL for traffic signals. LibSignal (Mei et al., 2024) provides cross-simulator comparison. Neither benchmarks federated approaches. That's the gap we fill.

---

## LITERATURE CONTEXT

Fewer than 10 papers worldwide on federated RL for traffic signal control:

| Paper | Year | Venue | Unique Contribution |
|-------|------|-------|---------------------|
| FedLight (Ye et al.) | 2021 | DAC | First FedRL for traffic signals, A2C |
| SEAL (Hudson et al.) | 2022 | SMARTCOMP | Intersection-agnostic representation |
| Bao et al. | 2023 | Scientific Reports | Partial model aggregation (split DQN) |
| Fed-PPO (Li et al.) | 2025 | Scientific Reports | Soft-update aggregation with tau |
| HFRL (Fu et al.) | 2025 | arXiv | Hierarchical clustering of intersections |
| FitLight (Ye et al.) | 2025 | arXiv | Imitation learning bootstrap + model pruning |

Additional references:
- McMahan et al. (2017) — FedAvg, AISTATS
- Li et al. (2020) — FedProx, MLSys
- Ault & Sharon (2021) — RESCO benchmark, NeurIPS
- Varaiya (2013) — Max-pressure control, IEEE TAC
- Schulman et al. (2017) — PPO, arXiv

### Our positioning
We are not one of these papers. We are building what none of them built: the standardized evaluation framework. Each paper compares against its own baselines with its own setup. We compare all strategies under identical conditions. That's the contribution.

---

## EXTENSIONS — IMPLEMENTED, PENDING FULL EVALUATION

### FedProx
- What: proximal regularization preventing local drift from global model
- Formula: `L = L_ppo + (μ/2) × ||w - w_global||²`
- Parameter: μ ∈ {0.0, 0.01, 0.1}
- Preliminary: μ=0.01 gives 4% improvement on Grid 3×3
- Code: `seal/trainer/fedprox_policy.py`

### Cooperative Reward Shaping
- What: agents care about neighbors' congestion, not just their own
- Formula: `r = α × r_local + (1-α) × mean(r_neighbors)`
- Parameter: α ∈ {1.0, 0.5, 0.1} (1.0 = selfish default)
- Neighbors: from TLS adjacency graph in network file
- Code: `seal/sumo/env.py`

### Time-of-Day Demand
- What: variable traffic instead of constant 360 VPLPH
- Periods: AM rush (500-700), midday (200-300), PM rush (400-600)
- Observation: sin/cos time encoding at indices 14-15
- Preliminary: converges despite harder task, 12.4s vs 11.5s eval
- Code: `seal/sumo/abstract_env.py`

---

## SECOND HALF ACTION PLAN

### Priority 1: Multi-Demand Evaluation
| Setting | VPLPH | Purpose |
|---------|------:|---------|
| Low | 150 | Does RL matter when traffic is light? |
| Medium (done) | 360 | Standard conditions |
| High | 600 | Stress test — graceful degradation? |
| Variable (ToD) | 200-700 | Adaptability |

32 configs total (4 settings × 4 strategies × 2 topologies).

### Priority 2: Trade-off Matrix
Define when each strategy wins/loses based on evidence:

| Strategy | Best When | Worst When | Communication | Scalability |
|----------|-----------|------------|:-------------:|:-----------:|
| SARL | Small networks | Large, bandwidth-limited | Highest | Poor |
| MARL | Need specialization | Comm-constrained | High | Moderate |
| FedRL | Large networks, privacy | Very small networks | Low | Good |
| Fixed-Time | Near-zero traffic | Any real demand | None | N/A |

### Priority 3: Aggregation Variants
- Naive FedAvg vs Reward-Weighted FedAvg vs FedProx (μ=0.01, 0.1)

### Priority 4: Statistical Rigor
- 10 MC runs per config
- Wilcoxon signed-rank tests
- 95% confidence intervals
- LaTeX tables with p-values

---

## QUICK TECHNICAL REFERENCE

### How to run experiments
```bash
cd BackEnd
python scripts/run_all_training.py --episodes 25 --eval-runs 5          # full run
python scripts/run_all_training.py --episodes 25 --eval-runs 5 --resume # resume
python scripts/generate_all_figures.py                                   # figures
```

### Key file locations
| What | Where |
|------|-------|
| FedRL trainer | `seal/trainer/fed_agent.py` |
| MARL trainer | `seal/trainer/multi_agent.py` |
| SARL trainer | `seal/trainer/single_agent.py` |
| Observations | `seal/sumo/kernel/trafficlight/light.py` |
| Reward function | `seal/sumo/env.py` |
| Feature indices | `seal/sumo/config.py` |
| Trainer factory | `api/training_runner.py` |
| MC evaluation | `api/evaluation/monte_carlo.py` |
| Experiment runner | `scripts/run_all_training.py` |
| Figure generation | `scripts/generate_all_figures.py` |
| Results JSON | `results/campaigns/training-curves/results.json` |
| Figures | `results/figures/` |

### Stack
SUMO 1.26.0, Ray RLlib 2.x (old API stack), PyTorch 2.x, Python 3.12, FastAPI, React + Vite, Windows 11
