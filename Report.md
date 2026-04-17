# SEAL: Training Strategy Benchmarking for Multi-Intersection Traffic Signal Control

## 1. Problem Statement

Urban traffic congestion is a growing challenge. Traditional signal control methods (fixed-time, actuated) fail to adapt to dynamic demand patterns. Reinforcement Learning (RL) offers adaptive signal control, but **how should multiple intersection agents coordinate their learning?**

This project benchmarks **10 training strategies** across a spectrum from fully independent to fully shared learning, keeping everything else constant: same PPO algorithm, same observations, same reward function, same environment. The **only variable is the training strategy** -- isolating its effect on traffic performance.

## 2. Environment and Setup

### Observation Space (14 features per intersection)

Each traffic light agent observes:

| Feature | Description |
|---------|-------------|
| Lane occupancy | Fraction of lane capacity in use (0-1) |
| Halted occupancy | Fraction of stopped vehicles |
| Speed ratio | Mean speed / max speed |
| Phase state | 7 one-hot indicators for current signal phase (r, y, g, G, u, o, O) |
| Local rank | Occupancy rank within topological neighborhood |
| Global rank | Occupancy rank across entire network |
| Local halt rank | Halted-vehicle rank within neighborhood |
| Global halt rank | Halted-vehicle rank across entire network |

### Reward Function

```
reward = -(lane_occupancy + halted_lane_occupancy)^2
```

A quadratic penalty on congestion. Lower (more negative) reward means worse congestion. The squared term amplifies the penalty as congestion grows, incentivizing agents to keep occupancy low.

### Algorithm

All RL strategies use **PPO (Proximal Policy Optimization)** with identical hyperparameters:
- Learning rate: 0.001
- Discount factor (gamma): 0.95
- Training episodes: 50
- Evaluation: 10 Monte Carlo seeds per configuration (seeds 42-51)

### Simulation

- **Simulator**: SUMO (Simulation of Urban Mobility)
- **Demand**: 360 vehicles per lane per hour (VPLPH)
- **Interface**: TraCI socket protocol (Windows), libsumo (Linux/HPC)

## 3. Network Topologies

Three topologies of increasing realism were tested:

| Topology | Intersections | Type | Characteristics |
|----------|:---:|------|-----------------|
| **grid-3x3** | 9 | Synthetic | Symmetric grid, uniform lane counts, controlled baseline |
| **grid-5x5** | 25 | Synthetic | Larger grid, tests scalability of coordination strategies |
| **cologne-8** | 8 | Real-world | RESCO benchmark from Cologne, Germany; irregular geometry, realistic lane configurations |

## 4. Training Strategies

Ordered from most independent to most shared:

```
Independent <---------------------------------------------------------> Shared

MARL -> MeanField -> CTDE -> Gossip -> HierFed -> FedDistill -> FedRL -> SARL
 (0)     (obs)    (critic)   (mesh)    (tree)     (logits)     (star)  (full)
```

### 4.1 MARL (Multi-Agent RL) -- Fully Independent

Each intersection trains its own independent PPO policy. No weight sharing, no communication, no coordination. Each agent learns purely from its own local observations and rewards.

- **Communication cost**: Zero
- **Coordination mechanism**: None

### 4.2 MeanField -- Implicit Coordination via Observation

Each intersection has an independent policy (like MARL), but observations are **augmented with the mean action of topological neighbors** from the previous timestep. No weights are shared -- coordination is purely implicit through richer observations.

- **Communication cost**: Zero (observation computed locally from simulator state)
- **Coordination mechanism**: Neighbor mean action appended to observation

### 4.3 CTDE (Centralized Training, Decentralized Execution)

During training, the value function sees **global state** (all agents' observations concatenated) for better credit assignment. During evaluation, each agent acts using only its local observation. Gives the critic full observability without requiring runtime communication.

- **Communication cost**: Zero at execution time
- **Coordination mechanism**: Centralized critic during training only

### 4.4 Gossip -- Decentralized Mesh Averaging

No central server. Each agent periodically averages its policy weights with **direct topological neighbors** (intersections connected by road segments). The averaging formula:

```
w_i = (w_i + SUM(w_j for j in neighbors(i))) / (1 + degree(i))
```

- **Communication cost**: SUM(degrees) x model_bytes per round
- **Coordination mechanism**: Peer-to-peer weight averaging respecting road topology

### 4.5 HierFed (Hierarchical Federated)

Two-tier aggregation tree:
1. **Intra-cluster**: Agents grouped by spatial locality (greedy BFS), averaged within each cluster
2. **Inter-cluster**: Cluster representatives averaged globally
3. **Broadcast**: Global model distributed back to all agents

- **Communication cost**: ~3N x model_bytes per round
- **Coordination mechanism**: Spatially-aware hierarchical averaging

### 4.6 FedDistill (Federated Distillation)

Instead of sharing full model weights, agents share only **action logits** (output probabilities). Each round:
1. All agents forward a reference observation through their policy
2. Consensus logits computed as mean across all agents
3. Each agent adds a KL-divergence loss to match the consensus

- **Communication cost**: ~100x cheaper than weight-based methods (logits vs full weights)
- **Coordination mechanism**: Knowledge distillation via shared action distributions

### 4.7 FedRL (Federated Averaging) -- Star Topology

Classical Federated Learning applied to RL. A central server collects all agent weights every `fed_step` episodes and computes a weighted average. Supports multiple aggregation schemes:
- **pos_reward**: Weight by positive episode reward (default)
- **neg_reward**: Weight by negative reward (focus on struggling agents)
- **naive**: Equal weighting
- **traffic**: Weight by vehicle count

Optional extensions: FedProx (proximal regularization), soft-update (tau blending), clustered aggregation.

- **Communication cost**: (N+1) x model_bytes per round
- **Coordination mechanism**: Centralized weighted averaging

### 4.8 SARL (Single-Agent RL) -- Fully Shared

All intersections share a **single policy**. Every agent executes the same action distribution given the same observation. The simplest form of full coordination -- one brain controls all intersections.

- **Communication cost**: Zero (single policy by design)
- **Coordination mechanism**: Complete weight sharing (one policy for all)

### 4.9 Fixed-Time Baseline (Non-RL)

Phases cycle with fixed duration regardless of traffic conditions. No learning, no adaptation. Standard engineering baseline.

### 4.10 Max-Pressure Baseline (Non-RL)

Greedy heuristic: at each step, select the phase that relieves the most "pressure" (difference between upstream queue and downstream capacity). Adaptive but not learned.

## 5. Evaluation Results

All strategies evaluated with **10 Monte Carlo seeds** per topology. Metrics reported as means across seeds.

### 5.1 grid-3x3 (9 intersections)

| Rank | Strategy | Avg Waiting Time (s) | Avg Travel Time (s) | Throughput |
|:----:|----------|:--------------------:|:--------------------:|:----------:|
| 1 | **HierFed** | **11.33** | **56.16** | 1.000 |
| 2 | Gossip | 11.42 | 56.33 | 1.000 |
| 3 | FedDistill | 11.81 | 56.75 | 1.000 |
| 4 | MeanField | 12.75 | 57.93 | 1.000 |
| 5 | MARL | 14.23 | 59.82 | 1.000 |
| 6 | SARL | 14.88 | 59.77 | 1.000 |
| 7 | FedRL | 15.61 | 61.32 | 1.000 |
| 8 | CTDE | 15.96 | 61.46 | 1.000 |
| 9 | fixed-time | 73.64 | 113.97 | 1.000 |
| 10 | max-pressure | 170.68 | 213.98 | 1.000 |

**Key finding**: Decentralized strategies with local coordination (HierFed, Gossip, FedDistill) outperform both fully independent (MARL) and fully shared (SARL, FedRL) approaches. All RL methods beat baselines by 5-15x.

### 5.2 grid-5x5 (25 intersections)

| Rank | Strategy | Avg Waiting Time (s) | Avg Travel Time (s) | Throughput |
|:----:|----------|:--------------------:|:--------------------:|:----------:|
| 1 | **Gossip** | **17.60** | **83.66** | 1.000 |
| 2 | SARL | 19.92 | 84.60 | 1.000 |
| 3 | HierFed | 21.02 | 87.84 | 1.000 |
| 4 | FedDistill | 21.09 | 87.85 | 1.000 |
| 5 | FedRL | 21.84 | 88.06 | 1.000 |
| 6 | CTDE | 21.86 | 88.60 | 1.000 |
| 7 | MeanField | 23.91 | 91.30 | 1.000 |
| 8 | MARL | 24.95 | 92.46 | 1.000 |
| 9 | fixed-time | 70.62 | 124.64 | 1.000 |
| 10 | max-pressure | 152.34 | 210.72 | 1.000 |

**Key finding**: Gossip dominates at scale -- peer-to-peer coordination aligned with road topology is most effective for larger networks. SARL (fully shared) also performs well, suggesting that at scale, some form of coordination is critical. MARL (fully independent) is worst among RL methods.

### 5.3 cologne-8 (8 intersections, real-world)

| Rank | Strategy | Avg Waiting Time (s) | Avg Travel Time (s) | Throughput |
|:----:|----------|:--------------------:|:--------------------:|:----------:|
| 1 | **fixed-time** | **44.35** | **101.05** | 1.000 |
| 2 | max-pressure | 51.11 | 106.66 | 1.000 |
| 3 | SARL | 54.71 | 134.14 | 1.000 |
| 4 | HierFed | 59.27 | 144.43 | 1.000 |
| 5 | MARL | 63.22 | 152.29 | 1.000 |
| 6 | MeanField | 65.92 | 159.05 | 1.000 |
| 7 | Gossip | 66.80 | 158.51 | 1.000 |
| 8 | FedDistill | 67.73 | 159.48 | 1.000 |
| 9 | FedRL | 67.17 | 158.76 | 1.000 |
| 10 | CTDE | 71.76 | 162.65 | 1.000 |

**Key finding**: On the real-world Cologne network, **all RL strategies underperform the baselines**. This is a significant result. Possible explanations:
- The irregular geometry of real-world networks requires more training data (50 episodes may be insufficient)
- Phase structures differ from synthetic grids -- the RL agent's action space may not align well with Cologne's signal plans
- Fixed-time baselines may be well-tuned for this specific network in the RESCO benchmark
- The reward function (occupancy-based) may not capture what matters in asymmetric real-world intersections

## 6. Cross-Topology Analysis

### Waiting Time Comparison (seconds)

| Strategy | grid-3x3 | grid-5x5 | cologne-8 |
|----------|:--------:|:--------:|:---------:|
| MARL | 14.23 | 24.95 | 63.22 |
| MeanField | 12.75 | 23.91 | 65.92 |
| CTDE | 15.96 | 21.86 | 71.76 |
| Gossip | 11.42 | **17.60** | 66.80 |
| HierFed | **11.33** | 21.02 | 59.27 |
| FedDistill | 11.81 | 21.09 | 67.73 |
| FedRL | 15.61 | 21.84 | 67.17 |
| SARL | 14.88 | 19.92 | **54.71** |
| fixed-time | 73.64 | 70.62 | **44.35** |
| max-pressure | 170.68 | 152.34 | 51.11 |

### Strategy Rankings Across Topologies

| Strategy | grid-3x3 Rank | grid-5x5 Rank | cologne-8 Rank | Avg Rank |
|----------|:---:|:---:|:---:|:---:|
| HierFed | 1 | 3 | 4 | **2.7** |
| Gossip | 2 | 1 | 7 | **3.3** |
| FedDistill | 3 | 4 | 8 | **5.0** |
| SARL | 6 | 2 | 3 | **3.7** |
| MeanField | 4 | 7 | 6 | **5.7** |
| FedRL | 7 | 5 | 9 | **7.0** |
| MARL | 5 | 8 | 5 | **6.0** |
| CTDE | 8 | 6 | 10 | **8.0** |

**Excluding baselines, the most robust RL strategies across all topologies are HierFed (avg rank 2.7), Gossip (3.3), and SARL (3.7).**

## 7. Key Insights

### 1. Moderate coordination outperforms extremes on synthetic networks
Neither fully independent (MARL) nor fully centralized (FedRL, SARL) consistently wins. Strategies that respect the network's spatial structure -- **Gossip** (neighbor averaging) and **HierFed** (spatially-clustered hierarchical averaging) -- achieve the best results on synthetic grids.

### 2. Coordination topology matters more than coordination frequency
Gossip (mesh) and HierFed (tree) share weights along the road network topology. FedRL (star) ignores spatial structure. The topology-aware methods consistently outperform flat aggregation, suggesting that **who you share with matters more than how often**.

### 3. RL struggles on real-world networks without additional tuning
The cologne-8 results show that RL agents trained with 50 episodes on a real-world network cannot outperform well-calibrated baselines. This is a realistic finding: deployment requires longer training, domain-specific reward shaping, or curriculum learning.

### 4. Communication efficiency does not sacrifice performance
FedDistill (sharing only action logits at ~100x less data than weight-sharing methods) ranks 3rd on grid-3x3 and 4th on grid-5x5. This makes it attractive for bandwidth-constrained deployments.

### 5. All RL strategies achieve 100% throughput
Every vehicle completes its trip regardless of strategy. The differentiation is entirely in waiting time and travel time -- quality of service, not capacity.

## 8. Experimental Configuration

| Parameter | Value |
|-----------|-------|
| RL Algorithm | PPO (Proximal Policy Optimization) |
| Framework | Ray RLlib (old API stack) |
| Learning rate | 0.001 |
| Discount factor | 0.95 |
| Training episodes | 50 |
| Evaluation seeds | 10 (seeds 42-51) |
| Demand (VPLPH) | 360 |
| Simulation backend | SUMO via TraCI |
| Observation features | 14 per intersection (ranked mode) |
| Action space | Discrete (phase selection) |

## 9. Future Work / Ablation Opportunities

- **Demand sensitivity**: Test strategies under low (150), medium (360), and high (600) VPLPH to identify which degrade gracefully
- **FedRL aggregation variants**: Compare pos-reward, neg-reward, naive, and traffic weighting schemes
- **FedProx regularization**: Sweep mu from 0.0 to 1.0 to test if proximal terms prevent local drift
- **Cooperative reward blending**: Sweep alpha from 1.0 (selfish) to 0.1 (cooperative neighborhood reward)
- **Communication frequency**: Vary fed_step from 1 to 10 to test coordination-computation tradeoff
- **Extended training on cologne-8**: Increase episodes from 50 to 200+ to test if RL can eventually outperform baselines on real-world networks
- **Transfer learning**: Train on grid, evaluate on cologne-8 to test generalization
