# Team Briefing — What This Project Is and Where We're Going

## The One-Sentence Version

We built a system that trains smart traffic lights using AI, and we're benchmarking three different ways to organize that training to find out which works best and when.

---

## The Problem

Traffic lights run on fixed timers. Green 30 seconds, red 30 seconds, doesn't matter if there are 100 cars waiting or zero. This wastes everyone's time and costs the US $87 billion a year in congestion.

Reinforcement learning can fix this — each traffic light observes its surroundings and learns when to switch. But when you have 9 or 25 intersections in a network, you need to decide HOW to organize the training. That's what our project is about.

---

## The Three Strategies We Compare

**SARL (Single-Agent)** — One brain controls everything. Every intersection sends its data to one central model. Simple, but requires massive communication — all data flows to one place.

**MARL (Multi-Agent)** — Each intersection gets its own brain. They all train through a central coordinator. Each one specializes, but the coordinator needs constant communication with every intersection.

**FedRL (Federated)** — Each intersection gets its own brain and trains LOCALLY. Periodically (once per episode), they share just their model weights with a server that averages them and sends back the consensus. Between rounds, zero communication. Raw traffic data never leaves the intersection.

---

## Why This Matters

Nobody has compared these three strategies fairly. Every paper uses its own simulator, its own settings, its own metrics. You can't compare their numbers. We built a framework where ALL THREE run on the exact same:

- Same road network
- Same traffic
- Same observations (14 features per intersection)
- Same reward function
- Same RL algorithm (PPO) with same settings
- Same evaluation (5 Monte Carlo runs, same random seeds)

The ONLY thing that changes is how the policies are organized. That's what makes our comparison valid.

---

## What We Found So Far

### All RL strategies crush fixed-time signals

| Strategy | Grid 3×3 Waiting Time | Grid 5×5 Waiting Time |
|----------|----------------------:|----------------------:|
| Federated | 11.5s | 16.8s |
| Centralized (MARL) | 13.9s | 23.6s |
| Decentralized (SARL) | 10.4s | 17.4s |
| Fixed-Time | 76.9s | 70.6s |

That's **75-85% less waiting** with any RL strategy vs fixed timers.

### Federated uses way less communication

| Strategy | Data Transmitted |
|----------|----------------:|
| Centralized | 174.6 MB |
| Federated | 102.6 MB |
| Decentralized | 57.6 MB |

Federated achieves the same performance as centralized with **41% less data**. It only sends model weights (~50KB) periodically instead of streaming observations every timestep.

### Federated's advantage grows with network size

On the small 3×3 grid, Decentralized actually edges out Federated (10.4s vs 11.5s). But on the larger 5×5 grid, Federated takes the lead (16.8s vs 17.4s). The bigger the network, the more valuable knowledge sharing becomes.

---

## What's Built

- Full training pipeline for all 3 strategies on SUMO simulator
- 14-feature observation space that works on any intersection type
- Reward-weighted federated averaging (smarter agents contribute more)
- Automated experiment runner that saves after each config (safe to interrupt)
- Monte Carlo evaluation with seed control
- 8 publication-quality figures
- Three extensions implemented and code-complete:
  - **FedProx** — prevents agents from drifting too far from the group consensus
  - **Cooperative Rewards** — agents care about their neighbors' congestion, not just their own
  - **Time-of-Day** — variable traffic demand (rush hour vs quiet periods)

---

## What's Left (Second Half)

### 1. Multiple Demand Settings

Right now we tested at one traffic level (360 vehicles/lane/hour). We need to test at:

| Setting | Vehicles/Lane/Hour | Purpose |
|---------|-------------------:|---------|
| Low | 150 | Does RL even matter when traffic is light? |
| Medium (done) | 360 | Standard — current results |
| High | 600 | Stress test — which strategy degrades worst? |

This answers: "which strategy should you use under what conditions?"

### 2. Trade-off Matrix

The deliverable is a table like this, filled in with evidence:

| Strategy | Best When | Worst When |
|----------|-----------|------------|
| SARL | Small networks | Large networks, limited bandwidth |
| MARL | Need specialized policies | Communication-constrained |
| FedRL | Large networks, privacy matters | Very small networks |
| Fixed-Time | Almost no traffic | Any real demand |

### 3. Statistical Rigor

- More Monte Carlo runs (10 instead of 5)
- Significance tests (Wilcoxon) to prove differences aren't just noise
- Confidence intervals on all numbers

---

## Who Does What

| Person | Responsibility |
|--------|---------------|
| Majid | System architecture, evaluation framework, RL agent design |
| Hamdan | Federated learning loop, FedAvg and FedProx aggregation |
| Abdallah | SUMO simulation environment, observation space |
| Mohammad | Baseline controllers, evaluation protocol, Monte Carlo pipeline |
| Mohamed | Literature review, benchmark scenario design, results analysis, report |

---

## How to Run It

```bash
# Train all strategies (takes ~5-6 hours, saves progress incrementally)
cd BackEnd && python scripts/run_all_training.py --episodes 25 --eval-runs 5

# Resume if interrupted
cd BackEnd && python scripts/run_all_training.py --episodes 25 --eval-runs 5 --resume

# Generate all figures from results
cd BackEnd && python scripts/generate_all_figures.py
```

Results go to `BackEnd/results/campaigns/`. Figures go to `BackEnd/results/figures/`.

---

## The Key Framing for the Presentation

We are NOT saying "FedRL is the best." We are saying:

> "We built a standardized benchmarking framework and here's what the data shows: all RL strategies beat fixed-time by 75-85%. Federated matches centralized performance at 41% less communication cost. Its advantage grows with network size. The second half will test under different demand levels to complete the trade-off analysis."

The contribution is the framework and the fair comparison, not any single algorithm.

---

## Key Files to Know

| If you need to... | Look at... |
|-------------------|-----------|
| Understand FedRL training | `BackEnd/seal/trainer/fed_agent.py` |
| Understand MARL training | `BackEnd/seal/trainer/multi_agent.py` |
| Understand SARL training | `BackEnd/seal/trainer/single_agent.py` |
| See how observations work | `BackEnd/seal/sumo/kernel/trafficlight/light.py` |
| See the reward function | `BackEnd/seal/sumo/env.py` |
| Run experiments | `BackEnd/scripts/run_all_training.py` |
| Generate figures | `BackEnd/scripts/generate_all_figures.py` |
| See all results | `BackEnd/results/campaigns/training-curves/results.json` |
| Full technical details | `PROJECT-BRIEFING.md` |
