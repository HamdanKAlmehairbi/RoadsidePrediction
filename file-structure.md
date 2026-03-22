# Project File Structure

```
1-Sumo-FedRL/
├── CLAUDE.md                       # Agent instructions — hard rules, architecture, known patterns
├── BUILD-RULE.md                   # Workflow rules — plan mode, verification, lessons
├── PROJECT-PLAN.md                 # Full spec — API contract, page designs, acceptance criteria
├── README.md                       # Project overview, quick start, API summary
├── SKILL.md                        # /build-with-agent-team skill definition
├── file-structure.md               # This file — live project tree
├── tasks/
│   ├── todo.md                     # Current status and bug fix history
│   ├── bugfix-plan.md              # Active bugfix task for agent team
│   └── lessons.md                  # Lessons learned from past bugs and corrections
├── BackEnd/                        # FastAPI backend — SEAL Dashboard API
│   ├── seal/                       # Copied from SUMO-FedRL-main/seal/
│   ├── configs/SMARTCOMP/          # Copied .net.xml files (3x3, 5x5, 7x7)
│   ├── example_weights/ICCPS/Final/  # Copied .pkl weight files (FedRL, MARL, SARL)
│   ├── netfiles.py                 # Copied from SUMO-FedRL-main/netfiles.py
│   ├── requirements.txt            # fastapi, uvicorn, websockets
│   ├── README.md                   # Setup instructions, SUMO install guide, API docs
│   ├── trained_weights/             # Output dir for newly trained weights (auto-created)
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app, CORS, router mounting
│       ├── jobs.py                 # In-memory job store (job_id -> status/queue/results)
│       ├── training_runner.py      # SEAL trainer wrapper: Ray singleton, per-episode streaming, topology mapping
│       ├── baselines/
│       │   ├── __init__.py
│       │   └── max_pressure.py     # Max-pressure baseline: selects phase with highest upstream-downstream queue diff
│       ├── evaluation/
│       │   ├── __init__.py         # Package init for evaluation module
│       │   ├── runner.py           # run_trial(): single evaluation episode for any trainer/topology; TrialResult dataclass; resolve_weights_path()
│       │   ├── baselines.py        # run_fixed_time_trial(), run_max_pressure_trial(), run_rl_trial() wrappers
│       │   ├── metrics.py          # compute_trial_metrics(), TripinfoMetrics, TrialMetrics, metrics_to_dict()
│       │   ├── transfer.py         # build_transfer_matrix(), compute_transfer_gap(), transfer_matrix_to_dict()
│       │   └── monte_carlo.py      # MCConfig, MCAggregatedResult, run_monte_carlo(), run_full_campaign(), campaign_to_dict()
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── networks.py         # GET /api/networks, GET /api/network/{topology}
│       │   ├── weights.py          # GET /api/weights (scans example_weights/ + trained_weights/)
│       │   ├── simulate.py         # POST /api/simulate (RL policy / fixed-timing / max-pressure via SUMO)
│       │   ├── train.py            # POST /api/train (real SEAL training with mock fallback)
│       │   └── results.py          # GET /api/results/{job_id}
│       └── ws/
│           ├── __init__.py
│           ├── simulate.py         # WS /ws/simulate/{job_id} — streams vehicle+TLS frames
│           └── train.py            # WS /ws/train/{job_id} — streams episode rewards
├── FrontEnd/                       # React dashboard — copied from LovableOutput, wired to BackEnd API
│   ├── index.html
│   ├── package.json                # React 18, Vite, shadcn/ui, Recharts, React Router, React Query
│   ├── package-lock.json
│   ├── bun.lockb
│   ├── components.json             # shadcn/ui config
│   ├── vite.config.ts              # Vite dev server on :5173
│   ├── tailwind.config.ts          # Dark theme + SEAL accent colours
│   ├── tsconfig.json / tsconfig.app.json / tsconfig.node.json
│   ├── vitest.config.ts
│   ├── eslint.config.js
│   ├── postcss.config.js
│   ├── public/
│   │   ├── favicon.ico
│   │   ├── placeholder.svg
│   │   └── robots.txt
│   └── src/
│       ├── main.tsx                # React app entry
│       ├── App.tsx                 # BrowserRouter + QueryClientProvider + Toaster; all 5 routes
│       ├── App.css
│       ├── index.css               # Global CSS vars (#0f1117 bg, #1a1d27 card, SEAL colours)
│       ├── vite-env.d.ts
│       ├── lib/
│       │   ├── utils.ts            # cn() Tailwind merge helper
│       │   └── api.ts              # Typed REST wrappers — BASE_URL = http://localhost:8000
│       ├── hooks/
│       │   ├── use-mobile.tsx      # Mobile viewport hook (from Lovable)
│       │   ├── use-toast.ts        # Toast notification hook (from Lovable)
│       │   ├── useSimStream.ts     # WS /ws/simulate/{job_id} — drives SimCanvas animation
│       │   └── useTrainStream.ts   # WS /ws/train/{job_id} — appends episode data to chart
│       ├── components/
│       │   ├── DashboardLayout.tsx  # Sidebar + TopNav + content wrapper
│       │   ├── AppSidebar.tsx       # Collapsible sidebar (240px / icon-only)
│       │   ├── TopNav.tsx           # Fixed header: SEAL logo + nav links
│       │   ├── NavLink.tsx          # Active-state nav link wrapper
│       │   ├── ErrorBoundary.tsx     # React error boundary — catches render errors, shows fallback UI
│       │   ├── SimCanvas.tsx        # Canvas 2D renderer: roads, vehicles, TLS signals
│       │   └── ui/                  # 40+ shadcn/ui primitives (button, card, select, table, chart…)
│       ├── pages/
│       │   ├── Index.tsx            # / — hero, 4 stat cards, How It Works, CTAs
│       │   ├── Simulation.tsx       # /simulation — SimCanvas + useSimStream + controls + metrics
│       │   ├── Training.tsx         # /training — useTrainStream + Recharts chart + status bar
│       │   ├── Compare.tsx          # /compare — two SimCanvas + diff panel + dual-line chart
│       │   ├── Communication.tsx    # /communication — stacked bar + Pareto scatter + table
│       │   └── NotFound.tsx         # 404 page
│       └── test/
│           ├── example.test.ts
│           └── setup.ts
├── LovableOutput/                  # Lovable-generated UI scaffold — source for FrontEnd/
│   └── seal-traffic-flow-main/     # Root of the generated React project
│       ├── index.html
│       ├── package.json            # React 18, Vite, shadcn/ui, Recharts, React Router, React Query
│       ├── bun.lockb
│       ├── components.json         # shadcn/ui config
│       ├── vite.config.ts
│       ├── tailwind.config.ts      # Custom dark theme: #0f1117 bg, #1a1d27 card, SEAL accent colours
│       ├── tsconfig.json / tsconfig.app.json / tsconfig.node.json
│       ├── vitest.config.ts
│       ├── public/
│       └── src/
│           ├── main.tsx
│           ├── App.tsx             # BrowserRouter + QueryClientProvider + Toaster; all 5 routes
│           ├── index.css           # Global CSS vars: --background #0f1117, --card #1a1d27, SEAL colours
│           ├── lib/
│           │   └── utils.ts        # cn() Tailwind merge helper
│           ├── hooks/
│           │   ├── use-mobile.tsx
│           │   └── use-toast.ts
│           ├── components/
│           │   ├── DashboardLayout.tsx  # Sidebar + TopNav + content wrapper
│           │   ├── AppSidebar.tsx       # Collapsible sidebar (240px / icon-only)
│           │   ├── TopNav.tsx           # Fixed header: SEAL logo + nav links
│           │   ├── NavLink.tsx          # Active-state nav link wrapper
│           │   └── ui/                  # 40+ shadcn/ui primitives (button, card, select, table, chart…)
│           ├── pages/
│           │   ├── Index.tsx            # / — hero, 4 stat cards, How It Works, CTAs (all mock data)
│           │   ├── Simulation.tsx       # /simulation — controls, SVG grid placeholder, metrics sidebar, reward chart
│           │   ├── Training.tsx         # /training — multi-line Recharts chart, status bar, summary table
│           │   ├── Compare.tsx          # /compare — split canvas, diff panel, dual-line chart
│           │   ├── Communication.tsx    # /communication — stacked bar + Pareto scatter, explainer, raw table
│           │   └── NotFound.tsx
│           └── test/
├── SUMO-FedRL-main/                # Federated RL traffic light control system (SMARTCOMP 2022)
│   │
│   ├── train.py                    # Main training entry point — runs FedRL, MARL, or SARL trainers
│   ├── eval.py                     # Evaluation script — loads trained weights, runs trials, exports metrics
│   ├── clean_data.py               # Post-processes training output CSVs/JSONs
│   ├── netfiles.py                 # Defines file paths for road network configurations
│   ├── requirements.txt            # Dependencies: gym, numpy, ray, rllib, traci, stable-baselines, networkx
│   ├── reward.pdf                  # Reward function documentation
│   ├── .tmp-tripinfo.xml           # Temporary SUMO trip info output (auto-generated during simulation)
│   ├── Evaluation Tutorial.ipynb   # How-to guide for running evaluations
│   ├── Testing Eval.ipynb          # Testing the evaluation pipeline
│   │
│   ├── configs/                    # SUMO road network files (.net.xml)
│   │   ├── README.md               # Documents netgenerate commands to reproduce each network
│   │   ├── SMARTCOMP/              # Active configs used in SMARTCOMP submission
│   │   │   ├── grid-3x3.net.xml    # 3×3 intersection grid network
│   │   │   ├── grid-5x5.net.xml    # 5×5 intersection grid network
│   │   │   ├── grid-7x7.net.xml    # 7×7 intersection grid network
│   │   │   └── make_networks.sh    # Shell script to regenerate networks via netgenerate
│   │   └── ICCPS/                  # Older/deprecated network configs
│   │       ├── grid_2x2/ ... grid_9x9/   # Grid variants
│   │       ├── double_loop/        # Double loop intersection
│   │       ├── new_grid/           # Alternative grid layouts
│   │       ├── __old/              # Earlier development versions
│   │       └── dummy.net.xml       # Dummy test network
│   │
│   ├── example_weights/            # Pre-trained policy weights (.pkl) for evaluation
│   │   └── ICCPS/
│   │       ├── Final/              # Final submission weights
│   │       │   ├── FedRL/          # FedRL weights by aggregation function and topology
│   │       │   ├── MARL/           # MARL weights by topology
│   │       │   └── SARL/           # SARL weights by topology
│   │       ├── Old/                # Previous version weights
│   │       └── depr/               # Deprecated development weights
│   │       # Naming pattern: v{N}_{aggregation}_{ranked_type}.pkl
│   │
│   ├── notebooks/                  # Jupyter notebooks for analysis and plotting
│   │   ├── SMARTCOMP '22 - Training.ipynb      # Conference training runs
│   │   ├── SMARTCOMP '22 - Testing.ipynb       # Conference evaluation runs
│   │   ├── SMARTCOMP '22 - Aggregation Eval.ipynb  # Weight aggregation strategy comparison
│   │   ├── Experiments Analysis.ipynb          # General experiment analysis
│   │   ├── Experiments Analysis - v2.ipynb     # Improved analysis
│   │   ├── Training Analysis.ipynb             # Training progress visualization
│   │   ├── Real-World Experiment Analysis.ipynb # Real-world testbed results
│   │   ├── Data Transmission Sizes.ipynb       # Communication overhead analysis
│   │   ├── comm_data.py                        # Communication data extraction utility
│   │   └── results/                            # Generated PDF plots for conference papers
│   │
│   └── seal/                       # Core SEAL framework ("Smart Edge-enabled Adaptive Lights")
│       ├── __init__.py             # Exports TRIPINFO_OUT_FILENAME constant
│       ├── logging.py              # Logging configuration (INFO level, formatted output)
│       ├── core.py                 # (empty placeholder)
│       │
│       ├── sumo/                   # SUMO simulation environment
│       │   ├── env.py              # SumoEnv — multi-agent gym env; step(), reset(), reward
│       │   ├── abstract_env.py     # AbstractSumoEnv — base class; episode lifecycle, random routes
│       │   ├── config.py           # Constants: feature indices, MIN/MAX_DELAY, phase state defs
│       │   ├── timer.py            # ActionTimer — enforces min/max delays between phase changes
│       │   ├── README.md           # Env documentation: action space, yellow light transitions
│       │   │
│       │   ├── kernel/             # SUMO TraCI API wrapper
│       │   │   ├── kernel.py       # SumoKernel — launches SUMO, wraps TraCI, manages simulation
│       │   │   ├── const.py        # Traffic light state constants (r, y, g, G, s, u, o, O)
│       │   │   ├── __init__.py     # Module exports
│       │   │   ├── README.md       # Kernel documentation
│       │   │   └── trafficlight/   # Traffic light agent components
│       │   │       ├── light.py    # TrafficLight — individual agent; observations, phase management
│       │   │       ├── hub.py      # TrafficLightHub — stores all agents, builds adjacency graph
│       │   │       ├── space.py    # Creates gym Box observation space; scales features to [0, 1]
│       │   │       └── config.py   # Phase transition rules and state mappings
│       │   │
│       │   └── utils/              # Simulation utilities
│       │       ├── random_routes.py  # Generates vehicle trips (arcsine/uniform/zipf distributions)
│       │       ├── random_trips.py   # SUMO's built-in random trip generation utility
│       │       └── core.py           # Core helpers (e.g., node ID extraction)
│       │
│       └── trainer/                # Training framework (Ray RLLib / PPO)
│           ├── base.py             # BaseTrainer — abstract; checkpoints, PPO setup, training loop
│           ├── fed_agent.py        # FedPolicyTrainer — FedRL with federated averaging
│           ├── multi_agent.py      # MultiPolicyTrainer — MARL with independent per-agent policies
│           ├── single_agent.py     # SinglePolicyTrainer — SARL with one shared policy
│           ├── weight_aggr.py      # Aggregation functions: naive, pos_reward, neg_reward, traffic
│           ├── defaults.py         # Default hyperparameters (GUI, ranked obs, random routes)
│           ├── util.py             # Policy mapping helpers, env config builder
│           ├── counter.py          # Persists run counts to avoid experiment name collisions
│           │
│           ├── communication/      # Communication cost tracking (Ray callbacks)
│           │   ├── __init__.py     # Comm type constants (EDGE2TLS, TLS2EDGE, VEH2TLS, etc.)
│           │   ├── base_callback.py   # BaseCommCallback — tracks comm costs per episode
│           │   ├── fed_callback.py    # FedRL comm callback (edge↔TLS policy, rank, obs, vehicles)
│           │   ├── multi_callback.py  # MARL comm callback
│           │   └── single_callback.py # SARL comm callback
│           │
│           └── data/               # Training result parsing
│               └── parser.py       # DataParser — extracts rewards, comm costs, vehicle counts
```

---

## Architecture Overview

### Training Flow
```
train.py
  └─> Trainer (FedRL / MARL / SARL)
        └─> Ray RLLib PPOTrainer
              └─> SumoEnv  (multi-agent gym environment)
                    └─> SumoKernel  (TraCI — controls SUMO)
                          └─> TrafficLightHub
                                └─> TrafficLight agents
                                      └─> Observations → Actions → Rewards
```

### Evaluation Flow
```
eval.py
  └─> Load .pkl weights
        └─> Run MC trials on test network
              └─> Export rewards, features, trip metrics to CSV
```

### Trainer Comparison

| Trainer | Class | Description |
|---------|-------|-------------|
| **FedRL** | `FedPolicyTrainer` | Independent agents + periodic federated averaging; configurable weighting |
| **MARL** | `MultiPolicyTrainer` | Independent per-agent policies; naive gradient averaging at end |
| **SARL** | `SinglePolicyTrainer` | Single shared policy across all traffic lights |

### FedRL Aggregation Functions

| Function | Weights by |
|----------|------------|
| `naive` | Equal weight (1/n) for all policies |
| `pos_reward` | Proportional to episode reward |
| `neg_reward` | Inversely proportional to reward |
| `traffic` | Proportional to vehicle count |

---

## Author's Research Report: What SUMO-FedRL Is Trying to Do

**Paper:** Hudson, N., Oza, P., Khamfroush, H., & Chantem, T. (2022). *Smart Edge-Enabled Traffic Light Control: Improving Reward-Communication Trade-offs with Federated Reinforcement Learning.* IEEE SMARTCOMP 2022, pp. 40–47.

---

### 1. The Problem

Urban traffic congestion is a well-established challenge, and adaptive traffic light control via reinforcement learning (RL) is a known solution. However, a critical and often ignored tension exists in real deployments:

- **To learn well**, RL agents (traffic lights) need to share information — observations, policies, gradients — with an edge server that aggregates them.
- **Sharing information has a cost** — bandwidth, latency, energy — especially when agents are edge devices (embedded controllers at intersections) sending raw sensor data (video, LIDAR) to the cloud.
- **Naive multi-agent RL (MARL)** lets agents share freely, which produces good rewards but incurs high communication overhead.
- **Naive single-agent RL (SARL)** uses one global policy with minimal communication, but cannot adapt to heterogeneous local traffic conditions.

The author's core question: **Can federated reinforcement learning find a middle ground — achieving high reward (low congestion) while keeping communication costs low?**

The economic motivation is explicit: traffic congestion has enormous real-world costs. Any system that can reduce it while avoiding the prohibitive data transmission costs of centralised cloud AI is directly deployable on real smart city infrastructure.

---

### 2. The Proposed Solution: SEAL (Smart Edge-Enabled Adaptive Lights)

The authors propose a system where each traffic light is an independent RL agent that:

1. **Learns its own policy locally** using Proximal Policy Optimization (PPO) via Ray RLLib — no raw sensor data leaves the intersection.
2. **Periodically synchronises** only its learned policy weights with a nearby **edge server** via **Federated Averaging (FedAvg)** — far cheaper than sending raw video or LIDAR streams.
3. **Uses configurable aggregation weighting** so that more "useful" intersections (e.g., high-traffic, high-performing) contribute more to the shared model.

A key design requirement is that the SEAL model is **intersection-agnostic**: the same observation space, action space, and reward function work identically for a 2-lane intersection and an 8-lane intersection, with no architectural changes. This is what makes the system deployable across heterogeneous real-world networks.

This is Federated Learning applied to a continuous-control RL problem over a road network — hence **FedRL**.

---

### 3. The Environment

Each simulation episode is a SUMO road network run:

- **Networks:** Grid topologies of 3×3, 5×5, and 7×7 intersections.
- **Vehicles:** Generated randomly each episode using configurable traffic distributions (arcsine, uniform, zipf), at a rate defined by vehicles-per-lane-per-hour.
- **Agents:** Each traffic light is an independent RL agent with its own observation and action.

**Observation space** (per traffic light, 10–14 features):

| Feature | Description |
|---------|-------------|
| Lane occupancy | Fraction of controlled lane length filled by vehicles |
| Halted lane occupancy | Fraction filled by stopped vehicles (speed < 0.1 m/s) |
| Speed ratio | Average vehicle speed / speed limit across controlled lanes |
| Phase state proportions | Fraction of signal heads in each state (r, y, g, G, u, o, O) |
| Local rank *(ranked mode)* | How congested this intersection is vs. its immediate neighbours |
| Global rank *(ranked mode)* | How congested this intersection is vs. all intersections |
| Local halt rank *(ranked mode)* | Same as local rank, but for halted vehicles |
| Global halt rank *(ranked mode)* | Same as global rank, but for halted vehicles |

**Action space:** Binary per traffic light — `0` = hold current phase, `1` = advance to next phase. Phase changes are governed by enforced minimum (4 s) and maximum (120 s) delays.

**Reward function:**
```
reward = -1 × (lane_occupancy + halted_lane_occupancy)²
```
This is a purely negative signal — agents are penalised for congestion. The goal is to minimise it (i.e., push toward zero).

---

### 4. The Three Approaches Compared

The paper benchmarks three training paradigms on the same environment to isolate the effect of the federation strategy:

| Approach | Communication pattern | Policy sharing |
|----------|----------------------|---------------|
| **SARL** (baseline low-comm) | None during training; single policy for all lights | One global policy updated by all agents simultaneously |
| **MARL** (baseline high-comm) | Gradients shared every step via RLLib | Each light has its own policy, aggregated naively at the end |
| **FedRL** (proposed) | Policy weights shared every `fed_step` episodes | Each light has its own policy; weights periodically averaged |

---

### 5. The Aggregation Mechanism (FedAvg with Weighted Coefficients)

The FedRL trainer's key innovation is **how it weights each agent's policy** when performing the federated average. Rather than giving all agents equal weight, it proposes four strategies:

- **Naive:** All agents weighted equally → `coeff = 1/n`. Used as a baseline and noted in the code as the best-performing strategy in the publication.
- **Positive reward weighting:** Agents that achieved higher reward contribute more → `coeff = reward_i / Σ reward`. Rewards well-performing intersections.
- **Negative reward weighting:** Agents with lower reward contribute more → inversely weighted. Intended to up-weight struggling intersections, but noted as performing poorly.
- **Traffic weighting:** Agents controlling more vehicles contribute more → `coeff = vehicles_i / Σ vehicles`. Experimental; gives busier intersections more influence.

After every `fed_step` episodes, the FedAvg routine:
1. Retrieves each agent's current neural network weights.
2. Computes the weighted average across all parameter tensors.
3. Pushes the new shared weights back to every agent via the edge server.
4. Resets the reward/vehicle trackers for the next federation window.

> **Paper vs. code note:** The paper describes aggregation as occurring every ~4000 timesteps. In the codebase, `train.py` sets `fed_step=1` with a horizon of 360 steps per episode — meaning aggregation happens every episode (~360 timesteps). The `fed_step` parameter controls the episode interval and can be tuned to match any timestep target.

---

### 6. The Ranked Observation Innovation

A secondary contribution is the **ranked observation mode**. Beyond local sensor data, each traffic light also receives:

- Its **global rank** (normalised position in the sorted congestion order across the entire network).
- Its **local rank** (normalised position relative to directly adjacent intersections only).

These rankings are computed centrally each step and distributed to agents, introducing a small but deliberate **communication cost** in exchange for better situational awareness. The paper tests both ranked and unranked modes to quantify this trade-off.

---

### 7. What the Authors Are Measuring

The paper evaluates the trade-off across three axes:

1. **Reward** — mean episode reward per traffic light (higher = less congestion).
2. **Communication cost** — number of messages exchanged between components per episode, broken down by type:
   - Edge server → Traffic light: policy weights, action commands, rank values
   - Traffic light → Edge server: observations
   - Vehicle → Traffic light: vehicle count queries
3. **SUMO trip metrics** — travel time, waiting time, time loss per vehicle (from SUMO's `tripinfo` output).

**Published results (from the paper):**

| Metric | FedRL vs. Centralised MARL |
|--------|---------------------------|
| Communication cost | **36.24% reduction** |
| Average reward | **2.11% decrease** (negligible) |
| Travel time | **18.14% improvement** over standard fixed-timing lights |

FedRL with naive weighting achieves reward nearly identical to MARL while cutting communication overhead by over a third — and both outperform standard timed lights on travel time by ~18%. This demonstrates FedRL sits at the Pareto-efficient frontier of the reward-communication trade-off.

---

### 8. Experimental Scale

| Axis | Values tested |
|------|--------------|
| Network topologies | 3×3, 5×5, 7×7 grids |
| Trainer types | SARL, MARL, FedRL (×4 aggregation functions) |
| Observation modes | Ranked, Unranked |
| Training episodes | 50 per configuration |
| Evaluation | Multiple Monte Carlo trials per trained policy |

This yields a large combinatorial matrix of experiments, all automated through `train.py` and post-processed via the notebooks in `/notebooks/`.

---

### 9. Summary

The author is trying to show that **federated reinforcement learning is a principled and practical way to deploy adaptive traffic light control on edge hardware** — where communication is constrained — without sacrificing the traffic throughput benefits that centralised MARL approaches provide. The SEAL framework is the software artifact that implements, evaluates, and quantifies this claim across varied urban grid topologies.
